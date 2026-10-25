"""Background jobs on a thread pool, with progress, ETA and cancellation.

A job is a function `work(progress, cancel)` taking a `ProgressCallback` and a `CancelToken`, like the library
functions it usually wraps. Listeners (the WebSocket) receive job snapshots on every state change and on progress,
throttled to about 10 per second per job.
"""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from pd_xray.core import CancelToken, OperationCancelledError, Progress, ProgressCallback
from pd_xray.server.datasets import NotFoundError, jsonable

logger = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "failed", "cancelled"]
Work = Callable[[ProgressCallback, CancelToken], Any]
Listener = Callable[[dict[str, Any]], None]

_NOTIFY_INTERVAL = 0.1  # seconds between progress notifications per job
_FINISHED = ("done", "failed", "cancelled")


@dataclass
class Job:
    """A background job. Fields change only under the manager's lock.

    Attributes:
        result: Return value of the work function (JSON-compatible), once done.
        error: Error message, once failed.
    """

    id: str
    title: str
    state: JobState = "queued"
    current: int = 0
    total: int | None = None
    message: str = ""
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    last_notified: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        """JSON description for the UI, including `fraction` and `eta_seconds` (None if unknown)."""
        fraction = Progress(self.current, self.total).fraction
        eta = None
        if self.state == "running" and self.started is not None and fraction:
            elapsed = time.time() - self.started
            eta = elapsed * (1 - fraction) / fraction
        return {
            "id": self.id,
            "title": self.title,
            "state": self.state,
            "current": self.current,
            "total": self.total,
            "fraction": fraction,
            "message": self.message,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "eta_seconds": eta,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """Runs jobs on a thread pool and keeps the most recent `keep` jobs for display.

    Args:
        workers: Jobs that may run at once. Others wait as "queued".
        keep: Finished jobs to keep; older finished jobs are dropped.
    """

    def __init__(self, workers: int, keep: int = 100) -> None:
        self._executor = ThreadPoolExecutor(workers, thread_name_prefix="pd-xray-job")
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._listeners: list[Listener] = []
        self._keep = keep
        self._lock = threading.Lock()

    def submit(self, title: str, work: Work) -> dict[str, Any]:
        """Queue `work` and return the new job's snapshot."""
        job = Job(uuid.uuid4().hex[:12], title)
        with self._lock:
            self._jobs[job.id] = job
            finished = [j.id for j in self._jobs.values() if j.state in _FINISHED]
            for job_id in finished[: max(0, len(finished) - self._keep)]:
                del self._jobs[job_id]
        self._notify(job)
        self._executor.submit(self._run, job, work)
        return self._snapshot(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop. A queued job is cancelled at once; a running one stops at its next check.

        Raises:
            NotFoundError: For an unknown job id.
        """
        job = self._get(job_id)
        job.cancel_token.cancel()
        with self._lock:
            if job.state == "queued":
                job.state, job.finished = "cancelled", time.time()
        self._notify(job)
        return self._snapshot(job)

    def get(self, job_id: str) -> dict[str, Any]:
        """Snapshot of one job.

        Raises:
            NotFoundError: For an unknown job id.
        """
        return self._snapshot(self._get(job_id))

    def list(self) -> list[dict[str, Any]]:
        """Snapshots of all kept jobs, oldest first."""
        with self._lock:
            return [job.snapshot() for job in self._jobs.values()]

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Call `listener(snapshot)` on job updates, from worker threads. Returns a function that unsubscribes."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def shutdown(self) -> None:
        """Cancel all jobs and stop the pool without waiting for running work."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel_token.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"No job with id '{job_id}'.")
        return job

    def _snapshot(self, job: Job) -> dict[str, Any]:
        with self._lock:
            return job.snapshot()

    def _notify(self, job: Job) -> None:
        with self._lock:
            job.last_notified = time.monotonic()
            snapshot = job.snapshot()
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:
                logger.exception("Job listener failed")

    def _progress(self, job: Job, progress: Progress) -> None:
        with self._lock:
            job.current, job.total = progress.current, progress.total
            job.message = progress.message or job.message
            due = time.monotonic() - job.last_notified >= _NOTIFY_INTERVAL or progress.current == progress.total
        if due:
            self._notify(job)

    def _run(self, job: Job, work: Work) -> None:
        with self._lock:
            if job.state != "queued":  # cancelled while queued
                return
            job.state, job.started = "running", time.time()
        self._notify(job)
        try:
            result = work(lambda progress: self._progress(job, progress), job.cancel_token)
            state: JobState = "done"
            error = None
        except OperationCancelledError:
            state, result, error = "cancelled", None, None
        except Exception as exc:
            logger.exception("Job '%s' failed", job.title)
            state, result, error = "failed", None, str(exc) or type(exc).__name__
        with self._lock:
            job.state, job.result, job.error, job.finished = state, jsonable(result), error, time.time()
        self._notify(job)
