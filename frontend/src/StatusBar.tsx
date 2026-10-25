// Bottom bar: server connection, what is loading, background jobs (progress, ETA, cancel) and the last error.

import { useEffect, useState } from "react";

import type { Job } from "./api";
import { formatDuration } from "./format";

interface Props {
  connected: boolean;
  busy: string[];
  jobs: Job[];
  error: string | null;
  onCancel: (job: Job) => void;
  onDismissError: () => void;
}

/** Seconds since `label` last changed, updated twice a second while it is set. */
function useElapsed(label: string | null): number {
  const [since, setSince] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setSince(Date.now());
    setNow(Date.now());
    if (label === null) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [label]);
  return (now - since) / 1000;
}

export function StatusBar({ connected, busy, jobs, error, onCancel, onDismissError }: Props) {
  const current = busy.length > 0 ? busy[busy.length - 1]! : null;
  const elapsed = useElapsed(current);
  return (
    <div className="status">
      <span
        className={`status-dot ${connected ? "ok" : "bad"}`}
        title={connected ? "Connected to the server" : "Not connected to the server; retrying"}
      />
      {!connected && <span className="status-warning">Reconnecting…</span>}
      <span className="status-busy">
        {current === null ? "Ready" : `${current}…${elapsed >= 1 ? ` ${Math.floor(elapsed)} s` : ""}`}
      </span>
      <span className="status-jobs">
        {jobs.map((job) => (
          <span key={job.id} className={`job job-${job.state}`} title={job.error ?? job.message}>
            <span className="job-title">{job.title}</span>
            {job.state === "running" && (
              <>
                <progress max={1} value={job.fraction ?? undefined} />
                <span>
                  {job.total ? `${job.current}/${job.total}` : job.current}{" "}
                  {job.eta_seconds !== null && `· ${formatDuration(job.eta_seconds)} left`}
                </span>
              </>
            )}
            {job.state !== "running" && <span className="job-state">{job.state}</span>}
            {(job.state === "running" || job.state === "queued") && (
              <button onClick={() => onCancel(job)} title="Cancel this job">
                ×
              </button>
            )}
          </span>
        ))}
      </span>
      {error && (
        <span className="status-error" title={error}>
          {error}
          <button onClick={onDismissError} title="Dismiss">
            ×
          </button>
        </span>
      )}
    </div>
  );
}
