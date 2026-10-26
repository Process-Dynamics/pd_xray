// Layout and top-level state: file browser | viewer + frame slider | side panel, with the status bar below.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  ApiError,
  type DatasetInfo,
  DEFAULT_TONE,
  type Display,
  type Job,
  type Operation,
  type Preview,
  type ServerInfo,
} from "./api";
import { DisplayPanel } from "./DisplayPanel";
import { connectJobEvents } from "./events";
import { FileBrowser, type OpenRequest } from "./FileBrowser";
import { FrameSlider } from "./FrameSlider";
import { PipelineActions } from "./PipelineActions";
import { type EditableStep, editableStep, newStep, ProcessingPanel } from "./ProcessingPanel";
import { clampFrame } from "./scheduling";
import { StatusBar } from "./StatusBar";
import { type View, Viewer } from "./Viewer";

const FINISHED_JOB_SHOWN_MS = 8000;
const CLIENT_ID = crypto.randomUUID(); // this tab, so the server can cancel its outdated previews

export function App() {
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [frame, setFrame] = useState<number[]>([]);
  const [display, setDisplay] = useState<Display>({
    vmin: 0,
    vmax: 1,
    cmap: "grey",
    format: "png",
    tone: DEFAULT_TONE,
  });
  const [shown, setShown] = useState<View | null>(null);
  const [viewerLoading, setViewerLoading] = useState<string | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [steps, setSteps] = useState<EditableStep[]>([]);
  const [showProcessed, setShowProcessed] = useState(false);
  const [processedRange, setProcessedRange] = useState<{ vmin: number; vmax: number } | null>(null); // null: auto
  const [preview, setPreview] = useState<{ steps: string; result: Preview } | null>(null); // latest preview + its steps
  const [tasks, setTasks] = useState<{ id: number; label: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [recentJobs, setRecentJobs] = useState<Set<string>>(new Set());
  const [rangeJob, setRangeJob] = useState<{ jobId: string; datasetId: string } | null>(null);
  const nextTask = useRef(0);
  const watchedJobs = useRef(new Set<string>()); // jobs this page saw queued or running
  const viewerErrorShown = useRef(false); // the error on display came from the viewer: clear it once a view loads

  /** Show `label` in the status bar while `work` runs; report failures there. Resolves to undefined on failure. */
  const track = useCallback(async <T,>(label: string, work: Promise<T>): Promise<T | undefined> => {
    const id = nextTask.current++;
    setTasks((t) => [...t, { id, label }]);
    try {
      return await work;
    } catch (e) {
      viewerErrorShown.current = false;
      setError(e instanceof ApiError ? e.message : `Unexpected error: ${String(e)}`);
      console.error(e);
      return undefined;
    } finally {
      setTasks((t) => t.filter((task) => task.id !== id));
    }
  }, []);

  useEffect(() => {
    void track("Connecting", api.info()).then((result) => result && setInfo(result));
    void track("Loading operations", api.operations()).then((result) => result && setOperations(result));
    return connectJobEvents((job) => {
      setJobs((all) => ({ ...all, [job.id]: job }));
      if (job.state === "queued" || job.state === "running") {
        watchedJobs.current.add(job.id);
      } else if (watchedJobs.current.delete(job.id)) {
        // Finished while this page watched it. Jobs that finished earlier arrive on connect and stay hidden.
        setRecentJobs((r) => new Set(r).add(job.id));
        window.setTimeout(
          () =>
            setRecentJobs((r) => {
              const next = new Set(r);
              next.delete(job.id);
              return next;
            }),
          FINISHED_JOB_SHOWN_MS,
        );
        if (job.state === "failed" && job.error) setError(`${job.title}: ${job.error}`);
      }
    }, setConnected);
  }, [track]);

  // Apply a finished "dataset range" job to the contrast, if it belongs to the dataset still open.
  useEffect(() => {
    const job = rangeJob && jobs[rangeJob.jobId];
    if (!job || !rangeJob || job.state !== "done") return;
    const result = job.result as { min: number; max: number };
    if (dataset?.id === rangeJob.datasetId) setDisplay((d) => ({ ...d, vmin: result.min, vmax: result.max }));
    setRangeJob(null);
  }, [jobs, rangeJob, dataset]);

  const open = useCallback(
    async (request: OpenRequest) => {
      const options = { dataset: request.dataset, pattern: request.pattern, connection: request.connection };
      const opened = await track(`Opening ${request.label}`, api.open(request.path, options));
      if (!opened) return;
      const first = opened.frame_shape.map(() => 0);
      const stats = await track("Reading the first frame", api.stats(opened.id, first));
      if (!stats) return void api.close(opened.id).catch(() => undefined);
      if (dataset) void api.close(dataset.id).catch(() => undefined); // frees the server's cache
      setDataset(opened);
      setFrame(first);
      setDisplay((d) => ({ ...d, vmin: stats.low, vmax: stats.high }));
      setError(null);
    },
    [dataset, track],
  );

  /**
   * After a viewer error or a reconnect: if the server no longer has the dataset open (e.g. it restarted), reopen the
   * same file with the same options and keep the frame and display settings. Otherwise show `message`, if any.
   */
  const recover = useCallback(
    async (message?: string) => {
      if (!dataset) return message && setError(message);
      const gone = await api.dataset(dataset.id).then(
        () => false,
        (e: unknown) => e instanceof ApiError && e.status === 404,
      );
      if (!gone) return message && setError(message);
      const label = `Reopening ${dataset.name} (the server had closed it)`;
      const reopened = await track(label, api.open(dataset.path, dataset.options));
      if (!reopened) return;
      setDataset(reopened);
      setFrame((f) => clampFrame(f, reopened.frame_shape));
      setError(null);
    },
    [dataset, track],
  );

  useEffect(() => {
    if (connected) void recover();
  }, [connected]); // only on (re)connect, not when `recover` changes

  const enabledSteps = steps.filter((s) => s.enabled);
  const processing = showProcessed && enabledSteps.length > 0;
  const apiSteps = useMemo(() => steps.map(({ op, params, enabled }) => ({ op, params, enabled })), [steps]);
  const currentPreview = preview?.steps === JSON.stringify(apiSteps) ? preview.result : null;

  // A different set or order of steps usually changes the value range: go back to automatic contrast.
  const structure = steps.map((s) => `${s.op}:${s.enabled}`).join(",");
  const keepRange = useRef(false); // set when a change of steps chooses the processed contrast itself
  useEffect(() => {
    if (!keepRange.current) setProcessedRange(null);
    keepRange.current = false;
  }, [structure]);

  // P switches between the original and processed view (except while typing).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.key !== "p" || event.ctrlKey || event.metaKey || event.altKey) return;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName) && (target as HTMLInputElement).type !== "range")
        return;
      setShowProcessed((show) => !show);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const autoContrast = async (mode: "auto" | "full") => {
    if (!dataset) return;
    const pipeline = processing ? currentPreview?.pipeline : undefined;
    if (processing && !pipeline) return;
    const stats = await track("Computing contrast", api.stats(dataset.id, frame, pipeline));
    if (!stats) return;
    const [vmin, vmax] = mode === "auto" ? [stats.low, stats.high] : [stats.min, stats.max];
    if (processing) setProcessedRange({ vmin, vmax });
    else setDisplay((d) => ({ ...d, vmin, vmax }));
  };

  const datasetRange = async () => {
    if (!dataset) return;
    const job = await track("Starting job", api.valueRange(dataset.id));
    if (job) {
      setJobs((all) => ({ ...all, [job.id]: job }));
      setRangeJob({ jobId: job.id, datasetId: dataset.id });
    }
  };

  const view = useMemo<View | null>(() => {
    if (!dataset) return null;
    if (!processing) return { dataset, frame, display, processing: null };
    const range = processedRange ?? { vmin: null, vmax: null };
    return { dataset, frame, display: { ...display, ...range }, processing: { steps: apiSteps, client: CLIENT_ID } };
  }, [dataset, frame, display, processing, processedRange, apiSteps]);
  // The contrast the panel edits: the original's, or the processed view's (until set: the latest preview's auto range).
  const lastStats = (currentPreview ?? preview?.result)?.stats;
  const activeDisplay: Display = processing
    ? {
        ...display,
        ...(processedRange ?? { vmin: lastStats?.low ?? 0, vmax: lastStats?.high ?? 1 }),
      }
    : display;
  const onDisplay = (next: Display) => {
    if (processing) {
      setProcessedRange({ vmin: next.vmin, vmax: next.vmax });
      setDisplay((d) => ({ ...d, cmap: next.cmap, format: next.format, tone: next.tone }));
    } else {
      setDisplay(next);
    }
  };
  // The display tone becomes the last step, with the current Min/Max as its black/white points. Its output stays in
  // that range, so showing it with the same Min/Max (and no display tone) looks the same.
  const toneOperation = operations.find((op) => op.name === "tone");
  const addToneStep = () => {
    if (!toneOperation) return;
    const step = newStep(toneOperation);
    step.params = { ...step.params, black: activeDisplay.vmin, white: activeDisplay.vmax, ...display.tone };
    keepRange.current = true;
    setSteps([...steps, step]);
    setProcessedRange({ vmin: activeDisplay.vmin, vmax: activeDisplay.vmax });
    setDisplay((d) => ({ ...d, tone: DEFAULT_TONE }));
    setShowProcessed(true);
  };
  // The histogram shows the frame on screen (for a processed view, once its preview is known).
  const shownPipeline =
    shown?.processing && preview?.steps === JSON.stringify(shown.processing.steps)
      ? preview.result.pipeline
      : undefined;
  const histogramSource =
    shown && dataset && shown.dataset.id === dataset.id && (!shown.processing || shownPipeline)
      ? { datasetId: dataset.id, frame: shown.frame, pipeline: shownPipeline }
      : null;
  const datasetFolder = dataset
    ? dataset.options.pattern
      ? dataset.path
      : dataset.path.slice(0, dataset.path.lastIndexOf("/")) || "/"
    : "";
  // Defaults for a run: the processed frame size, and the processed view's contrast (or the preview's automatic one).
  const lastPreview = currentPreview ?? preview?.result;
  const processedDefaults = lastPreview
    ? {
        height: lastPreview.height,
        width: lastPreview.width,
        ...(processing
          ? { vmin: activeDisplay.vmin, vmax: activeDisplay.vmax }
          : { vmin: lastPreview.stats.low, vmax: lastPreview.stats.high }),
      }
    : null;
  const busy = [...tasks.map((t) => t.label), ...(viewerLoading ? [viewerLoading] : [])];
  const visibleJobs = Object.values(jobs).filter(
    (j) => j.state === "queued" || j.state === "running" || recentJobs.has(j.id),
  );

  return (
    <div className="app">
      <FileBrowser
        track={track}
        onOpen={(r) => void open(r)}
        opened={dataset ? { path: dataset.path, connection: dataset.options.connection } : null}
        onDisconnected={(id) => {
          if (dataset?.options.connection === id) setDataset(null);
        }}
      />
      <main className="main">
        <Viewer
          view={view}
          onShown={(v) => {
            setShown(v);
            if (viewerErrorShown.current) setError(null);
            viewerErrorShown.current = false;
          }}
          onLoading={setViewerLoading}
          onPreview={(v, result) => setPreview({ steps: JSON.stringify(v.processing?.steps), result })}
          onError={(m) => {
            viewerErrorShown.current = true;
            void recover(m);
          }}
        />
        {dataset && (
          <FrameSlider
            axes={dataset.frame_axes}
            shape={dataset.frame_shape}
            frame={frame}
            shownFrame={shown?.dataset.id === dataset.id ? shown.frame : null}
            onChange={setFrame}
          />
        )}
      </main>
      <aside className="side">
        {dataset ? (
          <div className="panel">
            <DisplayPanel
              dataset={dataset}
              display={activeDisplay}
              processed={processing}
              colormaps={info?.colormaps ?? ["grey"]}
              rangeJob={rangeJob ? (jobs[rangeJob.jobId] ?? null) : null}
              histogramSource={histogramSource}
              toneParams={(toneOperation?.params ?? []).filter((p) => p.name !== "black" && p.name !== "white")}
              onDisplay={onDisplay}
              onToneStep={addToneStep}
              onAutoContrast={(mode) => void autoContrast(mode)}
              onDatasetRange={() => void datasetRange()}
            >
              <ProcessingPanel
                operations={operations}
                steps={steps}
                showProcessed={processing}
                preview={processing ? currentPreview : null}
                datasetFolder={datasetFolder}
                connection={dataset.options.connection}
                displayRange={activeDisplay}
                onSteps={(next) => {
                  setSteps(next);
                  if (next.some((s) => s.enabled) && !steps.some((s) => s.enabled)) setShowProcessed(true);
                }}
                onShowProcessed={setShowProcessed}
              />
              <PipelineActions
                dataset={dataset}
                datasetFolder={dataset.options.connection ? null : datasetFolder}
                steps={apiSteps}
                processed={processedDefaults}
                onLoaded={(loaded) => {
                  setSteps(loaded.map(editableStep));
                  setShowProcessed(loaded.some((s) => s.enabled));
                }}
                onJob={(job) => setJobs((all) => ({ ...all, [job.id]: job }))}
              />
            </DisplayPanel>
          </div>
        ) : (
          <div className="panel muted">No dataset open.</div>
        )}
      </aside>
      <StatusBar
        connected={connected}
        busy={busy}
        jobs={visibleJobs}
        error={error}
        onCancel={(job) => void track("Cancelling", api.cancelJob(job.id))}
        onDismissError={() => setError(null)}
      />
    </div>
  );
}
