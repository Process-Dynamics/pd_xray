// Load and save the processing steps as a pipeline file, and run them on the whole dataset as a background job.

import { useState } from "react";

import { api, ApiError, type DatasetInfo, type Job, type OutputType, type Step } from "./api";
import { formatBytes, formatValue } from "./format";
import { FileDialog } from "./FileDialog";

interface Props {
  dataset: DatasetInfo;
  datasetFolder: string;
  steps: Step[];
  /** Processed frame size and contrast from the latest preview, for defaults; null if not previewed yet. */
  processed: { height: number; width: number; vmin: number; vmax: number } | null;
  onLoaded: (steps: Step[]) => void;
  onJob: (job: Job) => void;
}

const FORMATS = [
  { extension: ".tif", label: "TIFF (multi-page)", types: ["float32", "uint16", "uint8"] },
  { extension: ".h5", label: "HDF5", types: ["float32", "uint16", "uint8"] },
  { extension: ".png", label: "PNG (one file per frame)", types: ["uint16", "uint8"] },
  { extension: ".jpg", label: "JPEG (one file per frame, lossy)", types: ["uint8"] },
] as const;
const TYPE_LABELS: Record<OutputType, string> = {
  float32: "float32 (exact)",
  uint16: "uint16 (scaled)",
  uint8: "uint8 (scaled)",
};
const BYTES: Record<OutputType, number> = { float32: 4, uint16: 2, uint8: 1 };

/** Runs `save(overwrite)`; if the file exists (409), asks before replacing it. */
async function saveAsking(save: (overwrite: boolean) => Promise<unknown>): Promise<void> {
  try {
    await save(false);
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 409) || !window.confirm(`${e.message}\nReplace it?`)) throw e;
    await save(true);
  }
}

const stem = (name: string) => (name.includes(".") ? name.slice(0, name.lastIndexOf(".")) : name);
const withExtension = (name: string, extension: string) =>
  FORMATS.some((f) => name.toLowerCase().endsWith(f.extension)) ? stem(name) + extension : name + extension;
const numberOrNull = (text: string) => (text.trim() === "" || !Number.isFinite(Number(text)) ? null : Number(text));

function RunDialog({ dataset, datasetFolder, steps, processed, onJob, onClose }: Props & { onClose: () => void }) {
  const [name, setName] = useState(`${stem(dataset.name)}_processed.tif`);
  const [dtype, setDtype] = useState<OutputType>("float32");
  const [low, setLow] = useState(processed ? formatValue(processed.vmin) : "");
  const [high, setHigh] = useState(processed ? formatValue(processed.vmax) : "");
  const count = dataset.frame_shape[0] ?? 1;
  const [from, setFrom] = useState("0");
  const [to, setTo] = useState(String(count - 1));
  const [every, setEvery] = useState("1");

  const format = FORMATS.find((f) => name.toLowerCase().endsWith(f.extension)) ?? FORMATS[0];
  const types: readonly OutputType[] = format.types;
  const type = types.includes(dtype) ? dtype : types[0]!;
  const start = Math.max(0, Math.round(numberOrNull(from) ?? 0));
  const stop = Math.min(count - 1, Math.round(numberOrNull(to) ?? count - 1));
  const step = Math.max(1, Math.round(numberOrNull(every) ?? 1));
  const selected = stop >= start ? Math.floor((stop - start) / step) + 1 : 0;
  const all = start === 0 && stop === count - 1 && step === 1;
  const perFrame = dataset.frame_shape.slice(1).reduce((a, b) => a * b, 1) * dataset.channels;
  const size = processed && selected * perFrame * processed.height * processed.width * BYTES[type];
  const range: [number, number] | null =
    numberOrNull(low) !== null && numberOrNull(high) !== null ? [numberOrNull(low)!, numberOrNull(high)!] : null;

  const run = async (path: string) => {
    if (selected === 0) throw new Error("No frames selected.");
    if (range && range[1] <= range[0]) throw new Error("Max must be above Min.");
    const body = {
      dataset_id: dataset.id,
      steps,
      destination: path,
      dtype: type,
      value_range: type === "float32" ? null : range,
      frames: all || !dataset.frame_shape.length ? null : ([start, stop + 1, step] as [number, number, number]),
      overwrite: false,
    };
    let job: Job | null = null;
    await saveAsking(async (overwrite) => (job = await api.run({ ...body, overwrite })));
    if (job) onJob(job);
    onClose();
  };

  return (
    <FileDialog
      title={`Process ${dataset.name} into…`}
      mode="save"
      start={datasetFolder}
      extensions={FORMATS.map((f) => f.extension)}
      name={name}
      onName={setName}
      confirmLabel="Run"
      onConfirm={run}
      onCancel={onClose}
    >
      <div className="field-grid run-options">
        <span>Format</span>
        <select value={format.extension} onChange={(e) => setName(withExtension(name, e.target.value))}>
          {FORMATS.map((f) => (
            <option key={f.extension} value={f.extension}>
              {f.label}
            </option>
          ))}
        </select>
        <span>Values</span>
        <select value={type} onChange={(e) => setDtype(e.target.value as OutputType)}>
          {types.map((t) => (
            <option key={t} value={t}>
              {TYPE_LABELS[t]}
            </option>
          ))}
        </select>
        {type !== "float32" && (
          <>
            <span>Range</span>
            <span
              className="run-range"
              title="Processed values from Min to Max become 0 to the type's maximum; values outside are clipped. Leave empty to use the processed data's own range (processes every frame twice)."
            >
              <input value={low} onChange={(e) => setLow(e.target.value)} placeholder="min" />
              <input value={high} onChange={(e) => setHigh(e.target.value)} placeholder="max" />
            </span>
          </>
        )}
        {dataset.frame_shape.length > 0 && (
          <>
            <span>Frames</span>
            <span className="run-range" title={`Along ${dataset.frame_axes[0]}: from, to (inclusive), every n-th`}>
              <input value={from} onChange={(e) => setFrom(e.target.value)} />
              <input value={to} onChange={(e) => setTo(e.target.value)} />
              <input value={every} onChange={(e) => setEvery(e.target.value)} />
            </span>
          </>
        )}
      </div>
      <div className="muted">
        {selected} of {count} frames
        {size ? ` · output about ${formatBytes(size)}` : ""}. Also writes {name}.pipeline.yaml next to it.
      </div>
    </FileDialog>
  );
}

export function PipelineActions(props: Props) {
  const { dataset, datasetFolder, steps } = props;
  const [dialog, setDialog] = useState<"load" | "save" | "run" | null>(null);
  const [saveName, setSaveName] = useState(`${stem(dataset.name)}_pipeline.yaml`);
  const close = () => setDialog(null);
  const yaml = [".yaml", ".yml"];

  return (
    <div className="button-row pipeline-actions">
      <button onClick={() => setDialog("load")} title="Replace the steps with a saved pipeline file">
        Load…
      </button>
      <button disabled={steps.length === 0} onClick={() => setDialog("save")} title="Save the steps as a file">
        Save…
      </button>
      <button
        className="primary"
        disabled={!steps.some((s) => s.enabled)}
        onClick={() => setDialog("run")}
        title="Process the whole dataset into a new file, in the background"
      >
        Run…
      </button>
      {dialog === "load" && (
        <FileDialog
          title="Load pipeline"
          mode="open"
          start={datasetFolder}
          extensions={yaml}
          confirmLabel="Load"
          onConfirm={async (path) => {
            props.onLoaded((await api.loadPipeline(path)).steps);
            close();
          }}
          onCancel={close}
        />
      )}
      {dialog === "save" && (
        <FileDialog
          title="Save pipeline"
          mode="save"
          start={datasetFolder}
          extensions={yaml}
          name={saveName}
          onName={setSaveName}
          confirmLabel="Save"
          onConfirm={async (path) => {
            const file = yaml.some((x) => path.toLowerCase().endsWith(x)) ? path : `${path}.yaml`;
            await saveAsking((overwrite) => api.savePipeline(file, steps, dataset.id, overwrite));
            close();
          }}
          onCancel={close}
        />
      )}
      {dialog === "run" && <RunDialog {...props} onClose={close} />}
    </div>
  );
}
