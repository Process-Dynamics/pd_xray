// Side panel: histogram, display settings (contrast, tone, colour map, image format) and dataset information.
// Display settings only change how frames are shown, never the data.

import { type ReactNode, useEffect, useRef, useState } from "react";

import {
  type DatasetInfo,
  DEFAULT_TONE,
  type Display,
  isDefaultTone,
  type Job,
  type OperationParam,
  type Tone,
} from "./api";
import { formatBytes, formatValue } from "./format";
import { FrameHistogram } from "./Histogram";

interface Props {
  dataset: DatasetInfo;
  display: Display;
  /** Whether the processed view is shown (contrast then applies to it; the dataset range is for original data). */
  processed: boolean;
  colormaps: string[];
  children?: ReactNode; // shown between the display settings and the dataset information
  rangeJob: Job | null;
  /** The frame on screen, for the histogram. */
  histogramSource: { datasetId: string; frame: number[]; pipeline?: string } | null;
  /** The tone sliders (from the `tone` operation's parameters). */
  toneParams: OperationParam[];
  onDisplay: (display: Display) => void;
  /** Turn the display tone into a pipeline step (between the current Min and Max). */
  onToneStep: () => void;
  onAutoContrast: (mode: "auto" | "full") => void;
  onDatasetRange: () => void;
}

/** Number field that commits on Enter or when leaving the field, so typing doesn't reload frames per keystroke. */
function ValueInput({ value, onCommit }: { value: number; onCommit: (value: number) => void }) {
  const [text, setText] = useState(formatValue(value));
  useEffect(() => setText(formatValue(value)), [value]);
  const commit = () => {
    const number = Number(text);
    if (text.trim() !== "" && Number.isFinite(number)) onCommit(number);
    else setText(formatValue(value));
  };
  return (
    <input
      type="text"
      inputMode="decimal"
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

/** Slider that shows changes at once and commits them when released (each commit reloads the frame's tiles). */
function ToneSlider(props: {
  param: OperationParam;
  value: number;
  onDraft: (value: number) => void;
  onCommit: () => void;
}) {
  const { param, value } = props;
  return (
    <>
      <span title={param.description}>{param.name}</span>
      <input
        type="range"
        min={param.minimum ?? -100}
        max={param.maximum ?? 100}
        step={param.name === "exposure" ? 0.05 : 1}
        value={value}
        title={`${param.description} Double-click: reset.`}
        onChange={(e) => props.onDraft(Number(e.target.value))}
        onPointerUp={props.onCommit}
        onKeyUp={props.onCommit}
        onDoubleClick={() => {
          props.onDraft(0);
          props.onCommit();
        }}
      />
      <span className="tone-value">{param.name === "exposure" ? value.toFixed(2) : value}</span>
    </>
  );
}

export function DisplayPanel(props: Props) {
  const { dataset, display, colormaps, rangeJob, onDisplay } = props;
  const rangeRunning = rangeJob !== null && (rangeJob.state === "queued" || rangeJob.state === "running");
  // Tone edits show at once in the panel (sliders, curve) and reach the frame when committed.
  const [tone, setTone] = useState<Tone>(display.tone);
  useEffect(() => setTone(display.tone), [display.tone]);
  const latest = useRef(tone);
  latest.current = tone;
  const draft = (change: Partial<Tone>) => {
    latest.current = { ...latest.current, ...change };
    setTone(latest.current);
  };
  const commit = () => {
    if (JSON.stringify(latest.current) !== JSON.stringify(display.tone))
      onDisplay({ ...display, tone: latest.current });
  };

  return (
    <>
      <section>
        <h3>Display{props.processed ? " (processed)" : ""}</h3>
        <FrameHistogram
          source={props.histogramSource}
          black={display.vmin}
          white={display.vmax}
          tone={tone}
          onBlackWhite={(vmin, vmax) => onDisplay({ ...display, vmin, vmax })}
          onCurveDraft={(curve) => draft({ curve })}
          onCurve={(curve) => {
            draft({ curve });
            commit();
          }}
          onReset={() => {
            draft({ curve: DEFAULT_TONE.curve });
            commit();
            props.onAutoContrast("auto");
          }}
        />
        <div className="field-grid">
          <span>Min</span>
          <ValueInput value={display.vmin} onCommit={(vmin) => onDisplay({ ...display, vmin })} />
          <span>Max</span>
          <ValueInput value={display.vmax} onCommit={(vmax) => onDisplay({ ...display, vmax })} />
        </div>
        <div className="button-row">
          <button onClick={() => props.onAutoContrast("auto")} title="0.1–99.9 percentile of the frame on screen">
            Auto
          </button>
          <button onClick={() => props.onAutoContrast("full")} title="Min and max of the frame on screen">
            Frame range
          </button>
          <button
            onClick={props.onDatasetRange}
            disabled={rangeRunning || props.processed}
            title="Min and max over every frame of the original data (reads the whole dataset as a background job)"
          >
            {rangeRunning ? "Scanning…" : "Dataset range"}
          </button>
        </div>
        <div className="field-grid">
          <span>Colours</span>
          <select
            value={display.cmap}
            disabled={dataset.channels > 1}
            onChange={(e) => onDisplay({ ...display, cmap: e.target.value })}
          >
            {colormaps.map((name) => (
              <option key={name} value={name}>
                {name.replace("_", " ")}
              </option>
            ))}
          </select>
          <span>Image</span>
          <select
            value={display.format}
            onChange={(e) => onDisplay({ ...display, format: e.target.value as Display["format"] })}
            title="PNG shows exact 8-bit values; JPEG is smaller and can be faster over slow networks"
          >
            <option value="png">PNG (exact)</option>
            <option value="jpeg">JPEG (smaller)</option>
          </select>
        </div>
      </section>

      <section>
        <h3>Tone (display)</h3>
        <div className="tone-grid">
          {props.toneParams
            .filter((p) => p.kind === "float")
            .map((param) => (
              <ToneSlider
                key={param.name}
                param={param}
                value={tone[param.name as Exclude<keyof Tone, "curve">]}
                onDraft={(value) => draft({ [param.name]: value })}
                onCommit={commit}
              />
            ))}
        </div>
        <div className="button-row">
          <button
            disabled={isDefaultTone(tone)}
            onClick={() => {
              draft(DEFAULT_TONE);
              commit();
            }}
            title="Every slider to 0 and a straight curve"
          >
            Reset
          </button>
          <button
            disabled={isDefaultTone(display.tone)}
            onClick={props.onToneStep}
            title="Add these settings as a Tone step (black/white = Min/Max), so a pipeline run applies them to the data"
          >
            Add as step
          </button>
        </div>
      </section>

      {props.children}

      <section>
        <h3>Dataset</h3>
        <div className="info-name" title={dataset.location}>
          {dataset.options.connection && "🔌 "}
          {dataset.name}
          {dataset.options.dataset && <span className="muted"> : {dataset.options.dataset}</span>}
          {dataset.options.pattern && <span className="muted"> / {dataset.options.pattern}</span>}
        </div>
        <table className="info">
          <tbody>
            <tr>
              <td>Shape</td>
              <td>
                {dataset.shape.join(" × ")} ({dataset.axes.join(", ")})
              </td>
            </tr>
            <tr>
              <td>Type</td>
              <td>{dataset.dtype}</td>
            </tr>
            <tr>
              <td>Size</td>
              <td>{formatBytes(dataset.nbytes)} uncompressed</td>
            </tr>
          </tbody>
        </table>
        {dataset.warnings.map((w) => (
          <div key={w} className="warning">
            ⚠ {w}
          </div>
        ))}
        {Object.keys(dataset.metadata).length > 0 && (
          <details>
            <summary>Metadata</summary>
            <pre className="metadata">{JSON.stringify(dataset.metadata, null, 2)}</pre>
          </details>
        )}
      </section>
    </>
  );
}
