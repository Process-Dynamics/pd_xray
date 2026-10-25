// Processing steps for the live preview, with controls built from the server's operation registry.

import { useEffect, useState } from "react";

import type { Curve, Operation, OperationParam, Preview, Reference, Tone } from "./api";
import { formatValue } from "./format";
import { CurveEditor, useToneCurve } from "./Histogram";
import { ReferencePicker } from "./ReferencePicker";

export interface EditableStep {
  id: number; // stable key for React while steps move
  op: string;
  params: Record<string, unknown>;
  enabled: boolean;
}

interface Props {
  operations: Operation[];
  steps: EditableStep[];
  showProcessed: boolean;
  preview: Preview | null;
  datasetFolder: string;
  /** Contrast on screen: a new Tone step starts with it as its black/white points, so the frame looks the same. */
  displayRange: { vmin: number; vmax: number };
  onSteps: (steps: EditableStep[]) => void;
  onShowProcessed: (show: boolean) => void;
}

let nextStepId = 1;

/** A new step with every parameter at its default. */
export function newStep(operation: Operation): EditableStep {
  return {
    id: nextStepId++,
    op: operation.name,
    params: Object.fromEntries(operation.params.map((p) => [p.name, p.default])),
    enabled: true,
  };
}

/** Number field that commits on Enter or blur, so typing doesn't start a preview per keystroke. Empty = null. */
function NumberInput({
  param,
  value,
  onCommit,
}: {
  param: OperationParam;
  value: unknown;
  onCommit: (v: unknown) => void;
}) {
  const show = (v: unknown) => (typeof v === "number" ? formatValue(v) : "");
  const [text, setText] = useState(show(value));
  useEffect(() => setText(show(value)), [value]);
  const commit = () => {
    if (text.trim() === "") {
      if (param.optional) onCommit(null);
      else setText(show(value));
      return;
    }
    const number = Number(text);
    if (Number.isFinite(number)) onCommit(param.kind === "int" ? Math.round(number) : number);
    else setText(show(value));
  };
  const range = [param.minimum, param.maximum].some((v) => v !== null)
    ? `${param.minimum ?? "…"} to ${param.maximum ?? "…"}`
    : "";
  return (
    <input
      type="text"
      inputMode="decimal"
      value={text}
      placeholder={param.optional ? "none" : ""}
      title={[param.description, range].filter(Boolean).join(" · ")}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

function describeReference(value: unknown): string {
  if (typeof value === "string") return `📁 ${value.split("/").filter(Boolean).pop() ?? value}`;
  if (Array.isArray(value)) {
    const names = value.map((v) => String(v).split("/").pop());
    return value.length === 1
      ? names[0]!
      : `${value.length} files: ${names.slice(0, 2).join(", ")}${value.length > 2 ? ", …" : ""}`;
  }
  return "none";
}

function ParamControl(props: {
  param: OperationParam;
  value: unknown;
  datasetFolder: string;
  onChange: (value: unknown) => void;
}) {
  const { param, value, onChange } = props;
  const [picking, setPicking] = useState(false);
  switch (param.kind) {
    case "bool":
      return <input type="checkbox" checked={value === true} onChange={(e) => onChange(e.target.checked)} />;
    case "choice":
      return (
        <select value={String(value)} onChange={(e) => onChange(e.target.value)} title={param.description}>
          {param.choices.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      );
    case "reference":
      return (
        <span
          className="reference"
          title={Array.isArray(value) ? value.join("\n") : String(value ?? param.description)}
        >
          <span className="reference-name">{describeReference(value)}</span>
          <button onClick={() => setPicking(true)}>Choose…</button>
          {value !== null && param.optional && (
            <button onClick={() => onChange(null)} title="Remove">
              ×
            </button>
          )}
          {picking && (
            <ReferencePicker
              title={`${param.name}: ${param.description}`}
              start={typeof value === "string" ? value : props.datasetFolder}
              onPick={(reference: Reference) => {
                setPicking(false);
                onChange(reference);
              }}
              onCancel={() => setPicking(false)}
            />
          )}
        </span>
      );
    default:
      return <NumberInput param={param} value={value} onCommit={onChange} />;
  }
}

/** Curve parameter of a step: the curve editor over 0..1, drawing the step's full tone response too. */
function StepCurve({
  params,
  name,
  onChange,
}: {
  params: Record<string, unknown>;
  name: string;
  onChange: (c: Curve) => void;
}) {
  const [draft, setDraft] = useState<Curve | null>(null); // while dragging a point
  const curve = params[name] as Curve;
  const { black: _black, white: _white, ...settings } = params;
  const samples = useToneCurve({ ...settings, [name]: draft ?? curve } as unknown as Tone);
  return (
    <CurveEditor
      counts={null}
      low={0}
      high={1}
      log={false}
      black={0}
      white={1}
      curve={curve}
      samples={samples}
      height={90}
      onCurveDraft={setDraft}
      onCurve={(c) => {
        setDraft(null);
        onChange(c);
      }}
    />
  );
}

export function ProcessingPanel(props: Props) {
  const { operations, steps, preview, onSteps } = props;
  const byName = Object.fromEntries(operations.map((op) => [op.name, op]));
  const update = (index: number, change: Partial<EditableStep>) =>
    onSteps(steps.map((s, i) => (i === index ? { ...s, ...change } : s)));
  const move = (index: number, by: number) => {
    const next = [...steps];
    const [step] = next.splice(index, 1);
    next.splice(index + by, 0, step!);
    onSteps(next);
  };
  // Timings come per enabled step, in order.
  const timings = new Map<number, number>();
  if (preview) {
    steps
      .filter((s) => s.enabled)
      .forEach((s, i) => preview.timings[i] && timings.set(s.id, preview.timings[i].seconds));
  }
  const total = preview?.timings.reduce((sum, t) => sum + t.seconds, 0) ?? 0;

  return (
    <section>
      <h3>Processing</h3>
      <div className="toggle" role="radiogroup" title="P switches between the two">
        <button className={props.showProcessed ? "" : "active"} onClick={() => props.onShowProcessed(false)}>
          Original
        </button>
        <button
          className={props.showProcessed ? "active" : ""}
          onClick={() => props.onShowProcessed(true)}
          disabled={!steps.some((s) => s.enabled)}
        >
          Processed
        </button>
      </div>
      {steps.map((step, index) => {
        const operation = byName[step.op];
        return (
          <div key={step.id} className={`step ${step.enabled ? "" : "step-disabled"}`}>
            <div className="step-header">
              <input
                type="checkbox"
                checked={step.enabled}
                onChange={(e) => update(index, { enabled: e.target.checked })}
                title="Enable this step"
              />
              <span className="step-title" title={operation?.description}>
                {index + 1}. {operation?.title ?? step.op}
              </span>
              {timings.has(step.id) && <span className="step-time">{timings.get(step.id)!.toFixed(2)} s</span>}
              <button disabled={index === 0} onClick={() => move(index, -1)} title="Move up">
                ↑
              </button>
              <button disabled={index === steps.length - 1} onClick={() => move(index, 1)} title="Move down">
                ↓
              </button>
              <button onClick={() => onSteps(steps.filter((_, i) => i !== index))} title="Remove step">
                ×
              </button>
            </div>
            {operation && operation.params.length > 0 && (
              <div className="field-grid step-params">
                {operation.params
                  .filter((param) => param.kind !== "curve")
                  .map((param) => (
                    <label key={param.name} className="contents">
                      <span title={param.description}>{param.name.replace(/_/g, " ")}</span>
                      <ParamControl
                        param={param}
                        value={step.params[param.name]}
                        datasetFolder={props.datasetFolder}
                        onChange={(value) => update(index, { params: { ...step.params, [param.name]: value } })}
                      />
                    </label>
                  ))}
              </div>
            )}
            {operation?.params
              .filter((param) => param.kind === "curve")
              .map((param) => (
                <StepCurve
                  key={param.name}
                  params={step.params}
                  name={param.name}
                  onChange={(curve) => update(index, { params: { ...step.params, [param.name]: curve } })}
                />
              ))}
          </div>
        );
      })}
      <select
        className="add-step"
        value=""
        onChange={(e) => {
          const operation = byName[e.target.value];
          if (!operation) return;
          const step = newStep(operation);
          if (operation.name === "tone") {
            step.params = { ...step.params, black: props.displayRange.vmin, white: props.displayRange.vmax };
          }
          onSteps([...steps, step]);
        }}
      >
        <option value="">+ Add step…</option>
        {operations.map((op) => (
          <option key={op.name} value={op.name} title={op.description}>
            {op.title}
          </option>
        ))}
      </select>
      {preview && props.showProcessed && (
        <div className="muted">
          Last preview: {preview.width} × {preview.height}, {total.toFixed(2)} s
        </div>
      )}
    </section>
  );
}
