// Histogram of the frame on screen with draggable black/white handles, and an editable tone curve drawn over it.
// Curves are sampled by the server (`/api/tone-curve`), so what is drawn is exactly what the tiles and the tone step do.

import { useEffect, useRef, useState } from "react";

import { api, type Curve, type Histogram, type Tone, type ToneCurve } from "./api";
import { formatValue } from "./format";

const W = 280; // SVG units; the editor scales to the panel width
const HANDLE_ZONE = 12; // strip below the plot holding the black/white handles
const HIT = 7; // how close (SVG units) a pointer must be to grab a point or handle
const MIN_GAP = 0.01; // smallest x distance between curve points

/** The server's samples of `tone` (the curve alone and every adjustment together), refetched when it changes. */
export function useToneCurve(tone: Tone | null): ToneCurve | null {
  const [samples, setSamples] = useState<ToneCurve | null>(null);
  const key = tone && JSON.stringify(tone);
  useEffect(() => {
    if (!tone) return setSamples(null);
    const controller = new AbortController();
    api.toneCurve(tone, controller.signal).then(setSamples, () => undefined); // drawing only: keep the last on error
    return () => controller.abort();
  }, [key]); // by value: a new but equal object doesn't refetch
  return samples;
}

interface EditorProps {
  /** Pixel counts in equal bins over `low`..`high`, or null for no histogram. */
  counts: number[] | null;
  low: number;
  high: number;
  log: boolean;
  /** Values where the curve's 0 and 1 lie (display Min/Max, or a tone step's black/white points). */
  black: number;
  white: number;
  curve: Curve;
  samples: ToneCurve | null;
  height?: number;
  /** Commit new black/white values after a handle drag. Without it, no handles are shown. */
  onBlackWhite?: (black: number, white: number) => void;
  /** The curve while a point is being dragged (e.g. to update `samples`). */
  onCurveDraft: (curve: Curve) => void;
  /** Commit the curve: after a drag, or after removing a point (double-click). */
  onCurve: (curve: Curve) => void;
}

type Drag = { kind: "black" | "white" } | { kind: "point"; index: number };

/** Histogram bars with the tone curve on top. Click between the handles to add a point; double-click one to remove it. */
export function CurveEditor(props: EditorProps) {
  const { counts, low, high, log, samples } = props;
  const height = props.height ?? 120;
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const [draft, setDraft] = useState({ black: props.black, white: props.white, curve: props.curve });
  // Follow the committed values unless a drag is in progress.
  useEffect(() => {
    if (!drag.current) setDraft({ black: props.black, white: props.white, curve: props.curve });
  }, [props.black, props.white, props.curve]);
  const latest = useRef(draft);
  latest.current = draft;

  const span = high > low ? high - low : 1;
  const xOf = (value: number) => ((value - low) / span) * W;
  const valueOf = (x: number) => low + (x / W) * span;
  const xb = xOf(draft.black);
  const xw = xOf(draft.white);
  const pointXY = ([x, y]: [number, number]) => [xb + x * (xw - xb), height * (1 - y)] as const;

  const toSvg = (event: React.PointerEvent | React.MouseEvent) => {
    const rect = svg.current!.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * W,
      ((event.clientY - rect.top) / rect.height) * (height + HANDLE_ZONE),
    ] as const;
  };
  const nearestPoint = (x: number, y: number) => {
    let best = -1;
    let distance = HIT;
    draft.curve.forEach((point, i) => {
      const [px, py] = pointXY(point);
      const d = Math.hypot(px - x, py - y);
      if (d <= distance) [best, distance] = [i, d];
    });
    return best;
  };

  const moveTo = (x: number, y: number) => {
    const current = drag.current;
    if (!current) return;
    if (current.kind !== "point") {
      const value = valueOf(Math.min(W, Math.max(0, x)));
      const gap = span / W; // keep at least one SVG unit between the handles
      setDraft((d) =>
        current.kind === "black"
          ? { ...d, black: Math.min(value, d.white - gap) }
          : { ...d, white: Math.max(value, d.black + gap) },
      );
      return;
    }
    const curve = latest.current.curve.map((p) => [...p] as [number, number]);
    const i = current.index;
    const lowest = i === 0 ? 0 : curve[i - 1]![0] + MIN_GAP;
    const highest = i === curve.length - 1 ? 1 : curve[i + 1]![0] - MIN_GAP;
    const cx = xw > xb ? (x - xb) / (xw - xb) : 0;
    curve[i] = [Math.min(highest, Math.max(lowest, cx)), Math.min(1, Math.max(0, 1 - y / height))];
    setDraft((d) => ({ ...d, curve }));
    props.onCurveDraft(curve);
  };

  const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    const [x, y] = toSvg(event);
    if (props.onBlackWhite && y > height) {
      drag.current = { kind: Math.abs(x - xb) <= Math.abs(x - xw) ? "black" : "white" };
    } else {
      let index = nearestPoint(x, y);
      if (index < 0) {
        if (xw - xb < 4 || x <= xb || x >= xw) return; // outside the curve's range
        const cx = (x - xb) / (xw - xb);
        const curve = [...draft.curve];
        index = curve.findIndex(([px]) => px > cx);
        if (index <= 0 || cx - curve[index - 1]![0] < MIN_GAP || curve[index]![0] - cx < MIN_GAP) return;
        curve.splice(index, 0, [cx, Math.min(1, Math.max(0, 1 - y / height))]);
        latest.current = { ...draft, curve };
        setDraft(latest.current);
      }
      drag.current = { kind: "point", index };
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    moveTo(x, y);
  };

  const onPointerUp = () => {
    const current = drag.current;
    drag.current = null;
    if (!current) return;
    const { black, white, curve } = latest.current;
    if (current.kind === "point") props.onCurve(curve);
    else if (black !== props.black || white !== props.white) props.onBlackWhite?.(black, white);
  };

  const onDoubleClick = (event: React.MouseEvent) => {
    const [x, y] = toSvg(event);
    const index = nearestPoint(x, y);
    if (index <= 0 || index >= draft.curve.length - 1) return; // the end points stay
    props.onCurve(draft.curve.filter((_, i) => i !== index));
  };

  // Bars as one path, scaled to the tallest bin (log: log(1 + count)).
  let bars = "";
  if (counts && counts.length) {
    const scale = (c: number) => (log ? Math.log1p(c) : c);
    const top = Math.max(...counts.map(scale)) || 1;
    const bin = W / counts.length;
    counts.forEach((c, i) => {
      const h = (scale(c) / top) * height;
      if (h > 0) bars += `M${(i * bin).toFixed(2)} ${height}v${(-h).toFixed(2)}h${bin.toFixed(2)}v${h.toFixed(2)}z`;
    });
  }
  const line = (values: number[]) =>
    values
      .map((v, i) => `${i ? "L" : "M"}${(xb + (i / (values.length - 1)) * (xw - xb)).toFixed(2)} ${height * (1 - v)}`)
      .join("");
  const straight = draft.curve.map((p, i) => `${i ? "L" : "M"}${pointXY(p).join(" ")}`).join("");
  const dragging = drag.current?.kind === "point";

  return (
    <svg
      ref={svg}
      className="curve-editor"
      viewBox={`0 0 ${W} ${height + HANDLE_ZONE}`}
      onPointerDown={onPointerDown}
      onPointerMove={(e) => drag.current && moveTo(...toSvg(e))}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={onDoubleClick}
    >
      <title>
        {"Blue: tone curve. Orange: every adjustment together. Click between the handles to add a point, drag to " +
          "move it, double-click to remove it." +
          (props.onBlackWhite ? " Drag the triangles to set Min (black) and Max (white)." : "")}
      </title>
      <rect className="curve-plot" x={0} y={0} width={W} height={height} />
      <path className="histogram-bars" d={bars} />
      <path className="curve-diagonal" d={`M${xb} ${height}L${xw} 0`} />
      <path className="curve-flat" d={`M0 ${height}H${xb}M${xw} 0H${W}`} />
      {samples && !isIdentity(samples.response, samples.curve) && (
        <path className="curve-response" d={line(samples.response)} />
      )}
      <path className="curve-line" d={samples && !dragging ? line(samples.curve) : straight} />
      {draft.curve.map((p, i) => {
        const [cx, cy] = pointXY(p);
        return <circle key={i} className="curve-point" cx={cx} cy={cy} r={3.5} />;
      })}
      {props.onBlackWhite && (
        <>
          <path className="handle handle-black" d={`M${xb} ${height + 1}l-5 ${HANDLE_ZONE - 2}h10z`} />
          <path className="handle handle-white" d={`M${xw} ${height + 1}l-5 ${HANDLE_ZONE - 2}h10z`} />
        </>
      )}
    </svg>
  );
}

/** Whether the full response equals the curve alone (then only the curve is drawn). */
function isIdentity(response: number[], curve: number[]): boolean {
  return response.every((v, i) => Math.abs(v - curve[i]!) < 1e-4);
}

interface FrameHistogramProps {
  /** The frame on screen: dataset id, frame index and the processed view's pipeline (if shown). */
  source: { datasetId: string; frame: number[]; pipeline?: string } | null;
  black: number;
  white: number;
  tone: Tone;
  onBlackWhite: (black: number, white: number) => void;
  onCurveDraft: (curve: Curve) => void;
  onCurve: (curve: Curve) => void;
  /** Straight curve and automatic Min/Max. */
  onReset: () => void;
}

const SETTLE_MS = 150; // fetch the histogram once the frame on screen stops changing (not for every scrubbed frame)

/** Histogram of the frame on screen with the display contrast handles and the display tone curve. */
export function FrameHistogram({ source, black, white, tone, onReset, ...callbacks }: FrameHistogramProps) {
  const [histogram, setHistogram] = useState<Histogram | null>(null);
  const [log, setLog] = useState(true);
  const [full, setFull] = useState(false); // false: around Min..Max (room to edit the curve); true: the frame's range
  const samples = useToneCurve(tone);
  const pad = (white - black) * 0.15;
  const range = full ? null : { low: black - pad, high: white + pad };
  const key = source && JSON.stringify([source, range]);

  useEffect(() => {
    if (!source) return setHistogram(null);
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api
        .histogram(source.datasetId, source.frame, range, source.pipeline, controller.signal)
        .then(setHistogram, () => undefined); // a missing histogram never blocks viewing
    }, SETTLE_MS);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [key]); // by value, as above

  return (
    <div className="histogram">
      <CurveEditor
        counts={histogram?.counts ?? null}
        low={histogram?.low ?? black}
        high={histogram?.high ?? white}
        log={log}
        black={black}
        white={white}
        curve={tone.curve}
        samples={samples}
        {...callbacks}
      />
      <div className="histogram-axis muted">
        <span>{histogram ? formatValue(histogram.low) : ""}</span>
        <span className="histogram-tools">
          <label title="Logarithmic counts show sparse values">
            <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log
          </label>
          <label title="Show the frame's whole value range instead of the range around Min–Max">
            <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} /> full range
          </label>
          <button className="histogram-reset" onClick={onReset} title="Straight curve and automatic Min/Max">
            Reset
          </button>
        </span>
        <span>{histogram ? formatValue(histogram.high) : ""}</span>
      </div>
    </div>
  );
}
