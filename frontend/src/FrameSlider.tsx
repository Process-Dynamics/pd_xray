// One slider per frame axis (time, angle, …). Keys: ←/→ step the last axis, ↑/↓ the one before it,
// Shift: × 10, Home/End: jump to the ends. Keys work anywhere except while typing in a text field.

import { useEffect } from "react";

import { clampFrame } from "./scheduling";

interface Props {
  axes: string[];
  shape: number[];
  frame: number[];
  shownFrame: number[] | null;
  onChange: (frame: number[]) => void;
}

export function FrameSlider({ axes, shape, frame, shownFrame, onChange }: Props) {
  useEffect(() => {
    if (shape.length === 0) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.isContentEditable || target.tagName === "TEXTAREA" || target.tagName === "SELECT") return;
      const input = target as HTMLInputElement;
      if (target.tagName === "INPUT" && (input.type !== "range" || !target.closest(".slider-row"))) return; // e.g. tone
      const last = shape.length - 1;
      const step = event.shiftKey ? 10 : 1;
      const moves: Record<string, [number, number]> = {
        ArrowRight: [last, step],
        ArrowLeft: [last, -step],
        ArrowUp: [last - 1, step],
        ArrowDown: [last - 1, -step],
        Home: [last, -Infinity],
        End: [last, Infinity],
      };
      const move = moves[event.key];
      if (!move || move[0] < 0) return;
      event.preventDefault(); // also stops a focused range input from moving a second time
      const next = [...frame];
      const axis = move[0];
      const current = next[axis] ?? 0;
      next[axis] = Number.isFinite(move[1]) ? current + move[1] : move[1] > 0 ? shape[axis]! - 1 : 0;
      onChange(clampFrame(next, shape));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [shape, frame, onChange]);

  if (shape.length === 0) return <div className="slider muted">Single frame</div>;
  const set = (axis: number, value: number) => {
    const next = [...frame];
    next[axis] = value;
    onChange(clampFrame(next, shape));
  };
  const waiting = shownFrame !== null && shownFrame.join(",") !== frame.join(",");

  return (
    <div className="slider">
      {axes.map((axis, i) => (
        <label key={axis} className="slider-row">
          <span className="slider-axis">{axis}</span>
          <input
            type="range"
            min={0}
            max={shape[i]! - 1}
            value={frame[i] ?? 0}
            onChange={(e) => set(i, Number(e.target.value))}
          />
          <input
            type="number"
            className="slider-number"
            min={0}
            max={shape[i]! - 1}
            value={frame[i] ?? 0}
            onChange={(e) => e.target.value !== "" && set(i, Number(e.target.value))}
          />
          <span className="slider-total">/ {shape[i]! - 1}</span>
        </label>
      ))}
      <span className={`slider-shown ${waiting ? "slider-waiting" : ""}`} title="The frame currently on screen">
        showing {shownFrame?.join(", ") ?? "…"}
      </span>
    </div>
  );
}
