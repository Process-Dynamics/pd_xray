// Pure logic behind smooth scrubbing, kept free of React and OpenSeadragon so it can be tested.

/**
 * Runs at most one load at a time, and only ever the latest request.
 *
 * While a load runs, new requests replace each other; when it finishes, only the most recent one starts. Dragging the
 * slider over 500 frames therefore never queues 500 loads: the viewer shows frames as fast as the server delivers them
 * and always ends on the frame the user stopped at.
 *
 * A newer request also aborts the running load's `signal`. Loads that are quick to finish may ignore it (finishing
 * keeps scrubbing smooth); slow ones (processing previews) should stop and call `done()`.
 */
export class LatestWins<K> {
  private running: { key: K; controller: AbortController } | undefined;
  private next: K | undefined;

  /** `start(key, done, signal)` begins a load and must call `done()` exactly once, on success, failure or abort. */
  constructor(
    private readonly start: (key: K, done: () => void, signal: AbortSignal) => void,
    private readonly same: (a: K, b: K) => boolean = Object.is,
  ) {}

  request(key: K): void {
    if (this.running === undefined) {
      this.run(key);
    } else if (!this.same(this.running.key, key)) {
      this.next = key;
      this.running.controller.abort();
    } else {
      // Back to the key already loading: drop the pending one, or reload it if its load was aborted.
      this.next = this.running.controller.signal.aborted ? key : undefined;
    }
  }

  get busy(): boolean {
    return this.running !== undefined;
  }

  private run(key: K): void {
    const controller = new AbortController();
    this.running = { key, controller };
    let finished = false;
    this.start(
      key,
      () => {
        if (finished) return;
        finished = true;
        this.running = undefined;
        const next = this.next;
        this.next = undefined;
        if (next !== undefined) this.run(next);
      },
      controller.signal,
    );
  }
}

/**
 * OpenSeadragon numbers levels from 0 (1 pixel) up to `maxLevel` (full resolution); the server numbers them from 0
 * (full resolution) up to `levels - 1` (fits one tile). Both halve the size per level, rounding up.
 */
export function osdLevels(width: number, height: number, levels: number): { minLevel: number; maxLevel: number } {
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height, 1)));
  return { minLevel: maxLevel - levels + 1, maxLevel };
}

/**
 * The frame to prefetch after showing `current`: one more step in the direction from `previous`, or null if there was
 * no movement, a big jump (more than `maxStep` on an axis), or the step would leave the dataset.
 */
export function nextFrame(
  previous: number[] | null,
  current: number[],
  shape: number[],
  maxStep = 10,
): number[] | null {
  if (previous === null || previous.length !== current.length) return null;
  const delta = current.map((c, axis) => c - previous[axis]!);
  if (delta.every((d) => d === 0) || delta.some((d) => Math.abs(d) > maxStep)) return null;
  const next = current.map((c, axis) => c + delta[axis]!);
  return next.every((n, axis) => n >= 0 && n < shape[axis]!) ? next : null;
}

/** Clamp a frame index to the frame axes' lengths. */
export function clampFrame(frame: number[], shape: number[]): number[] {
  return shape.map((n, axis) => Math.min(Math.max(Math.round(frame[axis] ?? 0), 0), n - 1));
}
