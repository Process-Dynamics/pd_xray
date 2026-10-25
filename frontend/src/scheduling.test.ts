import { describe, expect, it } from "vitest";

import { clampFrame, LatestWins, nextFrame, osdLevels } from "./scheduling";

describe("LatestWins", () => {
  it("runs only the latest request after the current load", () => {
    const started: number[] = [];
    const finish: (() => void)[] = [];
    const loader = new LatestWins<number>((key, done) => {
      started.push(key);
      finish.push(done);
    });
    loader.request(1);
    loader.request(2);
    loader.request(3);
    expect(started).toEqual([1]);
    finish[0]!();
    expect(started).toEqual([1, 3]);
    finish[1]!();
    expect(loader.busy).toBe(false);
  });

  it("ignores a second done() and repeated requests for the frame already loading", () => {
    const started: number[] = [];
    let finish = () => {};
    const loader = new LatestWins<number>((key, done) => {
      started.push(key);
      finish = done;
    });
    loader.request(1);
    loader.request(1);
    const first = finish;
    first();
    first();
    expect(started).toEqual([1]);
    expect(loader.busy).toBe(false);
  });

  it("aborts the running load for a newer request, and reloads a key whose load was aborted", () => {
    const started: number[] = [];
    const runs: { done: () => void; signal: AbortSignal }[] = [];
    const loader = new LatestWins<number>((key, done, signal) => {
      started.push(key);
      runs.push({ done, signal });
    });
    loader.request(1);
    loader.request(2);
    expect(runs[0]!.signal.aborted).toBe(true);
    loader.request(1); // back to 1, whose load was aborted: must load again
    runs[0]!.done();
    expect(started).toEqual([1, 1]);
    expect(runs[1]!.signal.aborted).toBe(false);
  });

  it("compares keys with the given function", () => {
    const started: string[] = [];
    const loader = new LatestWins<{ f: number }>(
      (key) => started.push(String(key.f)),
      (a, b) => a.f === b.f,
    );
    loader.request({ f: 1 });
    loader.request({ f: 1 });
    expect(started).toEqual(["1"]);
  });
});

describe("osdLevels", () => {
  it("maps the server pyramid onto OpenSeadragon levels", () => {
    // 4096 x 2304 with 512 px tiles: server levels 0..3, OpenSeadragon level 12 is full resolution.
    expect(osdLevels(4096, 2304, 4)).toEqual({ minLevel: 9, maxLevel: 12 });
    expect(osdLevels(5120, 2101, 5)).toEqual({ minLevel: 9, maxLevel: 13 });
    expect(osdLevels(300, 200, 1)).toEqual({ minLevel: 9, maxLevel: 9 });
  });
});

describe("nextFrame", () => {
  it("continues the last step, within the dataset", () => {
    expect(nextFrame([4], [5], [10])).toEqual([6]);
    expect(nextFrame([5], [4], [10])).toEqual([3]);
    expect(nextFrame([0, 7], [1, 7], [3, 10])).toEqual([2, 7]);
    expect(nextFrame([8], [9], [10])).toBeNull(); // would leave the dataset
    expect(nextFrame([5], [5], [10])).toBeNull(); // no movement (display change)
    expect(nextFrame([0], [500], [1000])).toBeNull(); // a jump, not stepping
    expect(nextFrame(null, [5], [10])).toBeNull();
  });
});

describe("clampFrame", () => {
  it("keeps indices inside the frame axes", () => {
    expect(clampFrame([-3, 99], [10, 20])).toEqual([0, 19]);
    expect(clampFrame([], [5])).toEqual([0]);
    expect(clampFrame([1], [])).toEqual([]);
  });
});
