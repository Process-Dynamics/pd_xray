import { describe, expect, it } from "vitest";

import { DEFAULT_TONE, type Display, isDefaultTone, tileUrl } from "./api";

describe("tone in tile URLs", () => {
  const display: Display = { vmin: 0, vmax: 10, cmap: "grey", format: "png", tone: DEFAULT_TONE };

  it("leaves default tone out, so cached tiles stay valid", () => {
    expect(
      isDefaultTone({
        ...DEFAULT_TONE,
        curve: [
          [0, 0],
          [1, 1],
        ],
      }),
    ).toBe(true);
    expect(tileUrl("d", [3], 0, 1, 2, display)).not.toContain("tone=");
  });

  it("sends any change, including a curve alone", () => {
    const curve: [number, number][] = [
      [0, 0],
      [0.5, 0.7],
      [1, 1],
    ];
    expect(isDefaultTone({ ...DEFAULT_TONE, curve })).toBe(false);
    expect(isDefaultTone({ ...DEFAULT_TONE, shadows: -5 })).toBe(false);
    const url = new URL(tileUrl("d", [3], 0, 1, 2, { ...display, tone: { ...DEFAULT_TONE, curve } }), "http://x");
    expect(JSON.parse(url.searchParams.get("tone")!).curve).toEqual(curve);
  });
});
