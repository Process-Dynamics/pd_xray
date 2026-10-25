// Zoomable frame viewer (OpenSeadragon) fed by the server's tile API.
//
// Each frame (or display change) is a new tiled image, loaded invisibly on top of the current one and swapped in only
// once every visible tile has arrived. The image on screen therefore always belongs to exactly one frame and one
// display setting, and `onShown` reports which. Loads go through `LatestWins`, so fast scrubbing skips frames instead
// of queueing them. A processed view first asks the server for a preview (which can take seconds and is abandoned as
// soon as a newer view is requested), then shows its tiles the same way.

import OpenSeadragon from "openseadragon";
import { useEffect, useRef } from "react";

import { api, ApiError, type DatasetInfo, type Display, frameKey, type Preview, type Step, tileUrl } from "./api";
import { LatestWins, nextFrame, osdLevels } from "./scheduling";

export interface View {
  dataset: DatasetInfo;
  frame: number[];
  /** `vmin`/`vmax` null means the frame's automatic contrast (processed views only). */
  display: Omit<Display, "vmin" | "vmax"> & { vmin: number | null; vmax: number | null };
  /** Pipeline steps to preview, or null for the original data. `client` identifies this tab to the server. */
  processing: { steps: Step[]; client: string } | null;
}

const viewKey = (v: View) =>
  [
    v.dataset.id,
    frameKey(v.frame),
    v.display.vmin,
    v.display.vmax,
    v.display.cmap,
    v.display.format,
    JSON.stringify(v.display.tone),
    v.processing ? JSON.stringify(v.processing.steps) : "",
  ].join("|");

const LOAD_TIMEOUT_MS = 60_000; // safety net for tiles: a load that never completes must not block later frames

/** Why a tile failed: tiles load as plain images (fastest), so ask the server again to read its error message. */
async function tileError(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    if (response.ok) return "A tile failed to load, but loads now; step the frame to retry.";
    const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
    const detail = typeof body.detail === "string" ? body.detail : `${response.status} ${response.statusText}`;
    return `Can't load the frame: ${detail}`;
  } catch {
    return "Can't reach the server to load the frame.";
  }
}

interface Props {
  view: View | null;
  onShown: (view: View) => void;
  /** What is loading ("Loading frame 3", "Processing frame 3"), or null when idle. */
  onLoading: (label: string | null) => void;
  onPreview: (view: View, preview: Preview) => void;
  onError: (message: string) => void;
}

export function Viewer({ view, onShown, onLoading, onPreview, onError }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const viewer = useRef<OpenSeadragon.Viewer | null>(null);
  const loader = useRef<LatestWins<View> | null>(null);
  const shown = useRef<string | null>(null);
  const requested = useRef<string | null>(null); // errors from older requests are not reported
  const callbacks = useRef({ onShown, onLoading, onPreview, onError });
  callbacks.current = { onShown, onLoading, onPreview, onError };

  useEffect(() => {
    const osd = OpenSeadragon({
      element: element.current!,
      drawer: "canvas", // the long-established renderer; WebGL is newer in OpenSeadragon
      showNavigationControl: false,
      immediateRender: true, // load the needed level directly, not coarse-to-fine: fewer requests per frame
      blendTime: 0, // no fade: a swapped frame appears at once
      animationTime: 0.3,
      imageSmoothingEnabled: false, // show real pixels when zoomed in
      maxZoomPixelRatio: 32,
      minZoomImageRatio: 0.5,
      visibilityRatio: 0.2,
      crossOriginPolicy: false,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
    });
    viewer.current = osd;
    let generation = 0; // ignores loads that finish after the viewer was reset
    let previous: View | null = null; // last shown view, to guess the next frame
    let shownSize = ""; // "dataset id: width x height" on screen; a change refits the view
    const prefetch = (urls: string[]) => {
      // Plain low-priority image loads fill the browser's and the server's caches for the next step.
      for (const url of urls) {
        const image = new Image();
        image.fetchPriority = "low";
        image.src = url;
      }
    };
    const keepOnly = (keep: OpenSeadragon.TiledImage) => {
      for (let i = osd.world.getItemCount() - 1; i >= 0; i--) {
        const other = osd.world.getItemAt(i);
        if (other !== keep) osd.world.removeItem(other);
      }
    };

    /** Show `next` from tiles: the original frame, or the processed frame described by `preview`. */
    const showTiles = (next: View, preview: Preview | null, done: () => void) => {
      const mine = generation;
      const { dataset, frame } = next;
      const width = preview?.width ?? dataset.width;
      const height = preview?.height ?? dataset.height;
      const display: Display = {
        ...next.display,
        vmin: next.display.vmin ?? preview?.stats.low ?? 0,
        vmax: next.display.vmax ?? preview?.stats.high ?? 1,
      };
      const { minLevel, maxLevel } = osdLevels(width, height, preview?.levels ?? dataset.levels);
      const size = `${dataset.id}:${width}x${height}`;
      const refit = osd.world.getItemCount() === 0 || size !== shownSize;
      let item: OpenSeadragon.TiledImage | null = null;
      let settled = false;
      const tiles = new Map<string, [number, number, number]>(); // server level, column, row of requested tiles

      const finish = (error?: string) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        osd.removeHandler("tile-load-failed", failed);
        if (error !== undefined) {
          if (item) osd.world.removeItem(item);
          if (requested.current === viewKey(next)) callbacks.current.onError(error);
        } else if (item && mine === generation) {
          keepOnly(item);
          item.setOpacity(1);
          shown.current = viewKey(next);
          shownSize = size;
          callbacks.current.onShown(next);
        }
        callbacks.current.onLoading(null);
        const stepping = previous?.dataset.id === dataset.id && !preview && !previous.processing && !error;
        const ahead = stepping ? nextFrame(previous!.frame, frame, dataset.frame_shape) : null;
        if (error === undefined) previous = next;
        done();
        if (ahead && !loader.current?.busy) {
          prefetch([...tiles.values()].map(([level, x, y]) => tileUrl(dataset.id, ahead, level, x, y, display)));
        }
      };
      const failed = (event: OpenSeadragon.TileLoadFailedEvent) => {
        if (event.tiledImage === item) void tileError(event.tile.getUrl()).then(finish);
      };
      const timeout = () => finish("A frame took over a minute to load; skipped it.");
      const timer = window.setTimeout(timeout, LOAD_TIMEOUT_MS);
      osd.addHandler("tile-load-failed", failed);

      osd.addTiledImage({
        tileSource: {
          width,
          height,
          tileSize: dataset.tile_size,
          tileOverlap: 0,
          minLevel,
          maxLevel,
          getTileUrl: (level: number, x: number, y: number) => {
            tiles.set(`${level}/${x}/${y}`, [maxLevel - level, x, y]);
            return tileUrl(dataset.id, frame, maxLevel - level, x, y, display, preview?.pipeline);
          },
        },
        // A new image size replaces everything at once; otherwise the frame loads hidden on top of the current one.
        opacity: refit ? 1 : 0,
        preload: true,
        success: (event) => {
          item = (event as unknown as { item: OpenSeadragon.TiledImage }).item;
          if (refit) {
            keepOnly(item);
            osd.viewport.goHome(true);
          }
          const loaded = item;
          if (loaded.getFullyLoaded()) {
            finish();
          } else {
            const handler = (e: { fullyLoaded: boolean }) => {
              if (!e.fullyLoaded) return;
              loaded.removeHandler("fully-loaded-change", handler);
              finish();
            };
            loaded.addHandler("fully-loaded-change", handler);
          }
        },
        error: (error) => finish(`Can't show the frame: ${String(error)}`),
      });
    };

    loader.current = new LatestWins<View>(
      (next, done, signal) => {
        if (shown.current === viewKey(next)) return done(); // already on screen (e.g. requested again after an abort)
        const label = `${next.processing ? "Processing" : "Loading"} frame ${next.frame.join(", ")}`;
        callbacks.current.onLoading(label);
        if (!next.processing) return showTiles(next, null, done);
        const { steps, client } = next.processing;
        api.preview(next.dataset.id, next.frame, steps, client, signal).then(
          (preview) => {
            if (signal.aborted) {
              callbacks.current.onLoading(null);
              return done();
            }
            callbacks.current.onPreview(next, preview);
            showTiles(next, preview, done);
          },
          (error: unknown) => {
            callbacks.current.onLoading(null);
            const superseded = signal.aborted || (error instanceof ApiError && error.status === 409);
            if (!superseded && requested.current === viewKey(next)) {
              callbacks.current.onError(error instanceof Error ? error.message : String(error));
            }
            done();
          },
        );
      },
      (a, b) => viewKey(a) === viewKey(b),
    );

    return () => {
      generation++;
      loader.current = null;
      viewer.current = null;
      shown.current = null;
      osd.destroy();
    };
  }, []);

  useEffect(() => {
    requested.current = view && viewKey(view);
    if (view === null) {
      viewer.current?.world.removeAll();
      shown.current = null;
      return;
    }
    if (shown.current !== viewKey(view) || loader.current?.busy) loader.current?.request(view);
  }, [view]);

  return (
    <div className="viewer">
      <div ref={element} className="viewer-canvas" />
      {view === null && <div className="viewer-empty">Open a file from the browser on the left.</div>}
      {view !== null && (
        <button
          className="viewer-fit"
          title="Fit to window (double-click zooms in)"
          onClick={() => viewer.current?.viewport.goHome()}
        >
          Fit
        </button>
      )}
    </div>
  );
}
