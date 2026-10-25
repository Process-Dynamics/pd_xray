// Folder browser within the server's allowed roots. Opens files, file sequences and HDF5 datasets.

import { useCallback, useEffect, useState } from "react";

import { api, type Hdf5Dataset, type Listing } from "./api";
import { formatBytes } from "./format";

export interface OpenRequest {
  path: string;
  dataset?: string;
  pattern?: string;
  label: string;
}

interface Props {
  track: <T>(label: string, work: Promise<T>) => Promise<T | undefined>;
  onOpen: (request: OpenRequest) => void;
  openPath: string | null;
}

const MAX_ROWS = 2000; // keep huge folders responsive; the filter finds the rest

export function FileBrowser({ track, onOpen, openPath }: Props) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [filter, setFilter] = useState("");
  const [picker, setPicker] = useState<{ path: string; datasets: Hdf5Dataset[] } | null>(null);

  const go = useCallback(
    async (path: string | null) => {
      const result = await track("Listing folder", api.list(path));
      if (result) {
        setListing(result);
        setFilter("");
        setPicker(null);
      }
    },
    [track],
  );

  useEffect(() => {
    void go(null);
  }, [go]);

  const openFile = async (path: string, name: string) => {
    const inspection = await track(`Inspecting ${name}`, api.inspect(path));
    if (!inspection) return;
    if (inspection.format !== "hdf5") return onOpen({ path, label: name });
    const images = inspection.datasets.filter((d) => d.is_image);
    if (images.length === 1) return onOpen({ path, dataset: images[0]!.name, label: `${name}:${images[0]!.name}` });
    setPicker({ path, datasets: inspection.datasets });
  };

  const needle = filter.trim().toLowerCase();
  const entries = (listing?.entries ?? []).filter((e) => e.name.toLowerCase().includes(needle));
  const sequences = (listing?.sequences ?? []).filter((s) => s.pattern.toLowerCase().includes(needle));
  const name = (path: string) => path.split("/").pop() ?? path;

  return (
    <div className="browser">
      <div className="browser-path" title={listing?.path ?? "Allowed folders"}>
        <button disabled={!listing?.path} onClick={() => void go(listing?.parent ?? null)} title="Up one folder">
          ↑
        </button>
        <span>
          <bdi>{listing?.path ?? "Allowed folders"}</bdi>
        </span>
      </div>
      <input
        className="browser-filter"
        placeholder="Filter"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        spellCheck={false}
      />
      {picker && (
        <div className="browser-picker">
          <div className="browser-picker-title">
            Choose a dataset in {name(picker.path)}
            <button onClick={() => setPicker(null)} title="Close">
              ×
            </button>
          </div>
          {picker.datasets.length === 0 && <div className="muted">No datasets in this file.</div>}
          {picker.datasets.map((d) => (
            <button
              key={d.name}
              className="row"
              disabled={!d.is_image}
              title={d.warnings.join("\n") || undefined}
              onClick={() => onOpen({ path: picker.path, dataset: d.name, label: `${name(picker.path)}:${d.name}` })}
            >
              <span className="row-name">{d.name}</span>
              <span className="row-meta">
                {d.shape.join("×")} {d.dtype}
                {d.warnings.length > 0 && " ⚠"}
              </span>
            </button>
          ))}
        </div>
      )}
      <div className="browser-list">
        {sequences.map((s) => (
          <button
            key={s.pattern}
            className="row row-sequence"
            title={`Open the ${s.count} files matching ${s.pattern} as one dataset`}
            onClick={() => onOpen({ path: listing!.path!, pattern: s.pattern, label: s.pattern })}
          >
            <span className="row-name">▦ {s.pattern}</span>
            <span className="row-meta">{s.count} files</span>
          </button>
        ))}
        {entries.slice(0, MAX_ROWS).map((e) =>
          e.kind === "dir" ? (
            <button key={e.path} className="row" onClick={() => void go(e.path)}>
              <span className="row-name">📁 {e.name}</span>
            </button>
          ) : (
            <button
              key={e.path}
              className={`row ${e.path === openPath ? "row-open" : ""}`}
              disabled={!e.openable}
              onClick={() => void openFile(e.path, e.name)}
            >
              <span className="row-name">{e.name}</span>
              <span className="row-meta">{formatBytes(e.size)}</span>
            </button>
          ),
        )}
        {entries.length > MAX_ROWS && (
          <div className="muted">{entries.length - MAX_ROWS} more; type in the filter to find them.</div>
        )}
        {listing && entries.length === 0 && sequences.length === 0 && <div className="muted">Nothing here.</div>}
      </div>
    </div>
  );
}
