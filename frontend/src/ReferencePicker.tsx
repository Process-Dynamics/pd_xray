// Dialog to choose a reference (e.g. dark or flat frames): a whole folder, or one or more files. All frames found are
// averaged by the server.

import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { api, ApiError, type Listing, type Reference } from "./api";
import { formatBytes } from "./format";

interface Props {
  title: string;
  start: string; // folder to open first
  onPick: (reference: Reference) => void;
  onCancel: () => void;
}

export function ReferencePicker({ title, start, onPick, onCancel }: Props) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState("");
  const [error, setError] = useState<string | null>(null);

  const go = useCallback(async (path: string | null): Promise<boolean> => {
    try {
      setListing(await api.list(path));
      setSelected(new Set());
      setFilter("");
      setError(null);
      return true;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      return false;
    }
  }, []);

  useEffect(() => {
    void go(start).then((ok) => ok || go(null)); // e.g. a folder that was moved: start from the allowed folders
  }, [go, start]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const needle = filter.trim().toLowerCase();
  const entries = (listing?.entries ?? []).filter((e) => e.name.toLowerCase().includes(needle));
  const files = (listing?.entries ?? []).filter((e) => e.kind === "file" && e.openable);
  const shownFiles = entries.filter((e) => e.kind === "file" && e.openable);
  const toggle = (path: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (!next.delete(path)) next.add(path);
      return next;
    });

  // Rendered at page level, so the surrounding panel's form styles and <label> don't apply to the dialog.
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="modal" role="dialog" aria-label={title}>
        <div className="modal-title">{title}</div>
        <div className="browser-path" title={listing?.path ?? "Allowed folders"}>
          <button disabled={!listing?.path} onClick={() => void go(listing?.parent ?? null)} title="Up one folder">
            ↑
          </button>
          <span>
            <bdi>{listing?.path ?? "Allowed folders"}</bdi>
          </span>
        </div>
        <div className="modal-tools">
          <input placeholder="Filter, e.g. _dar" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <button
            disabled={shownFiles.length === 0}
            onClick={() => setSelected(new Set(shownFiles.map((f) => f.path)))}
            title="Select every file shown"
          >
            Select shown ({shownFiles.length})
          </button>
          <button disabled={selected.size === 0} onClick={() => setSelected(new Set())}>
            Clear
          </button>
        </div>
        {error && <div className="warning">{error}</div>}
        <div className="modal-list">
          {entries.map((e) =>
            e.kind === "dir" ? (
              <button key={e.path} className="row" onClick={() => void go(e.path)}>
                <span className="row-name">📁 {e.name}</span>
              </button>
            ) : (
              <label key={e.path} className={`row ${e.openable ? "" : "row-disabled"}`}>
                <span className="row-name">
                  <input
                    type="checkbox"
                    disabled={!e.openable}
                    checked={selected.has(e.path)}
                    onChange={() => toggle(e.path)}
                  />{" "}
                  {e.name}
                </span>
                <span className="row-meta">{formatBytes(e.size)}</span>
              </label>
            ),
          )}
          {listing && entries.length === 0 && <div className="muted">Nothing here.</div>}
        </div>
        <div className="modal-buttons">
          <button onClick={onCancel}>Cancel</button>
          <button
            disabled={!listing?.path || files.length === 0}
            onClick={() => listing?.path && onPick(listing.path)}
            title="Every image file directly in this folder"
          >
            Use this folder ({files.length} files)
          </button>
          <button
            className="primary"
            disabled={selected.size === 0}
            onClick={() => onPick(files.filter((f) => selected.has(f.path)).map((f) => f.path))}
          >
            Use {selected.size} selected file{selected.size === 1 ? "" : "s"}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
