// Dialog to choose a file in the allowed folders: an existing one ("open"), or a folder and a name ("save"). Extra
// options (e.g. output format) go in `children`, shown under the list.

import { type ReactNode, useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { api, ApiError, type Listing } from "./api";
import { formatBytes } from "./format";

interface Props {
  title: string;
  mode: "open" | "save";
  start: string | null; // folder to open first; null for the allowed folders
  extensions: string[]; // files shown, e.g. [".yaml", ".yml"]; others are hidden
  name?: string; // save: the name typed so far
  onName?: (name: string) => void; // save: the name changed (typed, or a file clicked)
  confirmLabel: string;
  children?: ReactNode;
  /** Called with the chosen path. A thrown error is shown in the dialog, which then stays open. */
  onConfirm: (path: string) => Promise<void>;
  onCancel: () => void;
}

export function FileDialog(props: Props) {
  const { mode, extensions, onCancel } = props;
  const [listing, setListing] = useState<Listing | null>(null);
  const [chosen, setChosen] = useState<string | null>(null); // open: the file clicked
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const go = useCallback(async (path: string | null): Promise<boolean> => {
    try {
      setListing(await api.list(path));
      setChosen(null);
      setError(null);
      return true;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      return false;
    }
  }, []);

  useEffect(() => {
    void go(props.start).then((ok) => ok || go(null)); // e.g. a folder that was moved: start from the allowed folders
  }, [go, props.start]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const folder = listing?.path ?? null;
  const name = (props.name ?? "").trim();
  const target = mode === "open" ? chosen : folder && name && !name.includes("/") ? `${folder}/${name}` : null;
  const files = (listing?.entries ?? []).filter(
    (e) => e.kind === "file" && extensions.some((x) => e.name.toLowerCase().endsWith(x)),
  );
  const confirm = async (path: string | null = target) => {
    if (!path || busy) return;
    setBusy(true);
    try {
      await props.onConfirm(path);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="modal" role="dialog" aria-label={props.title}>
        <div className="modal-title">{props.title}</div>
        <div className="browser-path" title={folder ?? "Allowed folders"}>
          <button disabled={!folder} onClick={() => void go(listing?.parent ?? null)} title="Up one folder">
            ↑
          </button>
          <span>
            <bdi>{folder ?? "Allowed folders"}</bdi>
          </span>
        </div>
        <div className="modal-list">
          {listing?.entries
            .filter((e) => e.kind === "dir")
            .map((e) => (
              <button key={e.path} className="row" onClick={() => void go(e.path)}>
                <span className="row-name">📁 {e.name}</span>
              </button>
            ))}
          {files.map((e) => (
            <button
              key={e.path}
              className={`row ${chosen === e.path ? "row-open" : ""}`}
              onClick={() => (mode === "open" ? setChosen(e.path) : props.onName?.(e.name))}
              onDoubleClick={() => mode === "open" && void confirm(e.path)}
            >
              <span className="row-name">📄 {e.name}</span>
              <span className="row-meta">{formatBytes(e.size)}</span>
            </button>
          ))}
          {listing && listing.entries.every((e) => e.kind !== "dir") && files.length === 0 && (
            <div className="muted">Nothing here.</div>
          )}
        </div>
        <div className="modal-body">
          {mode === "save" && (
            <label className="file-name">
              Name{" "}
              <input
                value={props.name ?? ""}
                onChange={(e) => props.onName?.(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void confirm()}
              />
            </label>
          )}
          {props.children}
          {error && <div className="warning">{error}</div>}
        </div>
        <div className="modal-buttons">
          <button onClick={onCancel}>Cancel</button>
          <button
            className="primary"
            disabled={!target || busy}
            onClick={() => void confirm()}
            title={target ?? (mode === "save" && !folder ? "Open a folder first" : "")}
          >
            {props.confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
