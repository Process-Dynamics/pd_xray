// Dialog to download a remote file or folder: choose a folder on the server (inside the allowed folders) and options.
// The download runs as a background job shown in the status bar.

import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { api, ApiError, type Listing } from "./api";

interface Props {
  name: string; // what is downloaded, for the title
  onDownload: (folder: string, overwrite: boolean, verify: boolean) => void;
  onCancel: () => void;
}

const LAST_FOLDER_KEY = "pd_xray.downloadFolder";

function lastFolder(): string | null {
  try {
    return localStorage.getItem(LAST_FOLDER_KEY);
  } catch {
    return null;
  }
}

export function DownloadDialog({ name, onDownload, onCancel }: Props) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [overwrite, setOverwrite] = useState(false);
  const [verify, setVerify] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const go = useCallback(async (path: string | null): Promise<boolean> => {
    try {
      setListing(await api.list(path));
      setError(null);
      return true;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      return false;
    }
  }, []);

  useEffect(() => {
    void go(lastFolder()).then((ok) => ok || go(null)); // the last folder may be gone: start from the allowed folders
  }, [go]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const folder = listing?.path ?? null;
  const download = () => {
    if (!folder) return;
    try {
      localStorage.setItem(LAST_FOLDER_KEY, folder);
    } catch {
      // not remembered; fine
    }
    onDownload(folder, overwrite, verify);
  };

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="modal" role="dialog" aria-label={`Download ${name}`}>
        <div className="modal-title">Download {name} to…</div>
        <div className="browser-path" title={folder ?? "Allowed folders"}>
          <button disabled={!folder} onClick={() => void go(listing?.parent ?? null)} title="Up one folder">
            ↑
          </button>
          <span>
            <bdi>{folder ?? "Allowed folders"}</bdi>
          </span>
        </div>
        {error && <div className="warning">{error}</div>}
        <div className="modal-list">
          {listing?.entries
            .filter((e) => e.kind === "dir")
            .map((e) => (
              <button key={e.path} className="row" onClick={() => void go(e.path)}>
                <span className="row-name">📁 {e.name}</span>
              </button>
            ))}
          {listing && !listing.entries.some((e) => e.kind === "dir") && <div className="muted">No folders here.</div>}
        </div>
        <div className="modal-tools">
          <label title="Replace files that are already there with a different size or date. Identical files are always skipped.">
            <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} /> Overwrite
            different files
          </label>
          <label title="After copying, read both copies again and compare SHA-256. Takes about twice as long.">
            <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} /> Verify (slower)
          </label>
        </div>
        <div className="modal-buttons">
          <button onClick={onCancel}>Cancel</button>
          <button className="primary" disabled={!folder} onClick={download} title={folder ? `Into ${folder}` : ""}>
            Download here
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
