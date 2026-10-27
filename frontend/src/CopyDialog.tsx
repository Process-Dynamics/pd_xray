// Dialog to copy a file or folder: choose a destination folder on the server (inside the allowed folders) or on a
// connection, and options. The copy runs as a background job shown in the status bar.

import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { api, ApiError, type Connection, type Listing, type Location } from "./api";

interface Props {
  name: string; // what is copied, for the title
  onCopy: (destination: Location, overwrite: boolean, verify: boolean) => void;
  onCancel: () => void;
}

const LAST_DESTINATION_KEY = "pd_xray.copyDestination";

function lastDestination(): Location | null {
  try {
    const saved = JSON.parse(localStorage.getItem(LAST_DESTINATION_KEY) ?? "null") as Location | null;
    return saved && typeof saved.path === "string" ? saved : null;
  } catch {
    return null;
  }
}

export function CopyDialog({ name, onCopy, onCancel }: Props) {
  const [listing, setListing] = useState<Listing | null>(null);
  const [remote, setRemote] = useState<Connection | null>(null); // connection being browsed
  const [connections, setConnections] = useState<Connection[]>([]);
  const [overwrite, setOverwrite] = useState(false);
  const [verify, setVerify] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** A server folder (`connection` null; path null = the top level) or a connection's folder. */
  const go = useCallback(async (connection: Connection | null, path: string | null): Promise<boolean> => {
    try {
      setListing(connection ? await api.listRemote(connection.id, path ?? "") : await api.list(path));
      setRemote(connection);
      setError(null);
      return true;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      return false;
    }
  }, []);

  useEffect(() => {
    void (async () => {
      const list = await api.connections().catch(() => [] as Connection[]);
      setConnections(list);
      const last = lastDestination(); // may be gone (folder deleted, connection closed): start from the top level
      const connection = last?.connection ? list.find((c) => c.id === last.connection) : null;
      if (!last || connection === undefined || !(await go(connection, last.path))) await go(null, null);
    })();
  }, [go]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const top = !remote && listing?.path == null;
  const destination: Location | null = remote
    ? { connection: remote.id, path: listing?.path ?? "" }
    : listing?.path
      ? { connection: null, path: listing.path }
      : null;
  const where = remote ? `${remote.name}: /${listing?.path ?? ""}` : (listing?.path ?? "Allowed folders");
  const up = () => {
    if (remote && listing?.parent != null) void go(remote, listing.parent);
    else void go(null, remote ? null : (listing?.parent ?? null));
  };
  const copy = () => {
    if (!destination) return;
    try {
      localStorage.setItem(LAST_DESTINATION_KEY, JSON.stringify(destination));
    } catch {
      // not remembered; fine
    }
    onCopy(destination, overwrite, verify);
  };

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="modal" role="dialog" aria-label={`Copy ${name}`}>
        <div className="modal-title">Copy {name} to…</div>
        <div className="browser-path" title={remote ? `${where}\n${remote.url}` : where}>
          <button disabled={top} onClick={up} title="Up one folder">
            ↑
          </button>
          <span>
            <bdi>{where}</bdi>
          </span>
        </div>
        {error && <div className="warning">{error}</div>}
        <div className="modal-list">
          {top &&
            connections.map((c) => (
              <button key={c.id} className="row" onClick={() => void go(c, "")} title={c.url}>
                <span className="row-name">🔌 {c.name}</span>
                <span className="row-meta">{c.kind === "sftp" ? "NAS" : "S3"}</span>
              </button>
            ))}
          {listing?.entries
            .filter((e) => e.kind === "dir")
            .map((e) => (
              <button key={e.path} className="row" onClick={() => void go(remote, e.path)}>
                <span className="row-name">📁 {e.name}</span>
              </button>
            ))}
          {listing && !top && !listing.entries.some((e) => e.kind === "dir") && (
            <div className="muted">No folders here.</div>
          )}
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
          <button className="primary" disabled={!destination} onClick={copy} title={destination ? `Into ${where}` : ""}>
            Copy here
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
