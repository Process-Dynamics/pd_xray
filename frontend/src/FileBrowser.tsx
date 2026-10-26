// Folder browser within the server's allowed roots and this browser's remote connections (NAS, S3). Opens local
// files, file sequences and HDF5 datasets, and copies files and folders anywhere.

import { useCallback, useEffect, useState } from "react";

import { api, type Connection, type Hdf5Dataset, type Listing, type Location } from "./api";
import { ConnectionDialog } from "./ConnectionDialog";
import { CopyDialog } from "./CopyDialog";
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
  const [connections, setConnections] = useState<Connection[]>([]);
  const [remote, setRemote] = useState<Connection | null>(null); // connection being browsed
  const [adding, setAdding] = useState(false);
  const [copying, setCopying] = useState<{ source: Location; name: string } | null>(null);

  const show = useCallback((result: Listing | undefined, connection: Connection | null) => {
    if (!result) return;
    setListing(result);
    setRemote(connection);
    setFilter("");
    setPicker(null);
  }, []);

  /** Local folder, or the top level (allowed folders and connections) for null. */
  const go = useCallback(
    async (path: string | null) => {
      if (path === null) void track("Listing connections", api.connections()).then((c) => c && setConnections(c));
      show(await track("Listing folder", api.list(path)), null);
    },
    [track, show],
  );

  const goRemote = async (connection: Connection, path: string) =>
    show(await track(`Listing ${connection.name}`, api.listRemote(connection.id, path)), connection);

  const up = () => {
    if (remote && listing?.parent != null) void goRemote(remote, listing.parent);
    else void go(remote ? null : (listing?.parent ?? null));
  };

  const copy = async (destination: Location, overwrite: boolean, verify: boolean) => {
    if (!copying) return;
    setCopying(null);
    await track(`Starting copy of ${copying.name}`, api.copy(copying.source, destination, overwrite, verify));
  };
  /** Opens the copy dialog for `path` in the folder shown. */
  const startCopy = (path: string, name: string) =>
    setCopying({ source: { connection: remote?.id ?? null, path }, name });

  const disconnect = async (connection: Connection) => {
    await track(`Disconnecting ${connection.name}`, api.disconnect(connection.id));
    setConnections((list) => list.filter((c) => c.id !== connection.id));
  };

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
  const top = !remote && listing?.path == null;
  const where = remote ? `${remote.name}: /${listing?.path ?? ""}` : (listing?.path ?? "Allowed folders");

  return (
    <div className="browser">
      <div className="browser-path" title={remote ? `${where}\n${remote.url}` : where}>
        <button disabled={top} onClick={up} title="Up one folder">
          ↑
        </button>
        <span>
          <bdi>{where}</bdi>
        </span>
        {listing && !top && (
          <button
            onClick={() => startCopy(listing.path ?? "", name(listing.path || remote!.name))}
            title="Copy this folder to…"
          >
            ⧉
          </button>
        )}
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
        {top &&
          connections.map((c) => (
            <div key={c.id} className="connection-row">
              <button className="row" onClick={() => void goRemote(c, "")} title={c.url}>
                <span className="row-name">🔌 {c.name}</span>
                <span className="row-meta">{c.kind === "sftp" ? "NAS" : "S3"}</span>
              </button>
              <button onClick={() => void disconnect(c)} title="Disconnect">
                ×
              </button>
            </div>
          ))}
        {top && (
          <button className="row add-connection" onClick={() => setAdding(true)}>
            <span className="row-name">+ Add connection (NAS or S3)</span>
          </button>
        )}
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
        {entries.slice(0, MAX_ROWS).map((e) => {
          const row =
            e.kind === "dir" ? (
              <button
                key={e.path}
                className="row"
                onClick={() => void (remote ? goRemote(remote, e.path) : go(e.path))}
              >
                <span className="row-name">📁 {e.name}</span>
              </button>
            ) : (
              <button
                key={e.path}
                className={`row ${e.path === openPath ? "row-open" : ""}`}
                disabled={!e.openable}
                title={remote ? "Opening files from a connection isn't supported yet" : undefined}
                onClick={() => void openFile(e.path, e.name)}
              >
                <span className="row-name">{e.name}</span>
                <span className="row-meta">{formatBytes(e.size)}</span>
              </button>
            );
          return (
            <div key={e.path} className="connection-row">
              {row}
              <button onClick={() => startCopy(e.path, e.name)} title={`Copy ${e.name} to…`}>
                ⧉
              </button>
            </div>
          );
        })}
        {entries.length > MAX_ROWS && (
          <div className="muted">{entries.length - MAX_ROWS} more; type in the filter to find them.</div>
        )}
        {listing && entries.length === 0 && sequences.length === 0 && <div className="muted">Nothing here.</div>}
      </div>
      {copying && (
        <CopyDialog
          name={copying.name}
          onCancel={() => setCopying(null)}
          onCopy={(destination, overwrite, verify) => void copy(destination, overwrite, verify)}
        />
      )}
      {adding && (
        <ConnectionDialog
          onCancel={() => setAdding(false)}
          onConnected={(connection) => {
            setAdding(false);
            setConnections((list) => [...list, connection]);
            void goRemote(connection, "");
          }}
        />
      )}
    </div>
  );
}
