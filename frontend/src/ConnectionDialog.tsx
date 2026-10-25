// Dialog to connect to a NAS (SFTP) or an S3 bucket. The server tests the connection before adding it. Settings
// without secrets are remembered in this browser; passwords and keys are sent once and kept only in server memory.

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { api, ApiError, type Connection, type ConnectionKind, type ConnectionSettings } from "./api";

interface Props {
  onConnected: (connection: Connection) => void;
  onCancel: () => void;
}

interface Saved {
  kind: ConnectionKind;
  name: string;
  settings: ConnectionSettings;
}

const SAVED_KEY = "pd_xray.connections";

const FIELDS: Record<ConnectionKind, { key: string; label: string; placeholder?: string; secret?: boolean }[]> = {
  sftp: [
    { key: "host", label: "Host", placeholder: "nas.lab or 192.168.1.20" },
    { key: "port", label: "Port", placeholder: "22" },
    { key: "username", label: "User name" },
    { key: "root", label: "Folder", placeholder: "optional, e.g. /volume1/xray" },
    { key: "password", label: "Password", secret: true },
    { key: "passphrase", label: "Key passphrase", placeholder: "only for an encrypted key", secret: true },
  ],
  s3: [
    { key: "endpoint_url", label: "Endpoint URL", placeholder: "empty for AWS, e.g. https://minio.lab:9000" },
    { key: "region", label: "Region", placeholder: "optional, e.g. eu-west-2" },
    { key: "bucket", label: "Bucket" },
    { key: "prefix", label: "Folder", placeholder: "optional, e.g. scans/2026" },
    { key: "access_key", label: "Access key", secret: true },
    { key: "secret_key", label: "Secret key", secret: true },
  ],
};

function loadSaved(): Saved[] {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(SAVED_KEY) ?? "[]");
    return Array.isArray(saved) ? (saved as Saved[]) : [];
  } catch {
    return [];
  }
}

function storeSaved(saved: Saved[]) {
  try {
    localStorage.setItem(SAVED_KEY, JSON.stringify(saved));
  } catch {
    // storage unavailable (private window): settings just aren't remembered
  }
}

export function ConnectionDialog({ onConnected, onCancel }: Props) {
  const [kind, setKind] = useState<ConnectionKind>("sftp");
  const [name, setName] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [keyFile, setKeyFile] = useState<{ name: string; text: string } | null>(null);
  const [saved, setSaved] = useState<Saved[]>(loadSaved);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fingerprint, setFingerprint] = useState<string | null>(null); // unknown SSH host key to confirm

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const fields = FIELDS[kind];
  const set = (key: string, value: string) =>
    setValues((v) => {
      const next = { ...v, [key]: value };
      if (key === "host" || key === "port") delete next.host_key; // a trusted key belongs to one server
      return next;
    });

  const fill = (entry: Saved) => {
    setKind(entry.kind);
    setName(entry.name);
    setValues(Object.fromEntries(Object.entries(entry.settings).map(([k, v]) => [k, String(v)])));
    setKeyFile(null);
    setError(null);
    setFingerprint(null);
  };

  const connect = async (hostKey?: string) => {
    const settings: ConnectionSettings = {};
    const credentials: Record<string, string> = {};
    for (const field of fields) {
      const value = (values[field.key] ?? "").trim();
      if (!value) continue;
      if (field.secret) credentials[field.key] = value;
      else settings[field.key] = field.key === "port" ? Number(value) : value;
    }
    if (kind === "sftp" && keyFile) credentials.key_data = keyFile.text;
    if (hostKey) settings.host_key = hostKey;
    else if (kind === "sftp" && values.host_key) settings.host_key = values.host_key;
    setBusy(true);
    setError(null);
    setFingerprint(null);
    try {
      const connection = await api.connect(kind, name.trim(), settings, credentials);
      const entry: Saved = { kind, name: name.trim(), settings: connection.settings };
      const same = (s: Saved) => s.kind === entry.kind && JSON.stringify(s.settings) === JSON.stringify(entry.settings);
      const next = [entry, ...saved.filter((s) => !same(s))].slice(0, 20);
      storeSaved(next);
      onConnected(connection);
    } catch (e) {
      const presented = e instanceof ApiError ? e.body?.fingerprint : undefined;
      if (typeof presented === "string")
        setFingerprint(presented); // explained below instead of the error text
      else setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const forget = (index: number) => {
    const next = saved.filter((_, i) => i !== index);
    setSaved(next);
    storeSaved(next);
  };

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onCancel()}>
      <form
        className="modal modal-form"
        role="dialog"
        aria-label="Add connection"
        onSubmit={(e) => {
          e.preventDefault();
          void connect();
        }}
      >
        <div className="modal-title">Add connection</div>
        <div className="modal-tools">
          <button type="button" className={kind === "sftp" ? "primary" : ""} onClick={() => setKind("sftp")}>
            NAS (SFTP)
          </button>
          <button type="button" className={kind === "s3" ? "primary" : ""} onClick={() => setKind("s3")}>
            S3 bucket
          </button>
        </div>
        <div className="modal-body">
          {saved.length > 0 && (
            <div className="saved-connections">
              <div className="muted">Saved (no passwords or keys; click to fill in):</div>
              {saved.map((s, i) => (
                <div key={i} className="saved-connection">
                  <button type="button" className="row" onClick={() => fill(s)}>
                    <span className="row-name">{s.name || describe(s)}</span>
                    <span className="row-meta">{s.kind === "sftp" ? "NAS" : "S3"}</span>
                  </button>
                  <button type="button" onClick={() => forget(i)} title="Forget">
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="field-grid">
            <label htmlFor="connection-name">Name</label>
            <input
              id="connection-name"
              placeholder="optional, e.g. Lab NAS"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            {fields.map((f) => (
              <FieldRow key={`${kind}-${f.key}`} field={f} value={values[f.key] ?? ""} onChange={set} />
            ))}
            {kind === "sftp" && (
              <>
                <label htmlFor="connection-key">Private key</label>
                <span className="key-file">
                  <input
                    id="connection-key"
                    type="file"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) void file.text().then((text) => setKeyFile({ name: file.name, text }));
                      else setKeyFile(null);
                    }}
                  />
                </span>
              </>
            )}
          </div>
          <div className="muted">
            {kind === "sftp"
              ? "Give a password or a private key. On a Synology, enable SFTP in Control Panel › File Services."
              : "Give the access key and secret key. Ask your admin for the endpoint URL of a non-AWS service."}{" "}
            Passwords and keys stay in the server's memory only and are gone when it restarts.
          </div>
          {error && <div className="warning">{error}</div>}
          {fingerprint && (
            <div className="warning">
              This server's identity isn't known yet (or has changed). Its key fingerprint is <code>{fingerprint}</code>
              . Trust it only if it matches the one your admin gives you, or the output of{" "}
              <code>ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub</code> on the server.
              <div className="button-row">
                <button type="button" disabled={busy} onClick={() => void connect(fingerprint)}>
                  Trust this key and connect
                </button>
              </div>
            </div>
          )}
        </div>
        <div className="modal-buttons">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={busy}>
            {busy ? "Connecting…" : "Connect"}
          </button>
        </div>
      </form>
    </div>,
    document.body,
  );
}

function FieldRow({
  field,
  value,
  onChange,
}: {
  field: { key: string; label: string; placeholder?: string; secret?: boolean };
  value: string;
  onChange: (key: string, value: string) => void;
}) {
  const id = `connection-${field.key}`;
  return (
    <>
      <label htmlFor={id}>{field.label}</label>
      <input
        id={id}
        type={field.secret ? "password" : "text"}
        autoComplete={field.secret ? "new-password" : "off"}
        placeholder={field.placeholder}
        value={value}
        onChange={(e) => onChange(field.key, e.target.value)}
        spellCheck={false}
      />
    </>
  );
}

function describe(saved: Saved): string {
  const s = saved.settings;
  return saved.kind === "sftp"
    ? `${s.username ?? ""}@${s.host ?? ""}${s.root ? ` ${s.root}` : ""}`
    : `s3://${s.bucket ?? ""}`;
}
