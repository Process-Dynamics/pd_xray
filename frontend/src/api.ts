// Typed access to the pd-xray server API (see /docs on the server). Errors carry the server's message.

export interface ServerInfo {
  roots: string[];
  tile_size: number;
  formats: ImageFormat[];
  colormaps: string[];
  extensions: string[];
}

export interface FileEntry {
  name: string;
  path: string;
  kind: "dir" | "file";
  size: number | null;
  openable?: boolean;
}

export interface SequenceGroup {
  pattern: string;
  count: number;
  first: string;
}

export interface Listing {
  connection?: string; // set for a remote connection's folder; paths are then relative to its root ("" = root)
  path: string | null;
  parent: string | null;
  entries: FileEntry[];
  sequences?: SequenceGroup[];
}

export type ConnectionKind = "sftp" | "s3";

/** Connection fields without secrets (sftp: host, port, username, root, host_key; s3: bucket, prefix, endpoint_url,
 * region). */
export type ConnectionSettings = Record<string, string | number>;

export interface Connection {
  id: string;
  kind: ConnectionKind;
  name: string;
  url: string;
  settings: ConnectionSettings;
}

/** A file or folder on a connection, or on the server (`connection` null: an absolute path inside the roots). */
export interface Location {
  connection: string | null;
  path: string;
}

export interface Hdf5Dataset {
  name: string;
  shape: number[];
  dtype: string;
  is_image: boolean;
  warnings: string[];
}

export interface Inspection {
  path: string;
  format: string;
  datasets: Hdf5Dataset[];
}

export interface DatasetInfo {
  id: string;
  path: string;
  name: string;
  options: { dataset?: string; pattern?: string };
  shape: number[];
  dtype: string;
  axes: string[];
  frame_axes: string[];
  frame_shape: number[];
  height: number;
  width: number;
  channels: number;
  levels: number;
  tile_size: number;
  nbytes: number;
  metadata: Record<string, unknown>;
  warnings: string[];
}

export interface FrameStats {
  min: number;
  max: number;
  low: number;
  high: number;
}

export type JobState = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  title: string;
  unit: string; // what current/total count: "bytes", or "" for items such as frames
  state: JobState;
  current: number;
  total: number | null;
  fraction: number | null;
  message: string;
  eta_seconds: number | null;
  rate: number | null; // units per second while running
  result: unknown;
  error: string | null;
}

export interface OperationParam {
  name: string;
  kind: "float" | "int" | "bool" | "choice" | "reference" | "curve";
  default: unknown;
  minimum: number | null;
  maximum: number | null;
  choices: string[];
  optional: boolean;
  description: string;
}

export interface Operation {
  name: string;
  title: string;
  description: string;
  params: OperationParam[];
}

/** A reference parameter: a folder or file path, or several file paths. */
export type Reference = string | string[];

/** One pipeline step, as in a pipeline file. */
export interface Step {
  op: string;
  params: Record<string, unknown>;
  enabled: boolean;
}

/** POST /api/jobs/run: apply steps to a whole dataset (or `frames`, a Python slice of the first axis). */
export interface RunRequest {
  dataset_id: string;
  steps: Step[];
  destination: string;
  dtype: OutputType;
  value_range: [number, number] | null;
  frames: [number | null, number | null, number | null] | null;
  overwrite: boolean;
}

export type OutputType = "float32" | "uint16" | "uint8";

export interface Preview {
  pipeline: string;
  height: number;
  width: number;
  levels: number;
  stats: FrameStats;
  timings: { op: string; seconds: number }[];
}

export type ImageFormat = "png" | "jpeg";

/** Tone curve: [x, y] points in 0..1 with increasing x. */
export type Curve = [number, number][];

/** Photo-style tone adjustments between vmin (0) and vmax (1); see `tone_map` in pd_xray.processing. */
export interface Tone {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  curve: Curve;
}

/** How frames are shown. Never changes the data. */
export interface Display {
  vmin: number;
  vmax: number;
  cmap: string;
  format: ImageFormat;
  tone: Tone;
}

export interface Histogram {
  low: number;
  high: number;
  counts: number[];
}

/** Samples of a tone setting over 0..1: the curve alone, and every adjustment together. */
export interface ToneCurve {
  curve: number[];
  response: number[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null = null, // HTTP status, or null if the server couldn't be reached
    readonly body: Record<string, unknown> | null = null, // the server's JSON error, e.g. {error, fingerprint}
  ) {
    super(message);
  }
}

async function request<T>(method: string, url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error; // the caller cancelled; not a connection problem
    throw new ApiError("Can't reach the server. Is `pd-xray serve` running?");
  }
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    let data: Record<string, unknown> | null = null;
    try {
      const json: unknown = await response.json();
      if (json && typeof json === "object") data = json as Record<string, unknown>;
      if (data && "detail" in data) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(detail, response.status, data);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

const query = (params: Record<string, string | number>) =>
  new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));

export const api = {
  info: () => request<ServerInfo>("GET", "/api/info"),
  list: (path: string | null) =>
    request<Listing>("GET", path === null ? "/api/files" : `/api/files?${query({ path })}`),
  connections: () => request<Connection[]>("GET", "/api/connections"),
  connect: (kind: ConnectionKind, name: string, settings: ConnectionSettings, credentials: Record<string, string>) =>
    request<Connection>("POST", "/api/connections", { kind, name, settings, credentials }),
  disconnect: (id: string) => request<void>("DELETE", `/api/connections/${id}`),
  listRemote: (id: string, path: string) => request<Listing>("GET", `/api/connections/${id}/files?${query({ path })}`),
  inspect: (path: string) => request<Inspection>("GET", `/api/inspect?${query({ path })}`),
  open: (path: string, options: { dataset?: string; pattern?: string } = {}) =>
    request<DatasetInfo>("POST", "/api/datasets", { path, ...options }),
  dataset: (id: string) => request<DatasetInfo>("GET", `/api/datasets/${id}`),
  close: (id: string) => request<void>("DELETE", `/api/datasets/${id}`),
  stats: (id: string, frame: number[], pipeline?: string) =>
    request<FrameStats>(
      "GET",
      `/api/datasets/${id}/frames/${frameKey(frame)}/stats` + (pipeline ? `?${query({ pipeline })}` : ""),
    ),
  histogram: (
    id: string,
    frame: number[],
    range: { low: number; high: number } | null,
    pipeline?: string,
    signal?: AbortSignal,
  ) =>
    request<Histogram>(
      "GET",
      `/api/datasets/${id}/frames/${frameKey(frame)}/histogram?` +
        query({ bins: 256, ...(range ?? {}), ...(pipeline ? { pipeline } : {}) }),
      undefined,
      signal,
    ),
  toneCurve: (tone: Tone, signal?: AbortSignal) =>
    request<ToneCurve>(
      "GET",
      `/api/tone-curve?${query({ tone: JSON.stringify(tone), points: 129 })}`,
      undefined,
      signal,
    ),
  operations: () => request<Operation[]>("GET", "/api/operations"),
  preview: (id: string, frame: number[], steps: Step[], client: string, signal?: AbortSignal) =>
    request<Preview>("POST", `/api/datasets/${id}/frames/${frameKey(frame)}/preview`, { steps, client }, signal),
  valueRange: (datasetId: string) => request<Job>("POST", "/api/jobs/value-range", { dataset_id: datasetId }),
  /** Job: copy `source` into the folder `destination`, between any two places. */
  copy: (source: Location, destination: Location, overwrite: boolean, verify: boolean) =>
    request<Job>("POST", "/api/jobs/copy", { source, destination, overwrite, verify }),
  /** Steps of a pipeline file; references may be relative to the dataset's folder. */
  loadPipeline: (path: string) => request<{ path: string; steps: Step[] }>("GET", `/api/pipelines?${query({ path })}`),
  /** Save steps; references inside the dataset's folder are stored relative to it. 409 if the file exists. */
  savePipeline: (path: string, steps: Step[], datasetId: string, overwrite: boolean) =>
    request<{ path: string }>("POST", "/api/pipelines", { path, steps, dataset_id: datasetId, overwrite }),
  run: (body: RunRequest) => request<Job>("POST", "/api/jobs/run", body),
  cancelJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/cancel`),
};

/** Frame index in URL form: "12", "3,40", or "-" for a single-frame dataset. */
export function frameKey(frame: number[]): string {
  return frame.length ? frame.join(",") : "-";
}

/**
 * URL of one tile. `level` is the server's level: 0 is full resolution, n is shrunk by 2**n. `pipeline` (from a
 * preview) shows the processed frame.
 */
export function tileUrl(
  id: string,
  frame: number[],
  level: number,
  column: number,
  row: number,
  display: Display,
  pipeline?: string,
): string {
  const params = query({
    vmin: display.vmin,
    vmax: display.vmax,
    cmap: display.cmap,
    ...(pipeline ? { pipeline } : {}),
    ...(isDefaultTone(display.tone) ? {} : { tone: JSON.stringify(display.tone) }),
  });
  return `/api/datasets/${id}/frames/${frameKey(frame)}/tiles/${level}/${column}_${row}.${display.format}?${params}`;
}

export const DEFAULT_TONE: Tone = {
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  curve: [
    [0, 0],
    [1, 1],
  ],
};

export function isDefaultTone(tone: Tone): boolean {
  const { curve, ...sliders } = tone;
  return Object.values(sliders).every((v) => v === 0) && JSON.stringify(curve) === JSON.stringify(DEFAULT_TONE.curve);
}
