// Number formatting for display.

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit++;
  }
  return `${unit === 0 ? value : value.toPrecision(3)} ${units[unit]}`;
}

/** Up to 6 significant digits, without trailing zeros: 1234, 0.0309411, 1.23457e+07. */
export function formatValue(value: number): string {
  return Number.isInteger(value) && Math.abs(value) < 1e9 ? String(value) : String(Number(value.toPrecision(6)));
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "";
  if (seconds < 60) return `${Math.ceil(seconds)} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ${Math.floor(seconds % 60)} s`;
  return `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min`;
}
