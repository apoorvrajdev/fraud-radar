/**
 * Display formatters for the dashboard.
 *
 * Money values arrive as strings (Decimal precision preserved over
 * the wire). For *display* we accept the loss-of-precision trade-off
 * of `Number(...)` because Intl.NumberFormat requires a number and
 * the dashboard never does math on these values. Any aggregation
 * stays on the backend.
 */

const COMPACT_NUMBER = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const FULL_NUMBER = new Intl.NumberFormat("en-US");

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const USD_PRECISE = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const PERCENT = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 2,
});

const DATE_TIME_SHORT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export function formatCompactInt(value: number): string {
  return COMPACT_NUMBER.format(value);
}

export function formatInt(value: number): string {
  return FULL_NUMBER.format(value);
}

export function formatMoney(value: string): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  return USD.format(n);
}

/**
 * Per-transaction money display — always shows cents so $412.5 reads
 * as "$412.50". Use this in tables and detail views; keep
 * `formatMoney` (no decimals) for KPI tiles.
 */
export function formatMoneyPrecise(value: string): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  return USD_PRECISE.format(n);
}

export function formatPercent(value: number): string {
  return PERCENT.format(value);
}

/**
 * "May 27, 14:32" — short relative-ish timestamp for table cells.
 * Falls back to the raw string on parse failure.
 */
export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return DATE_TIME_SHORT.format(d);
}

/**
 * Format a fraud score (0–1 string) as a 4-decimal display value.
 * Returns "—" for nulls so empty cells read cleanly.
 */
export function formatFraudScore(value: string | null): string {
  if (value == null) return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  return n.toFixed(4);
}

/**
 * Compact human-readable duration for queue ages, e.g. "5h 12m",
 * "32m", "2d 3h", "just now". The seconds input mirrors the backend
 * `age_seconds` field on alerts items.
 */
export function formatAge(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  if (hours < 24) return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`;
  const days = Math.floor(hours / 24);
  const hrs = hours % 24;
  return hrs > 0 ? `${days}d ${hrs}h` : `${days}d`;
}
