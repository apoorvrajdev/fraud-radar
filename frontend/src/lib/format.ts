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

const PERCENT = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 2,
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

export function formatPercent(value: number): string {
  return PERCENT.format(value);
}
