/**
 * Display formatters for the dashboard.
 *
 * Money values arrive as strings (Decimal precision preserved over
 * the wire). For *display* we accept the loss-of-precision trade-off
 * of `Number(...)` because Intl.NumberFormat requires a number and
 * the dashboard never does math on these values. Any aggregation
 * stays on the backend.
 *
 * Two kinds of money live in this app and they must not be formatted
 * the same way:
 *
 * 1. **A transaction's own amount**, in the currency the cardholder
 *    was charged in. Format it with `formatMoneyIn(value, currency)`.
 *    Formatting one of these as USD is not a cosmetic bug — it states
 *    something false about a payment.
 * 2. **A reporting aggregate**, which the backend already sums in the
 *    reporting currency (`COALESCE(amount_base, amount)` — see
 *    docs/FX_CONTRACT.md). Format it with `formatMoney` /
 *    `formatMoneyPrecise`, which are pinned to REPORTING_CURRENCY.
 */

/**
 * The currency every `/stats/*` money field is denominated in. Mirrors
 * the backend's `FX_BASE_CURRENCY`; changing one without the other
 * would mislabel every aggregate on the dashboard.
 */
export const REPORTING_CURRENCY = "USD";

const COMPACT_NUMBER = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const FULL_NUMBER = new Intl.NumberFormat("en-US");

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: REPORTING_CURRENCY,
  maximumFractionDigits: 0,
});

const USD_PRECISE = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: REPORTING_CURRENCY,
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

/**
 * `Intl.NumberFormat` construction is not cheap and these render in
 * table rows, so formatters are built once per currency and reused.
 */
const CURRENCY_FORMATTERS = new Map<string, Intl.NumberFormat>();

function currencyFormatter(currency: string): Intl.NumberFormat | null {
  const code = currency.toUpperCase();
  const cached = CURRENCY_FORMATTERS.get(code);
  if (cached) return cached;
  try {
    const formatter = new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: code,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    CURRENCY_FORMATTERS.set(code, formatter);
    return formatter;
  } catch {
    // `Intl` throws RangeError on a code it does not recognise. The
    // backend accepts any three-letter code, so this is reachable with
    // real data — fall back rather than blanking the cell.
    return null;
  }
}

/**
 * Format a transaction amount in the currency it was charged in.
 *
 * Returns e.g. "€125.00" for EUR and "¥1,400" for JPY, letting `Intl`
 * pick the symbol and the currency's own decimal convention. An
 * unrecognised code degrades to "125.00 XYZ" rather than throwing or
 * silently rendering a dollar sign.
 */
export function formatMoneyIn(value: string, currency: string): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return `${value} ${currency.toUpperCase()}`;
  const formatter = currencyFormatter(currency);
  if (formatter === null) {
    return `${FULL_NUMBER.format(n)} ${currency.toUpperCase()}`;
  }
  return formatter.format(n);
}

/**
 * True when `Intl` renders this currency with a symbol distinct from
 * its code (so "€125.00" needs the "EUR" suffix to be unambiguous, but
 * a code-rendered currency like "CHF 125.00" already carries it).
 */
export function needsCurrencySuffix(currency: string): boolean {
  const code = currency.toUpperCase();
  const formatter = currencyFormatter(code);
  if (formatter === null) return false; // the fallback already appends it
  return !formatter.format(0).includes(code);
}

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

/**
 * Reporting-currency money, no decimals — for KPI tiles and other
 * aggregates the backend has already summed in REPORTING_CURRENCY.
 * Never use this for a single transaction's amount.
 */
export function formatMoney(value: string): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  return USD.format(n);
}

/**
 * Reporting-currency money with cents, so $412.5 reads as "$412.50".
 * Use for a derived `amount_base`, or for an aggregate that wants the
 * cents; keep `formatMoney` (no decimals) for KPI tiles. For a
 * transaction's original amount use `formatMoneyIn`.
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
