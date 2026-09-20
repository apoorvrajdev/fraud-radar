/**
 * Transaction money display (Phase 5G).
 *
 * The one place the app renders a transaction's amount, so the rule
 * that the original value is authoritative holds everywhere by
 * construction rather than by five call sites agreeing.
 *
 * Four states, from the FX contract (docs/FX_CONTRACT.md):
 *
 * 1. Already in the reporting currency — one line. A second line
 *    saying "≈ $100.00 USD" under "$100.00" is noise, not information.
 * 2. Converted — the original, then the derived reporting figure
 *    marked with "≈" and its currency, visually subordinate so it can
 *    never be mistaken for what the cardholder was charged.
 * 3. Priced with a stale rate — as above, but the marker says so and
 *    carries the rate's date, because a converted figure from an old
 *    rate is an estimate with a knowable age.
 * 4. Not converted, or a row predating enrichment — the original
 *    alone. Nothing is invented, and state 4 is deliberately quiet:
 *    an unconverted row is normal, not an error.
 */
import { cn } from "../../lib/cn";
import {
  REPORTING_CURRENCY,
  formatMoneyIn,
  formatMoneyPrecise,
  needsCurrencySuffix,
} from "../../lib/format";
import type { FxFields } from "../../types/api";

interface Props extends Partial<FxFields> {
  amount: string;
  currency: string;
  /** `stacked` for detail views, `inline` for dense table cells. */
  layout?: "stacked" | "inline";
  className?: string;
}

/** Why there is no converted figure, phrased for a human. */
const NO_RATE_REASON: Record<string, string> = {
  unsupported: `No ${REPORTING_CURRENCY} reference rate is published for this currency, so no converted amount is shown.`,
  unavailable: `No ${REPORTING_CURRENCY} reference rate was available for this transaction's date, so no converted amount is shown.`,
};

export function Amount({
  amount,
  currency,
  amount_base,
  fx_rate,
  fx_rate_date,
  fx_source,
  layout = "inline",
  className,
}: Props) {
  const original = formatMoneyIn(amount, currency);
  const suffix = needsCurrencySuffix(currency) ? currency.toUpperCase() : null;
  const isReportingCurrency =
    currency.toUpperCase() === REPORTING_CURRENCY || fx_source === "identity";

  // State 1 — nothing to convert.
  if (isReportingCurrency) {
    return (
      <span className={cn("tabular-nums text-neutral-100", className)}>
        {original}
      </span>
    );
  }

  const converted = amount_base !== null && amount_base !== undefined;
  const stale = fx_source === "stale";
  const reason = fx_source ? NO_RATE_REASON[fx_source] : undefined;

  const secondary = converted ? (
    <span
      className={cn(
        "tabular-nums",
        stale ? "text-amber-400/80" : "text-neutral-500",
      )}
      title={
        stale
          ? `Converted at the ${fx_rate_date} reference rate (${fx_rate}), the most recent one available for this transaction's date.`
          : `Derived reporting value. Converted at the ${fx_rate_date} reference rate (${fx_rate}).`
      }
    >
      ≈ {formatMoneyPrecise(amount_base)} {REPORTING_CURRENCY}
      {stale && <span className="ml-1 text-[10px] uppercase">stale rate</span>}
    </span>
  ) : reason ? (
    // State 4, with a known reason — muted, and explained on hover.
    <span className="text-neutral-600" title={reason}>
      no {REPORTING_CURRENCY} rate
    </span>
  ) : null;

  if (layout === "stacked") {
    return (
      <span className={cn("block", className)}>
        <span className="block tabular-nums text-neutral-100">
          {original}
          {suffix && (
            <span className="ml-1 text-xs text-neutral-400">{suffix}</span>
          )}
        </span>
        {secondary && (
          <span className="mt-0.5 block text-xs">{secondary}</span>
        )}
      </span>
    );
  }

  return (
    <span className={cn("inline-flex flex-col items-end", className)}>
      <span className="tabular-nums text-neutral-100">
        {original}
        {suffix && (
          <span className="ml-1 text-[10px] text-neutral-400">{suffix}</span>
        )}
      </span>
      {secondary && <span className="text-[10px] leading-tight">{secondary}</span>}
    </span>
  );
}
