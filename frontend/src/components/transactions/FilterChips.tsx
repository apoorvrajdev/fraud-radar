/**
 * Inline pill row showing active filters. Click the X to clear one.
 *
 * Renders nothing when no filters are active so the page chrome stays
 * tight in the default view.
 */
import { X } from "lucide-react";
import {
  useTransactionFilters,
  type FilterKey,
} from "../../hooks/useTransactionFilters";
import { cn } from "../../lib/cn";

const LABELS: Record<FilterKey, string> = {
  decision: "Decision",
  country: "Country",
  min_amount: "Min",
  max_amount: "Max",
  start_time: "From",
  end_time: "Until",
  customer_id: "Customer",
  merchant_id: "Merchant",
};

const CHIP_ORDER: readonly FilterKey[] = [
  "decision",
  "country",
  "min_amount",
  "max_amount",
  "start_time",
  "end_time",
  "customer_id",
  "merchant_id",
];

function formatValue(key: FilterKey, value: string): string {
  if (key === "min_amount" || key === "max_amount") return `$${value}`;
  if (key === "customer_id" || key === "merchant_id") {
    return value.length > 12 ? `${value.slice(0, 8)}…` : value;
  }
  return value;
}

export function FilterChips() {
  const { filters, clearFilter } = useTransactionFilters();

  const active = CHIP_ORDER.flatMap((key) => {
    const value = filters[key];
    return value ? [{ key, value }] : [];
  });

  if (active.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {active.map(({ key, value }) => (
        <span
          key={key}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full",
            "border border-neutral-800 bg-neutral-900",
            "pl-2.5 pr-1.5 py-1 text-xs text-neutral-300",
          )}
        >
          <span className="text-neutral-500">{LABELS[key]}:</span>
          <span className="font-medium tabular-nums">
            {formatValue(key, value)}
          </span>
          <button
            type="button"
            onClick={() => clearFilter(key)}
            aria-label={`Clear ${LABELS[key]} filter`}
            className="rounded-full p-0.5 hover:bg-neutral-800"
          >
            <X size={12} />
          </button>
        </span>
      ))}
    </div>
  );
}
