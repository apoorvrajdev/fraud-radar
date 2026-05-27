/**
 * Filter topbar for the transactions page. Controlled by the URL via
 * `useTransactionFilters` — each change rewrites `?key=value` so the
 * view is shareable and refresh-resilient.
 *
 * The text inputs use the React "reset state with key" pattern so they
 * stay locally uncontrolled (no refetch per keystroke) but reset when
 * the URL changes externally — e.g. after Clear all.
 */
import type { FormEvent } from "react";
import { Filter, X } from "lucide-react";
import { useTransactionFilters } from "../../hooks/useTransactionFilters";
import type { Decision } from "../../types/api";
import { cn } from "../../lib/cn";

const DECISIONS: { value: Decision | ""; label: string }[] = [
  { value: "", label: "Any decision" },
  { value: "APPROVE", label: "Approve" },
  { value: "REVIEW", label: "Review" },
  { value: "DECLINE", label: "Decline" },
  { value: "PENDING", label: "Pending" },
];

const inputClass = cn(
  "h-9 rounded-md border border-neutral-800 bg-neutral-950 px-2.5",
  "text-sm text-neutral-100 placeholder:text-neutral-600",
  "focus:outline-none focus:border-neutral-600",
);

export function TransactionsFilters() {
  const { filters, activeCount, setFilter, clearAll } = useTransactionFilters();

  const commit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const country = (form.elements.namedItem("country") as HTMLInputElement)
      .value.trim().toUpperCase();
    const min = (form.elements.namedItem("min_amount") as HTMLInputElement)
      .value.trim();
    const max = (form.elements.namedItem("max_amount") as HTMLInputElement)
      .value.trim();
    setFilter("country", country || undefined);
    setFilter("min_amount", min || undefined);
    setFilter("max_amount", max || undefined);
  };

  return (
    <form
      // Remount the form (and reset uncontrolled inputs) whenever any
      // text-input-backed filter changes externally (e.g. Clear all,
      // chip clear, deep link). Decision-select is fully URL-controlled.
      key={`${filters.country ?? ""}|${filters.min_amount ?? ""}|${filters.max_amount ?? ""}`}
      onSubmit={commit}
      onBlur={(e) => {
        // Commit on any input blur within the form.
        if (e.currentTarget.contains(e.relatedTarget)) return;
        commit(e as unknown as FormEvent<HTMLFormElement>);
      }}
      className="flex flex-wrap items-end gap-3 rounded-xl border border-neutral-800 bg-neutral-900/50 px-4 py-3"
    >
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-neutral-500 mr-1">
        <Filter size={14} aria-hidden />
        Filters
      </div>

      <label className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          Decision
        </span>
        <select
          value={filters.decision ?? ""}
          onChange={(e) => setFilter("decision", e.target.value || undefined)}
          className={cn(inputClass, "w-36")}
        >
          {DECISIONS.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          Country
        </span>
        <input
          name="country"
          defaultValue={filters.country ?? ""}
          placeholder="US"
          maxLength={2}
          className={cn(inputClass, "w-20 uppercase")}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          Min amount
        </span>
        <input
          name="min_amount"
          defaultValue={filters.min_amount ?? ""}
          placeholder="0.00"
          inputMode="decimal"
          className={cn(inputClass, "w-28 tabular-nums")}
        />
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          Max amount
        </span>
        <input
          name="max_amount"
          defaultValue={filters.max_amount ?? ""}
          placeholder="10000.00"
          inputMode="decimal"
          className={cn(inputClass, "w-28 tabular-nums")}
        />
      </label>

      <button type="submit" className="hidden" aria-hidden tabIndex={-1} />

      <div className="flex-1" />

      {activeCount > 0 && (
        <button
          type="button"
          onClick={clearAll}
          className={cn(
            "flex items-center gap-1.5 h-9 rounded-md px-3",
            "text-xs text-neutral-400 hover:text-neutral-100",
            "border border-neutral-800 hover:border-neutral-700",
          )}
        >
          <X size={12} aria-hidden />
          Clear all ({activeCount})
        </button>
      )}
    </form>
  );
}
