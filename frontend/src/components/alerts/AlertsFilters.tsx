/**
 * Filter topbar for the alerts queue (Phase 3H). Same idioms as
 * `TransactionsFilters`: text inputs stay uncontrolled and reset via
 * the React `key` pattern, the select controls are fully URL-driven,
 * `Clear all` resets the whole bar in one click.
 */
import type { FormEvent } from "react";
import { Filter, X } from "lucide-react";
import {
  useAlertsFilters,
  type AgeBand,
} from "../../hooks/useAlertsFilters";
import { cn } from "../../lib/cn";

const AGE_BANDS: { value: AgeBand; label: string }[] = [
  { value: "", label: "Any age" },
  { value: "last_hour", label: "In the last hour" },
  { value: "older_than_1h", label: "Older than 1h" },
  { value: "older_than_24h", label: "Older than 24h" },
];

const inputClass = cn(
  "h-9 rounded-md border border-neutral-800 bg-neutral-950 px-2.5",
  "text-sm text-neutral-100 placeholder:text-neutral-600",
  "focus:outline-none focus:border-neutral-600",
);

export function AlertsFilters() {
  const {
    filters,
    ageBand,
    activeCount,
    setFilter,
    setAgeBand,
    clearAll,
  } = useAlertsFilters();

  const commit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const minScore = (form.elements.namedItem("min_score") as HTMLInputElement)
      .value.trim();
    const country = (form.elements.namedItem("country") as HTMLInputElement)
      .value.trim().toUpperCase();
    setFilter("min_score", minScore || undefined);
    setFilter("country", country || undefined);
  };

  return (
    <form
      key={`${filters.min_score ?? ""}|${filters.country ?? ""}`}
      onSubmit={commit}
      onBlur={(e) => {
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
          Min score
        </span>
        <input
          name="min_score"
          defaultValue={filters.min_score ?? ""}
          placeholder="0.00"
          inputMode="decimal"
          className={cn(inputClass, "w-24 tabular-nums")}
        />
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
          Age
        </span>
        <select
          value={ageBand}
          onChange={(e) => setAgeBand(e.target.value as AgeBand)}
          className={cn(inputClass, "w-44")}
        >
          {AGE_BANDS.map((b) => (
            <option key={b.value} value={b.value}>
              {b.label}
            </option>
          ))}
        </select>
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
