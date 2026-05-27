/**
 * Alerts queue table (Phase 3H).
 *
 * Same dense layout and load-more flow as `TransactionsTable`. The
 * row order is the backend's: score DESC, then created_at ASC, then
 * id ASC — so the highest-confidence-and-oldest alerts surface
 * first. Rows are full-row links into the existing transaction
 * detail page where the analyst submits a verdict.
 */
import { Link } from "react-router-dom";
import { ChevronRight, AlertCircle } from "lucide-react";
import { Card } from "../ui/Card";
import { ScoreChip } from "./ScoreChip";
import { formatAge, formatMoneyPrecise } from "../../lib/format";
import { cn } from "../../lib/cn";
import type { AlertItem } from "../../types/api";

const COLUMNS = [
  "Age",
  "Score",
  "Amount",
  "Country",
  "Rules",
  "Customer",
  "",
] as const;

interface Props {
  rows: AlertItem[];
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  hasMore: boolean;
  isFetchingMore: boolean;
  onLoadMore: () => void;
  /**
   * True when the queue itself is empty (summary.pending_count === 0),
   * not just when the current filters produced no matches. Drives the
   * celebratory empty state copy.
   */
  isQueueClear?: boolean;
}

export function AlertsTable({
  rows,
  isLoading,
  isError,
  errorMessage,
  hasMore,
  isFetchingMore,
  onLoadMore,
  isQueueClear = false,
}: Props) {
  const showSkeleton = isLoading && rows.length === 0;
  const showEmpty = !isLoading && !isError && rows.length === 0;

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800">
        <div className="text-sm text-neutral-300">
          {showSkeleton ? (
            <span className="text-neutral-500">Loading alerts…</span>
          ) : isError && rows.length === 0 ? (
            <span className="text-rose-400">
              {errorMessage ?? "Failed to load alerts"}
            </span>
          ) : (
            <span>
              <span className="font-medium text-neutral-100 tabular-nums">
                {rows.length}
              </span>
              <span className="text-neutral-500">
                {" "}
                alert{rows.length === 1 ? "" : "s"} loaded
                {hasMore ? " · more available" : ""}
              </span>
            </span>
          )}
        </div>
        <div className="text-[10px] uppercase tracking-wider text-neutral-600">
          Highest score first
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-800">
              {COLUMNS.map((col, i) => (
                <th
                  key={i}
                  className={cn(
                    "px-4 py-2 font-medium",
                    col === "Amount" ? "text-right" : "",
                  )}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {showSkeleton ? (
              <SkeletonRows />
            ) : showEmpty ? (
              <tr>
                <td
                  colSpan={COLUMNS.length}
                  className="px-4 py-12 text-center"
                >
                  {isQueueClear ? (
                    <div className="space-y-1">
                      <div className="text-sm font-medium text-emerald-300">
                        Queue clear.
                      </div>
                      <div className="text-xs text-neutral-500">
                        No transactions are waiting on a review.
                      </div>
                    </div>
                  ) : (
                    <span className="text-sm text-neutral-500">
                      No alerts match these filters.
                    </span>
                  )}
                </td>
              </tr>
            ) : (
              rows.map((row) => <AlertRow key={row.id} row={row} />)
            )}
          </tbody>
        </table>
      </div>

      {rows.length > 0 && (
        <div className="border-t border-neutral-800 px-4 py-3">
          {isError ? (
            <LoadMoreError
              message={errorMessage ?? "Failed to fetch next page"}
              onRetry={onLoadMore}
            />
          ) : hasMore ? (
            <button
              type="button"
              onClick={onLoadMore}
              disabled={isFetchingMore}
              className={cn(
                "w-full rounded-md border border-neutral-800 bg-neutral-900",
                "px-3 py-2 text-sm text-neutral-300",
                "hover:border-neutral-700 hover:text-neutral-100",
                "disabled:opacity-60 disabled:cursor-not-allowed",
                "transition-colors",
              )}
            >
              {isFetchingMore ? "Loading…" : "Load more"}
            </button>
          ) : (
            <div className="text-center text-xs text-neutral-600">
              End of results
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function AlertRow({ row }: { row: AlertItem }) {
  const rules = row.rules_triggered;
  const visibleRules = rules.slice(0, 2);
  const extra = rules.length - visibleRules.length;

  return (
    <tr className="border-b border-neutral-900 last:border-b-0 hover:bg-neutral-900/60">
      <td className="px-4 py-2.5 text-neutral-300 tabular-nums whitespace-nowrap">
        {formatAge(row.age_seconds)}
      </td>
      <td className="px-4 py-2.5">
        <ScoreChip score={row.fraud_score} />
      </td>
      <td className="px-4 py-2.5 text-right tabular-nums text-neutral-100">
        {formatMoneyPrecise(row.amount)}
      </td>
      <td className="px-4 py-2.5 text-neutral-300">{row.country}</td>
      <td className="px-4 py-2.5 text-neutral-400">
        {rules.length === 0 ? (
          <span className="text-neutral-600">—</span>
        ) : (
          <span className="text-xs">
            {visibleRules.join(", ")}
            {extra > 0 && (
              <span className="text-neutral-600"> · +{extra} more</span>
            )}
          </span>
        )}
      </td>
      <td className="px-4 py-2.5 font-mono text-xs text-neutral-400">
        {row.customer_id.slice(0, 8)}
      </td>
      <td className="px-4 py-2.5 text-right">
        <Link
          to={`/transactions/${row.id}`}
          className="inline-flex text-neutral-500 hover:text-neutral-100"
          aria-label={`Open transaction ${row.id}`}
        >
          <ChevronRight size={16} />
        </Link>
      </td>
    </tr>
  );
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 8 }).map((_, i) => (
        <tr key={i} className="border-b border-neutral-900 last:border-b-0">
          {COLUMNS.map((_col, j) => (
            <td key={j} className="px-4 py-3">
              <div className="h-3 w-24 rounded bg-neutral-800/70 animate-pulse" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

function LoadMoreError({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <div className="flex items-center gap-2 text-rose-400">
        <AlertCircle size={14} aria-hidden />
        <span>{message}</span>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className={cn(
          "rounded-md border border-neutral-800 bg-neutral-900",
          "px-3 py-1 text-xs text-neutral-300",
          "hover:border-neutral-700 hover:text-neutral-100",
        )}
      >
        Retry
      </button>
    </div>
  );
}
