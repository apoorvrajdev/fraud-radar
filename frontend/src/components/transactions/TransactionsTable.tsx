/**
 * Transactions table body (Phase 3F-3).
 *
 * Rows render the live API data and link to `/transactions/:id` for
 * the detail view (stub in 3F, real page in 3G). The "Load more"
 * footer drives the `useInfiniteQuery` cursor walk; on failure it
 * shows an inline retry rather than collapsing already-rendered rows.
 */
import { Link } from "react-router-dom";
import { ChevronRight, AlertCircle } from "lucide-react";
import { Card } from "../ui/Card";
import { DecisionBadge } from "./DecisionBadge";
import {
  formatDateTime,
  formatFraudScore,
  formatMoneyPrecise,
} from "../../lib/format";
import { cn } from "../../lib/cn";
import type { TransactionListItem } from "../../types/api";

const COLUMNS = [
  "Time",
  "Transaction",
  "Customer",
  "Amount",
  "Country",
  "Score",
  "Decision",
  "",
] as const;

interface Props {
  rows: TransactionListItem[];
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  hasMore: boolean;
  isFetchingMore: boolean;
  onLoadMore: () => void;
}

export function TransactionsTable({
  rows,
  isLoading,
  isError,
  errorMessage,
  hasMore,
  isFetchingMore,
  onLoadMore,
}: Props) {
  const showSkeleton = isLoading && rows.length === 0;
  const showEmpty = !isLoading && !isError && rows.length === 0;

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800">
        <div className="text-sm text-neutral-300">
          {showSkeleton ? (
            <span className="text-neutral-500">Loading transactions…</span>
          ) : isError && rows.length === 0 ? (
            <span className="text-rose-400">
              {errorMessage ?? "Failed to load transactions"}
            </span>
          ) : (
            <span>
              <span className="font-medium text-neutral-100 tabular-nums">
                {rows.length}
              </span>
              <span className="text-neutral-500">
                {" "}
                transaction{rows.length === 1 ? "" : "s"} loaded
                {hasMore ? " · more available" : ""}
              </span>
            </span>
          )}
        </div>
        <div className="text-[10px] uppercase tracking-wider text-neutral-600">
          Newest first
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
                    col === "Amount" || col === "Score" ? "text-right" : "",
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
                  className="px-4 py-10 text-center text-sm text-neutral-500"
                >
                  No transactions match these filters.
                </td>
              </tr>
            ) : (
              rows.map((row) => <TransactionRow key={row.id} row={row} />)
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

function TransactionRow({ row }: { row: TransactionListItem }) {
  return (
    <tr className="border-b border-neutral-900 last:border-b-0 hover:bg-neutral-900/60">
      <td className="px-4 py-2.5 text-neutral-400 tabular-nums whitespace-nowrap">
        {formatDateTime(row.created_at)}
      </td>
      <td className="px-4 py-2.5 font-mono text-xs text-neutral-300">
        {row.id.slice(0, 8)}
      </td>
      <td className="px-4 py-2.5 font-mono text-xs text-neutral-400">
        {row.customer_id.slice(0, 8)}
      </td>
      <td className="px-4 py-2.5 text-right tabular-nums text-neutral-100">
        {formatMoneyPrecise(row.amount)}
      </td>
      <td className="px-4 py-2.5 text-neutral-300">{row.country}</td>
      <td className="px-4 py-2.5 text-right tabular-nums text-neutral-300">
        {formatFraudScore(row.fraud_score)}
      </td>
      <td className="px-4 py-2.5">
        <DecisionBadge decision={row.fraud_decision} />
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
