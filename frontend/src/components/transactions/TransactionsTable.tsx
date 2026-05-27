/**
 * Skeleton table for the transactions page (Phase 3F-2).
 *
 * Renders the column headers and a small count summary so the page
 * has visible structure while we plumb live data. Slice 3F-3 swaps
 * the body for actual rows + the "Load more" pagination control.
 */
import { Card } from "../ui/Card";
import { cn } from "../../lib/cn";

const COLUMNS = [
  "Time",
  "Transaction",
  "Customer",
  "Merchant",
  "Amount",
  "Country",
  "Decision",
] as const;

interface Props {
  /** Number of rows already fetched across all pages. */
  loadedCount: number;
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
}

export function TransactionsTable({
  loadedCount,
  isLoading,
  isError,
  errorMessage,
}: Props) {
  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800">
        <div className="text-sm text-neutral-300">
          {isLoading && loadedCount === 0 ? (
            <span className="text-neutral-500">Loading transactions…</span>
          ) : isError ? (
            <span className="text-rose-400">
              {errorMessage ?? "Failed to load transactions"}
            </span>
          ) : (
            <span>
              <span className="font-medium text-neutral-100 tabular-nums">
                {loadedCount}
              </span>
              <span className="text-neutral-500">
                {" "}
                transaction{loadedCount === 1 ? "" : "s"} loaded
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
              {COLUMNS.map((col) => (
                <th key={col} className="px-4 py-2 font-medium">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading && loadedCount === 0 ? (
              <SkeletonRows />
            ) : loadedCount === 0 && !isError ? (
              <tr>
                <td
                  colSpan={COLUMNS.length}
                  className="px-4 py-10 text-center text-sm text-neutral-500"
                >
                  No transactions match these filters.
                </td>
              </tr>
            ) : (
              <tr>
                <td
                  colSpan={COLUMNS.length}
                  className="px-4 py-10 text-center text-sm text-neutral-500"
                >
                  Rows render in Slice 3F-3.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 6 }).map((_, i) => (
        <tr
          key={i}
          className={cn(
            "border-b border-neutral-900",
            i === 5 && "border-b-0",
          )}
        >
          {COLUMNS.map((col) => (
            <td key={col} className="px-4 py-3">
              <div className="h-3 w-24 rounded bg-neutral-800/70 animate-pulse" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}
