/**
 * Alerts queue page (Phase 3H).
 *
 * Composition pattern mirrors `TransactionsPage`: URL-bound filter
 * bar feeds an infinite query, summary tiles read off the first
 * page's summary block, the table flattens the cursor-walked pages
 * into a single dense list.
 */
import { useMemo } from "react";
import { useAlertsFilters } from "../hooks/useAlertsFilters";
import { useAlerts } from "../hooks/useAlerts";
import { AlertsSummaryStrip } from "../components/alerts/AlertsSummaryStrip";
import { AlertsFilters } from "../components/alerts/AlertsFilters";
import { AlertsTable } from "../components/alerts/AlertsTable";

export function AlertsPage() {
  const { filters } = useAlertsFilters();
  const query = useAlerts(filters);

  const rows = useMemo(
    () => query.data?.pages.flatMap((page) => page.items) ?? [],
    [query.data],
  );

  // Summary is queue-wide and identical across pages; read off page 0
  // so the strip stays stable during pagination.
  const summary = query.data?.pages[0]?.summary;

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Alerts</h1>
        <p className="text-sm text-neutral-500 mt-1">
          Pending-review queue. Sorted by score, oldest first within a
          tie. Submitting a verdict moves the row off this list.
        </p>
      </div>

      <AlertsSummaryStrip
        summary={summary}
        isLoading={query.isPending}
        isError={query.isError}
      />

      <AlertsFilters />

      <AlertsTable
        rows={rows}
        isLoading={query.isPending}
        isError={query.isError}
        errorMessage={query.error?.message}
        hasMore={query.hasNextPage}
        isFetchingMore={query.isFetchingNextPage}
        onLoadMore={() => query.fetchNextPage()}
        isQueueClear={summary?.pending_count === 0}
      />
    </div>
  );
}
