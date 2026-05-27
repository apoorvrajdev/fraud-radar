/**
 * Transactions list page (Phase 3F).
 *
 * Composes the URL-synced filter chrome with the live infinite-query
 * table. Each filter change re-keys the query (`useTransactionsList`)
 * so the cursor walk resets cleanly.
 */
import { useMemo } from "react";
import { TransactionsFilters } from "../components/transactions/TransactionsFilters";
import { FilterChips } from "../components/transactions/FilterChips";
import { TransactionsTable } from "../components/transactions/TransactionsTable";
import { useTransactionFilters } from "../hooks/useTransactionFilters";
import { useTransactionsList } from "../hooks/useTransactionsList";

export function TransactionsPage() {
  const { filters } = useTransactionFilters();
  const query = useTransactionsList(filters);

  const rows = useMemo(
    () => query.data?.pages.flatMap((page) => page.items) ?? [],
    [query.data],
  );

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Transactions</h1>
        <p className="text-sm text-neutral-500 mt-1">
          Live feed of scored transactions. Filter to narrow the view —
          the URL updates so you can share it.
        </p>
      </div>

      <TransactionsFilters />
      <FilterChips />

      <TransactionsTable
        rows={rows}
        isLoading={query.isPending}
        isError={query.isError}
        errorMessage={query.error?.message}
        hasMore={query.hasNextPage}
        isFetchingMore={query.isFetchingNextPage}
        onLoadMore={() => query.fetchNextPage()}
      />
    </div>
  );
}
