/**
 * Transactions list page (Phase 3F).
 *
 * Slice 3F-2 wires the filter chrome, URL-synced state, and a count
 * summary. Slice 3F-3 swaps the table body for actual rows and adds
 * "Load more" pagination.
 */
import { TransactionsFilters } from "../components/transactions/TransactionsFilters";
import { FilterChips } from "../components/transactions/FilterChips";
import { TransactionsTable } from "../components/transactions/TransactionsTable";
import { useTransactionFilters } from "../hooks/useTransactionFilters";
import { useTransactionsList } from "../hooks/useTransactionsList";

export function TransactionsPage() {
  const { filters } = useTransactionFilters();
  const query = useTransactionsList(filters);

  const loadedCount =
    query.data?.pages.reduce((sum, page) => sum + page.items.length, 0) ?? 0;

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
        loadedCount={loadedCount}
        isLoading={query.isPending}
        isError={query.isError}
        errorMessage={query.error?.message}
      />
    </div>
  );
}
