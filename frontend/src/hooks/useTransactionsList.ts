/**
 * useInfiniteQuery wrapper around the transactions list endpoint.
 *
 * Each "page" is a `TransactionList` envelope; `getNextPageParam`
 * reads `next_cursor` off the last page to drive the load-more
 * action wired in Slice 3F-3.
 */
import { useInfiniteQuery } from "@tanstack/react-query";
import { fetchTransactions } from "../lib/transactions";
import type { TransactionList, TransactionListFilters } from "../types/api";

export function useTransactionsList(filters: TransactionListFilters) {
  return useInfiniteQuery<TransactionList, Error>({
    queryKey: ["transactions", filters],
    queryFn: ({ pageParam }) =>
      fetchTransactions({ filters, cursor: pageParam as string | null }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor,
    // 15s feels right for a "what just hit the system" log view —
    // faster than the dashboard's 30s overview because the analyst
    // is staring at the rows.
    refetchInterval: 15_000,
  });
}
