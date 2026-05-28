/**
 * Single-transaction detail fetcher (Phase 3G).
 *
 * Lighter polling than the list view (60 s) — the detail page is a
 * focused read where staleness matters less than for the live feed,
 * and most of the time analysts are reading, not waiting for changes.
 * The 3G-4 mutation invalidates this query manually on success.
 */
/**
 * Single-transaction detail fetcher (Phase 3G).
 *
 * Lighter polling than the list view (60 s) — the detail page is a
 * focused read where staleness matters less than for the live feed,
 * and most of the time analysts are reading, not waiting for changes.
 * The 3G-4 mutation invalidates this query manually on success.
 */
import { useQuery } from "@tanstack/react-query";
import { fetchTransactionDetail } from "../lib/transactions";
import { demoRefetchInterval } from "../lib/demoMode";
import type { TransactionDetail } from "../types/api";

export function useTransactionDetail(id: string | undefined) {
  return useQuery<TransactionDetail, Error>({
    queryKey: ["transaction-detail", id],
    queryFn: () => fetchTransactionDetail(id as string),
    enabled: Boolean(id),
    refetchInterval: demoRefetchInterval(60_000),
  });
}
