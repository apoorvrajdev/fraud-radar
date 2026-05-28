/**
 * useInfiniteQuery wrapper around the alerts endpoint (Phase 3H).
 *
 * Each "page" is an `AlertsResponse`. The summary block is read off
 * the *first* page so the header strip stays stable as the analyst
 * paginates through the queue (subsequent pages carry the same
 * summary, but reading from page 0 avoids a flicker on slow
 * networks).
 */
/**
 * useInfiniteQuery wrapper around the alerts endpoint (Phase 3H).
 *
 * Each "page" is an `AlertsResponse`. The summary block is read off
 * the *first* page so the header strip stays stable as the analyst
 * paginates through the queue (subsequent pages carry the same
 * summary, but reading from page 0 avoids a flicker on slow
 * networks).
 */
import { useInfiniteQuery } from "@tanstack/react-query";
import { fetchAlerts } from "../lib/alerts";
import { demoRefetchInterval } from "../lib/demoMode";
import type { AlertsFilters, AlertsResponse } from "../types/api";

export function useAlerts(filters: AlertsFilters) {
  return useInfiniteQuery<AlertsResponse, Error>({
    queryKey: ["alerts", filters],
    queryFn: ({ pageParam }) =>
      fetchAlerts({ filters, cursor: pageParam as string | null }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor,
    // 10s stale lets filter focus events skip refetches; 15s polling
    // keeps the queue fresh during an active triage session. TanStack
    // automatically pauses polling when the tab is hidden. Polling
    // is disabled in demo mode.
    staleTime: 10_000,
    refetchInterval: demoRefetchInterval(15_000),
  });
}
