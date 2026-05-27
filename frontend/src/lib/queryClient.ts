/**
 * Shared TanStack Query client.
 *
 * Per-hook `refetchInterval` lives on the individual hooks so each
 * tile can poll at its own cadence (see ADR decision 3). Defaults
 * here only cover retry + stale-time behaviour.
 */
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 25s — slightly under the 30s polling interval so a refetch
      // always fires fresh.
      staleTime: 25_000,
      // One retry on transient failures; the dashboard surfaces
      // errors per-tile rather than blocking the whole screen.
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});
