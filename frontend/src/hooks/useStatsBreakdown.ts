import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { StatsBreakdown } from "../types/api";

/**
 * Top-10 country breakdown over the trailing 24h window. Polled at
 * the KPI cadence so the table stays in sync with the headline
 * tiles.
 */
export function useStatsBreakdown() {
  return useQuery<StatsBreakdown>({
    queryKey: ["stats", "breakdown", "country"],
    queryFn: async () => {
      const { data } = await api.get<StatsBreakdown>("/stats/breakdown", {
        params: { dimension: "country" },
      });
      return data;
    },
    refetchInterval: 30_000,
  });
}
