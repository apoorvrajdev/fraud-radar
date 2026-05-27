import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { StatsOverview } from "../types/api";

/**
 * Live KPI feed. Polls every 30s so the dashboard ticks visibly
 * without the operational surface area of a websocket route.
 */
export function useStatsOverview() {
  return useQuery<StatsOverview>({
    queryKey: ["stats", "overview"],
    queryFn: async () => {
      const { data } = await api.get<StatsOverview>("/stats/overview");
      return data;
    },
    refetchInterval: 30_000,
  });
}
