import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { demoRefetchInterval } from "../lib/demoMode";
import type { StatsOverview } from "../types/api";

/**
 * Live KPI feed. Polls every 30s so the dashboard ticks visibly
 * without the operational surface area of a websocket route. In
 * demo mode polling is disabled (data is bundled and static).
 */
export function useStatsOverview() {
  return useQuery<StatsOverview>({
    queryKey: ["stats", "overview"],
    queryFn: async () => {
      const { data } = await api.get<StatsOverview>("/stats/overview");
      return data;
    },
    refetchInterval: demoRefetchInterval(30_000),
  });
}
