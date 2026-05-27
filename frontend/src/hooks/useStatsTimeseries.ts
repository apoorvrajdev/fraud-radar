import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { StatsTimeseries } from "../types/api";

/**
 * 24-hour fraud-rate + volume timeseries, polled every 60s. Slower
 * cadence than the KPI tiles because hour buckets only roll over
 * once per hour anyway.
 */
export function useStatsTimeseries() {
  return useQuery<StatsTimeseries>({
    queryKey: ["stats", "timeseries", "24h", "1h"],
    queryFn: async () => {
      const { data } = await api.get<StatsTimeseries>("/stats/timeseries", {
        params: { window: "24h", bucket: "1h" },
      });
      return data;
    },
    refetchInterval: 60_000,
  });
}
