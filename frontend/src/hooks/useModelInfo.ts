/**
 * Model and dataset identity (Phase 5G).
 *
 * The one query in the app with no polling: model artifacts change
 * when someone retrains and redeploys, not while you are looking at
 * the page. A long stale time keeps the badge off the network.
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { ModelInfo } from "../types/api";

async function fetchModelInfo(): Promise<ModelInfo> {
  const { data } = await api.get<ModelInfo>("/model");
  return data;
}

export function useModelInfo() {
  return useQuery({
    queryKey: ["model-info"],
    queryFn: fetchModelInfo,
    staleTime: 60 * 60 * 1000,
    refetchInterval: false,
    retry: 1,
  });
}
