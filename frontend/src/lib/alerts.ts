/**
 * Network calls for the analyst alerts queue (Phase 3H).
 *
 * Kept in its own module — same shape as `lib/transactions.ts` —
 * so the per-resource fetchers don't pile up in `lib/api.ts`.
 */
import { api } from "./api";
import type { AlertsFilters, AlertsResponse } from "../types/api";

const PAGE_SIZE = 50;

export interface FetchAlertsParams {
  filters: AlertsFilters;
  cursor?: string | null;
  limit?: number;
}

/**
 * GET /alerts with filters and optional keyset cursor. Empty filter
 * values are stripped so the request URL stays clean and the backend
 * doesn't have to ignore explicit empty strings.
 */
export async function fetchAlerts({
  filters,
  cursor,
  limit = PAGE_SIZE,
}: FetchAlertsParams): Promise<AlertsResponse> {
  const params: Record<string, string | number> = { limit };
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "") {
      params[key] = value;
    }
  }
  if (cursor) {
    params.cursor = cursor;
  }
  const { data } = await api.get<AlertsResponse>("/alerts", { params });
  return data;
}
