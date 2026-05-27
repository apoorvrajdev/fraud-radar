/**
 * Network calls for the transactions list endpoint (Phase 3F).
 *
 * Kept separate from `lib/api.ts` so the centralised axios instance
 * stays a single small file and per-resource fetchers can grow without
 * crowding it.
 */
import { api } from "./api";
import type {
  TransactionDetail,
  TransactionList,
  TransactionListFilters,
} from "../types/api";

const PAGE_SIZE = 50;

export interface FetchTransactionsParams {
  filters: TransactionListFilters;
  cursor?: string | null;
  limit?: number;
}

/**
 * GET /transactions with filters and optional keyset cursor. Empty
 * filter values are stripped so the request URL stays clean and the
 * backend doesn't have to ignore explicit empty strings.
 */
export async function fetchTransactions({
  filters,
  cursor,
  limit = PAGE_SIZE,
}: FetchTransactionsParams): Promise<TransactionList> {
  const params: Record<string, string | number> = { limit };
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "") {
      params[key] = value;
    }
  }
  if (cursor) {
    params.cursor = cursor;
  }
  const { data } = await api.get<TransactionList>("/transactions", { params });
  return data;
}

/**
 * GET /transactions/{id}. Returns the full TransactionDetail envelope
 * (Phase 3G): row fields, rules, top SHAP contributors, audit trail,
 * and any analyst override.
 */
export async function fetchTransactionDetail(
  id: string,
): Promise<TransactionDetail> {
  const { data } = await api.get<TransactionDetail>(`/transactions/${id}`);
  return data;
}
