/**
 * URL <-> filter state bridge for the transactions page.
 *
 * Filter state lives in `useSearchParams` so refreshing the page or
 * sharing a URL restores the same view. Components consume `filters`
 * as a plain object and call `setFilter(key, value)` /
 * `clearFilter(key)` / `clearAll()`; the hook hides the URL encoding.
 *
 * Reading is synchronous (derived from the current location); writing
 * is asynchronous (round-trips through React Router). The `decision`
 * filter is validated against the enum so a hand-crafted URL with a
 * bad value falls back to `undefined` rather than poisoning the query.
 */
import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import type { Decision, TransactionListFilters } from "../types/api";

const FILTER_KEYS = [
  "decision",
  "country",
  "min_amount",
  "max_amount",
  "start_time",
  "end_time",
  "customer_id",
  "merchant_id",
] as const satisfies readonly (keyof TransactionListFilters)[];

export type FilterKey = (typeof FILTER_KEYS)[number];

const VALID_DECISIONS: readonly Decision[] = [
  "APPROVE",
  "REVIEW",
  "DECLINE",
  "PENDING",
];

function parseDecision(raw: string | null): Decision | undefined {
  if (raw == null) return undefined;
  return (VALID_DECISIONS as readonly string[]).includes(raw)
    ? (raw as Decision)
    : undefined;
}

export interface UseTransactionFiltersResult {
  filters: TransactionListFilters;
  activeCount: number;
  setFilter: (key: FilterKey, value: string | undefined) => void;
  clearFilter: (key: FilterKey) => void;
  clearAll: () => void;
}

export function useTransactionFilters(): UseTransactionFiltersResult {
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<TransactionListFilters>(() => {
    return {
      decision: parseDecision(searchParams.get("decision")),
      country: searchParams.get("country") ?? undefined,
      min_amount: searchParams.get("min_amount") ?? undefined,
      max_amount: searchParams.get("max_amount") ?? undefined,
      start_time: searchParams.get("start_time") ?? undefined,
      end_time: searchParams.get("end_time") ?? undefined,
      customer_id: searchParams.get("customer_id") ?? undefined,
      merchant_id: searchParams.get("merchant_id") ?? undefined,
    };
  }, [searchParams]);

  const activeCount = useMemo(
    () =>
      FILTER_KEYS.reduce(
        (count, key) => (filters[key] ? count + 1 : count),
        0,
      ),
    [filters],
  );

  const setFilter = useCallback(
    (key: FilterKey, value: string | undefined) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (value === undefined || value === "") {
            next.delete(key);
          } else {
            next.set(key, value);
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const clearFilter = useCallback(
    (key: FilterKey) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.delete(key);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const clearAll = useCallback(() => {
    setSearchParams(new URLSearchParams(), { replace: true });
  }, [setSearchParams]);

  return { filters, activeCount, setFilter, clearFilter, clearAll };
}
