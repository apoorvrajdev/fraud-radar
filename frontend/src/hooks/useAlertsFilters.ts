/**
 * URL <-> filter state bridge for the alerts queue (Phase 3H).
 *
 * Same shape and idioms as `useTransactionFilters` so the two
 * filter bars feel identical to use — values live in
 * `useSearchParams`, the hook exposes a plain object plus
 * `setFilter` / `clearAll`, and an "age band" select is folded
 * over the two numeric `min_age_seconds` / `max_age_seconds`
 * parameters under the hood.
 */
import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import type { AlertsFilters } from "../types/api";

const FILTER_KEYS = [
  "min_score",
  "country",
  "min_age_seconds",
  "max_age_seconds",
] as const satisfies readonly (keyof AlertsFilters)[];

export type AlertsFilterKey = (typeof FILTER_KEYS)[number];

export type AgeBand =
  | ""
  | "last_hour"
  | "older_than_1h"
  | "older_than_24h";

/**
 * Map a single user-facing "age band" choice to the two numeric
 * params the backend understands. Keeping this conversion here
 * means the URL stays in the backend-aligned shape (so deep links
 * remain valid even if the band labels change later) while the UI
 * works with a single friendly control.
 */
export function ageBandToParams(band: AgeBand): {
  min_age_seconds?: number;
  max_age_seconds?: number;
} {
  switch (band) {
    case "last_hour":
      return { max_age_seconds: 3600 };
    case "older_than_1h":
      return { min_age_seconds: 3600 };
    case "older_than_24h":
      return { min_age_seconds: 86400 };
    case "":
    default:
      return {};
  }
}

/** Reverse `ageBandToParams` so the select reflects the URL state. */
export function paramsToAgeBand(
  filters: Pick<AlertsFilters, "min_age_seconds" | "max_age_seconds">,
): AgeBand {
  if (filters.max_age_seconds === 3600 && filters.min_age_seconds == null) {
    return "last_hour";
  }
  if (filters.min_age_seconds === 86400) return "older_than_24h";
  if (filters.min_age_seconds === 3600) return "older_than_1h";
  return "";
}

export interface UseAlertsFiltersResult {
  filters: AlertsFilters;
  ageBand: AgeBand;
  activeCount: number;
  setFilter: (
    key: AlertsFilterKey,
    value: string | number | undefined,
  ) => void;
  setAgeBand: (band: AgeBand) => void;
  clearAll: () => void;
}

function parsePositiveInt(raw: string | null): number | undefined {
  if (raw == null) return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) return undefined;
  return n;
}

export function useAlertsFilters(): UseAlertsFiltersResult {
  const [searchParams, setSearchParams] = useSearchParams();

  const filters = useMemo<AlertsFilters>(() => {
    return {
      min_score: searchParams.get("min_score") ?? undefined,
      country: searchParams.get("country") ?? undefined,
      min_age_seconds: parsePositiveInt(searchParams.get("min_age_seconds")),
      max_age_seconds: parsePositiveInt(searchParams.get("max_age_seconds")),
    };
  }, [searchParams]);

  const ageBand = useMemo(() => paramsToAgeBand(filters), [filters]);

  const activeCount = useMemo(
    () =>
      FILTER_KEYS.reduce(
        (count, key) => (filters[key] != null ? count + 1 : count),
        0,
      ),
    [filters],
  );

  const setFilter = useCallback(
    (key: AlertsFilterKey, value: string | number | undefined) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (value === undefined || value === null || value === "") {
            next.delete(key);
          } else {
            next.set(key, String(value));
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const setAgeBand = useCallback(
    (band: AgeBand) => {
      const { min_age_seconds, max_age_seconds } = ageBandToParams(band);
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.delete("min_age_seconds");
          next.delete("max_age_seconds");
          if (min_age_seconds != null) {
            next.set("min_age_seconds", String(min_age_seconds));
          }
          if (max_age_seconds != null) {
            next.set("max_age_seconds", String(max_age_seconds));
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const clearAll = useCallback(() => {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        for (const key of FILTER_KEYS) next.delete(key);
        return next;
      },
      { replace: true },
    );
  }, [setSearchParams]);

  return { filters, ageBand, activeCount, setFilter, setAgeBand, clearAll };
}
