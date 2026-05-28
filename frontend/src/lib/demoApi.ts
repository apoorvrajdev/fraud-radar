/**
 * Axios adapter that resolves API requests from the bundled JSON
 * snapshot under `public/demo-data/` instead of hitting a backend.
 *
 * Installed on the central `api` instance only when
 * `isDemoMode() === true`. Hooks and feature code remain unchanged —
 * all branching lives here.
 *
 * URL contract is locked in `docs/adr/PHASE_4A_DEMO_SCOPE.md`. Do not
 * add new endpoints without updating the ADR and re-running the
 * Phase 4B export script.
 */
import type {
  AxiosAdapter,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from "axios";
import { AxiosHeaders } from "axios";
import type {
  AlertItem,
  AlertsFilters,
  AlertsResponse,
  TransactionList,
  TransactionListFilters,
  TransactionListItem,
} from "../types/api";

const SNAPSHOT_BASE = "/demo-data";

class DemoNotFoundError extends Error {
  readonly path: string;
  constructor(path: string) {
    super(`Demo snapshot has no file at ${path}`);
    this.path = path;
  }
}

async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(`${SNAPSHOT_BASE}${path}`);
  if (!response.ok) {
    if (response.status === 404) throw new DemoNotFoundError(path);
    throw new Error(`Failed to load demo snapshot ${path}: ${response.status}`);
  }
  return (await response.json()) as T;
}

function asResponse<T>(
  data: T,
  config: InternalAxiosRequestConfig,
  status = 200,
): AxiosResponse<T> {
  return {
    data,
    status,
    statusText: status === 200 ? "OK" : "Not Found",
    headers: {},
    config,
    request: null,
  };
}

// ---------------------------------------------------------------------------
// Filter helpers
// ---------------------------------------------------------------------------

function filterTransactions(
  rows: TransactionListItem[],
  filters: TransactionListFilters,
): TransactionListItem[] {
  return rows.filter((row) => {
    if (filters.decision && row.fraud_decision !== filters.decision)
      return false;
    if (filters.country && row.country !== filters.country) return false;
    if (filters.customer_id && row.customer_id !== filters.customer_id)
      return false;
    if (filters.merchant_id && row.merchant_id !== filters.merchant_id)
      return false;
    if (filters.min_amount !== undefined) {
      if (parseFloat(row.amount) < parseFloat(filters.min_amount)) return false;
    }
    if (filters.max_amount !== undefined) {
      if (parseFloat(row.amount) > parseFloat(filters.max_amount)) return false;
    }
    if (filters.start_time && row.created_at < filters.start_time) return false;
    if (filters.end_time && row.created_at > filters.end_time) return false;
    return true;
  });
}

function filterAlerts(
  rows: AlertItem[],
  filters: AlertsFilters,
): AlertItem[] {
  return rows.filter((row) => {
    if (filters.country && row.country !== filters.country) return false;
    if (filters.min_score !== undefined) {
      if (parseFloat(row.fraud_score) < parseFloat(filters.min_score))
        return false;
    }
    if (filters.min_age_seconds !== undefined) {
      if (row.age_seconds < filters.min_age_seconds) return false;
    }
    if (filters.max_age_seconds !== undefined) {
      if (row.age_seconds > filters.max_age_seconds) return false;
    }
    return true;
  });
}

// ---------------------------------------------------------------------------
// Request matchers
// ---------------------------------------------------------------------------

const TX_DETAIL_RE = /^\/transactions\/([^/]+)$/;

interface RouteHandler {
  (config: InternalAxiosRequestConfig): Promise<AxiosResponse>;
}

function getParam(config: AxiosRequestConfig, key: string): string | undefined {
  const value = (config.params as Record<string, unknown> | undefined)?.[key];
  if (value === undefined || value === null || value === "") return undefined;
  return String(value);
}

function paramNumber(
  config: AxiosRequestConfig,
  key: string,
): number | undefined {
  const raw = getParam(config, key);
  return raw === undefined ? undefined : Number(raw);
}

async function handleStatsOverview(
  config: InternalAxiosRequestConfig,
): Promise<AxiosResponse> {
  return asResponse(await loadJson("/stats-overview.json"), config);
}

async function handleStatsTimeseries(
  config: InternalAxiosRequestConfig,
): Promise<AxiosResponse> {
  return asResponse(await loadJson("/stats-timeseries.json"), config);
}

async function handleStatsBreakdown(
  config: InternalAxiosRequestConfig,
): Promise<AxiosResponse> {
  return asResponse(await loadJson("/stats-breakdown.json"), config);
}

async function handleTransactionsList(
  config: InternalAxiosRequestConfig,
): Promise<AxiosResponse> {
  const envelope = await loadJson<TransactionList>("/transactions.json");
  const filters: TransactionListFilters = {
    decision: getParam(config, "decision") as TransactionListFilters["decision"],
    country: getParam(config, "country"),
    min_amount: getParam(config, "min_amount"),
    max_amount: getParam(config, "max_amount"),
    start_time: getParam(config, "start_time"),
    end_time: getParam(config, "end_time"),
    customer_id: getParam(config, "customer_id"),
    merchant_id: getParam(config, "merchant_id"),
  };
  const items = filterTransactions(envelope.items, filters);
  // One-page response: the snapshot is bounded at 300 rows and
  // returning a cursor would imply we have more to fetch. The "Load
  // more" footer hides automatically when next_cursor is null.
  return asResponse<TransactionList>(
    { items, next_cursor: null, has_more: false },
    config,
  );
}

async function handleTransactionDetail(
  id: string,
  config: InternalAxiosRequestConfig,
): Promise<AxiosResponse> {
  try {
    const detail = await loadJson(`/transactions/${id}.json`);
    return asResponse(detail, config);
  } catch (err) {
    if (err instanceof DemoNotFoundError) {
      // Surface as a real 404 so existing useTransactionDetail error
      // states (the "not found" branch on the detail page) keep
      // working without a demo-mode special case.
      return asResponse(
        { detail: `Transaction ${id} is not in the public snapshot.` },
        config,
        404,
      );
    }
    throw err;
  }
}

async function handleAlerts(
  config: InternalAxiosRequestConfig,
): Promise<AxiosResponse> {
  const envelope = await loadJson<AlertsResponse>("/alerts.json");
  const filters: AlertsFilters = {
    min_score: getParam(config, "min_score"),
    country: getParam(config, "country"),
    min_age_seconds: paramNumber(config, "min_age_seconds"),
    max_age_seconds: paramNumber(config, "max_age_seconds"),
  };
  const items = filterAlerts(envelope.items, filters);
  return asResponse<AlertsResponse>(
    {
      // Summary is queue-wide by contract — it must not respond to
      // the caller's filters, so we pass the snapshot value through
      // unchanged.
      summary: envelope.summary,
      items,
      next_cursor: null,
      has_more: false,
    },
    config,
  );
}

function resolveHandler(
  method: string,
  url: string,
): RouteHandler | null {
  if (method !== "get") return null;
  if (url === "/stats/overview") return handleStatsOverview;
  if (url === "/stats/timeseries") return handleStatsTimeseries;
  if (url === "/stats/breakdown") return handleStatsBreakdown;
  if (url === "/transactions" || url === "/transactions/")
    return handleTransactionsList;
  if (url === "/alerts" || url === "/alerts/") return handleAlerts;
  const detailMatch = TX_DETAIL_RE.exec(url);
  if (detailMatch) {
    const id = detailMatch[1];
    return (config) => handleTransactionDetail(id, config);
  }
  return null;
}

// ---------------------------------------------------------------------------
// Adapter
// ---------------------------------------------------------------------------

/**
 * Axios adapter compliant with v1's contract. Returns a
 * snapshot-backed response for known routes; rejects writes with a
 * clear message so any UI that slips past the demo-mode guards
 * surfaces an honest error rather than silently 404ing the user.
 */
export const demoAdapter: AxiosAdapter = (config) => {
  const method = (config.method ?? "get").toLowerCase();
  const url = config.url ?? "";

  // Reject writes loudly — the analyst form is supposed to be
  // disabled in demo mode, but a defence-in-depth check here means a
  // missed disable still produces a friendly error instead of a
  // confusing CORS failure.
  if (method !== "get") {
    return Promise.reject(
      Object.assign(
        new Error(
          "Write actions are disabled in the public demo. Run the project locally to record analyst decisions.",
        ),
        { code: "DEMO_WRITE_DISABLED" },
      ),
    );
  }

  const handler = resolveHandler(method, url);
  if (!handler) {
    return Promise.reject(
      new Error(`Demo adapter has no route for ${method.toUpperCase()} ${url}`),
    );
  }

  // Normalise headers to AxiosHeaders so the InternalAxiosRequestConfig
  // contract is satisfied for downstream interceptors.
  const internalConfig: InternalAxiosRequestConfig = {
    ...config,
    headers: AxiosHeaders.from(config.headers),
  };
  return handler(internalConfig);
};
