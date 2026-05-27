/**
 * Hand-mirrored types for the backend's `/stats/*` Pydantic schemas.
 *
 * No codegen for Phase 3E (see ADR "Type sharing"). When the backend
 * schema changes, update both sides in the same commit.
 *
 * Money values cross the wire as strings (Pydantic v2 default for
 * `Decimal`) — keep them as `string` here and format with
 * `Intl.NumberFormat` at the component boundary. Never parse to
 * `number`, even for display, to preserve precision.
 */

export interface StatsOverview {
  total_transactions_24h: number;
  approved_count_24h: number;
  declined_count_24h: number;
  pending_review_count: number;
  approved_rate: number;
  fraud_caught_amount: string;
  avg_fraud_score: number | null;
}

export interface TimeseriesPoint {
  timestamp: string;
  transaction_count: number;
  fraud_rate: number;
}

export interface StatsTimeseries {
  window: "24h";
  points: TimeseriesPoint[];
}

export interface CategoryBreakdown {
  category: string;
  transaction_count: number;
  declined_count: number;
  total_amount: string;
}

export interface StatsBreakdown {
  dimension: "country";
  items: CategoryBreakdown[];
}

// ---------------------------------------------------------------------------
// Phase 3F: transactions list
// ---------------------------------------------------------------------------

export type Decision = "APPROVE" | "REVIEW" | "DECLINE" | "PENDING";

/**
 * Single row in the paginated transactions list. Mirrors the backend's
 * `TransactionResponse` Pydantic schema. `amount` and `fraud_score`
 * cross the wire as strings (Decimal) — keep them as strings here.
 */
export interface TransactionListItem {
  id: string;
  customer_id: string;
  merchant_id: string;
  amount: string;
  currency: string;
  status: string;
  payment_method: string;
  country: string;
  fraud_score: string | null;
  fraud_decision: Decision | null;
  created_at: string;
}

export interface TransactionList {
  items: TransactionListItem[];
  next_cursor: string | null;
  has_more: boolean;
}

/**
 * Wire-format query parameters for GET /transactions. Every field is
 * optional. `min_amount` / `max_amount` are strings because they
 * originate as `Decimal` on the backend; we pass them through as text
 * so precision is preserved.
 */
export interface TransactionListFilters {
  decision?: Decision;
  country?: string;
  min_amount?: string;
  max_amount?: string;
  start_time?: string;
  end_time?: string;
  customer_id?: string;
  merchant_id?: string;
}

// ---------------------------------------------------------------------------
// Phase 3G: transaction detail envelope
// ---------------------------------------------------------------------------

export type AnalystLabel = "CONFIRMED_FRAUD" | "CONFIRMED_LEGIT";

/**
 * One SHAP contributor surfaced on the detail page. Mirrors the
 * backend `ContributorEntry` schema. `direction` is pre-classified by
 * the explainer at scoring time (shap_value > 0 → "fraud", otherwise
 * "legit") so the frontend never has to interpret the sign.
 */
export interface ContributorEntry {
  feature: string;
  feature_value: number;
  shap_value: number;
  direction: "fraud" | "legit";
}

/**
 * One audit-log entry. The backend ships the trailing 20 newest-first.
 * `payload` is opaque structured JSON; rendering is per-action.
 */
export interface AuditEntry {
  id: number;
  actor: string;
  action: string;
  payload: Record<string, unknown> | null;
  created_at: string;
}

/**
 * Composite detail envelope for GET /transactions/{id}. The original
 * `fraud_decision` (model verdict) is preserved verbatim;
 * `effective_decision` reflects any analyst override.
 */
export interface TransactionDetail {
  id: string;
  customer_id: string;
  merchant_id: string;
  amount: string;
  currency: string;
  status: string;
  payment_method: string;
  country: string;
  card_last4: string | null;
  ip_address: string | null;
  device_id: string | null;
  is_card_present: boolean;
  fraud_score: string | null;
  fraud_decision: Decision | null;
  threshold: number | null;
  rules_triggered: string[];
  top_contributors: ContributorEntry[];
  effective_decision: Decision;
  analyst_label: AnalystLabel | null;
  analyst_notes: string | null;
  reviewed_at: string | null;
  created_at: string;
  audit: AuditEntry[];
}

// ---------------------------------------------------------------------------
// Phase 3H: analyst alerts queue
// ---------------------------------------------------------------------------

/**
 * One row in the analyst alerts queue. The backend guarantees
 * `fraud_decision === "REVIEW"` and `analyst_label IS NULL` for every
 * item in this envelope. `age_seconds` is computed server-side at
 * response time against the queue's `now` so the value is stable for
 * the duration of a single fetch.
 */
export interface AlertItem {
  id: string;
  created_at: string;
  age_seconds: number;
  amount: string;
  currency: string;
  country: string;
  customer_id: string;
  merchant_id: string;
  fraud_score: string;
  fraud_decision: "REVIEW";
  rules_triggered: string[];
}

/**
 * Queue-wide health summary. The buckets are fixed:
 * `low` (< 0.20), `mid` (0.20–0.50), `high` (>= 0.50). Counts sum
 * exactly to `pending_count`. The summary block ignores the
 * caller's filters by design — it reports the whole queue, not the
 * visible page.
 */
export interface AlertsSummary {
  pending_count: number;
  oldest_pending_seconds: number | null;
  score_buckets: {
    low: number;
    mid: number;
    high: number;
  };
}

export interface AlertsResponse {
  summary: AlertsSummary;
  items: AlertItem[];
  next_cursor: string | null;
  has_more: boolean;
}

/**
 * Wire-format query parameters for GET /alerts. The age window
 * params are integer seconds; the score floor is a string (Decimal)
 * so backend precision is preserved.
 */
export interface AlertsFilters {
  min_score?: string;
  country?: string;
  min_age_seconds?: number;
  max_age_seconds?: number;
}
