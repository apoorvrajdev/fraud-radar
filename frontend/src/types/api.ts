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
