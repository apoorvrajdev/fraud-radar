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
