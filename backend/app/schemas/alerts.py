"""Pydantic schemas for the analyst alerts queue (Phase 3H).

See `docs/adr/PHASE_3H_DESIGN.md` for the predicate, sort order,
cursor format, and score-bucket boundaries.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AlertsQuery(BaseModel):
    """Query parameters for GET /api/v1/alerts.

    The cross-field constraint (`min_age_seconds <= max_age_seconds`)
    is enforced here so the router stays a one-liner. Malformed
    combinations raise 422 with a Pydantic error body.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    limit: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None
    min_score: Decimal | None = Field(
        default=None, ge=Decimal("0"), le=Decimal("1")
    )
    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    min_age_seconds: int | None = Field(default=None, ge=0)
    max_age_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_age_range(self) -> Self:
        if (
            self.min_age_seconds is not None
            and self.max_age_seconds is not None
            and self.min_age_seconds > self.max_age_seconds
        ):
            raise ValueError("min_age_seconds must be <= max_age_seconds")
        return self


class AlertItem(BaseModel):
    """One pending-review row as the queue page sees it."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    age_seconds: int
    amount: Decimal
    currency: str
    country: str
    customer_id: str
    merchant_id: str
    fraud_score: Decimal
    fraud_decision: Literal["REVIEW"]
    rules_triggered: list[str] = Field(default_factory=list)


class AlertsSummary(BaseModel):
    """Queue-wide health stats, ignoring the caller's filters.

    `score_buckets` keys are fixed: ``low`` (score < 0.20),
    ``mid`` (0.20 <= score < 0.50), ``high`` (score >= 0.50).
    Bucket counts sum exactly to ``pending_count`` — no overflow
    bucket needed because the auto-decline threshold (~0.74) keeps
    REVIEW rows comfortably below 1.0 in practice.
    """

    pending_count: int = Field(ge=0)
    oldest_pending_seconds: int | None = Field(default=None, ge=0)
    score_buckets: dict[str, int]


class AlertsResponse(BaseModel):
    """Envelope for GET /api/v1/alerts."""

    summary: AlertsSummary
    items: list[AlertItem]
    next_cursor: str | None = None
    has_more: bool = False
