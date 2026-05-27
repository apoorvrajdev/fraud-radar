"""Pydantic schemas for transaction ingestion and retrieval."""
from datetime import datetime
from decimal import Decimal
from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.fraud.decision import Decision
from app.schemas.common import PaymentMethod, TransactionStatus
from app.schemas.fraud import FraudScoreResult


class TransactionCreate(BaseModel):
    """Request body for creating a new transaction.

    The idempotency_key is supplied via the Idempotency-Key HTTP header,
    not in this body — same pattern as the Stripe API.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(min_length=36, max_length=36)
    merchant_id: str = Field(min_length=36, max_length=36)
    amount: Decimal = Field(
        gt=Decimal("0"),
        decimal_places=4,
        description="Transaction amount in the given currency",
    )
    currency: str = Field(
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code, e.g. 'USD'",
    )
    payment_method: PaymentMethod
    card_last4: str | None = Field(
        default=None, min_length=4, max_length=4, pattern=r"^\d{4}$"
    )
    ip_address: str | None = Field(default=None, max_length=45)
    device_id: str | None = Field(default=None, max_length=128)
    country: str = Field(
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    is_card_present: bool = False


class TransactionResponse(BaseModel):
    """Response returned after creating or fetching a transaction."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    customer_id: str
    merchant_id: str
    amount: Decimal
    currency: str
    status: TransactionStatus
    payment_method: PaymentMethod
    country: str
    fraud_score: Decimal | None = None
    fraud_decision: str | None = None
    created_at: datetime


class TransactionDetail(TransactionResponse):
    """Detailed transaction view including fraud explanation and analyst feedback."""

    model_config = ConfigDict(from_attributes=True)

    card_last4: str | None = None
    ip_address: str | None = None
    device_id: str | None = None
    is_card_present: bool
    fraud: FraudScoreResult | None = None
    analyst_label: str | None = None
    analyst_notes: str | None = None
    reviewed_at: datetime | None = None


class TransactionScored(BaseModel):
    """Response payload for POST /transactions and GET /transactions/{id}.

    In Phase 3B `fraud_score`, `threshold`, `rules_triggered`, and
    `top_contributors` are placeholders (None / empty) because real
    scoring lands in Phase 3C. The schema shape stays stable; 3C just
    fills the fields in.
    """

    model_config = ConfigDict(from_attributes=True)

    transaction_id: UUID
    fraud_score: float | None
    decision: Decision
    threshold: float | None
    rules_triggered: list[str] = Field(default_factory=list)
    top_contributors: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: datetime


class AnalystDecisionRequest(BaseModel):
    """Request body when an analyst confirms or rejects a flagged transaction."""

    label: str = Field(
        description="Must be CONFIRMED_FRAUD or CONFIRMED_LEGIT",
        pattern=r"^(CONFIRMED_FRAUD|CONFIRMED_LEGIT)$",
    )
    notes: str | None = Field(default=None, max_length=2000)


# ---------------------------------------------------------------------------
# Phase 3F: paginated list endpoint
# ---------------------------------------------------------------------------


class TransactionListQuery(BaseModel):
    """Query parameters for GET /api/v1/transactions.

    Cross-field constraints (`min_amount <= max_amount`,
    `start_time < end_time`) are enforced here so the router stays a
    one-liner. Bad combinations raise 422 with a Pydantic error body.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    limit: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None
    decision: Decision | None = None
    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    min_amount: Decimal | None = Field(default=None, ge=Decimal("0"))
    max_amount: Decimal | None = Field(default=None, ge=Decimal("0"))
    start_time: datetime | None = None
    end_time: datetime | None = None
    customer_id: str | None = Field(default=None, min_length=36, max_length=36)
    merchant_id: str | None = Field(default=None, min_length=36, max_length=36)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount > self.max_amount
        ):
            raise ValueError("min_amount must be <= max_amount")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.start_time >= self.end_time
        ):
            raise ValueError("start_time must be < end_time")
        return self


class TransactionList(BaseModel):
    """Paginated transactions response. Sort is fixed to created_at DESC."""

    model_config = ConfigDict(from_attributes=True)

    items: list[TransactionResponse]
    next_cursor: str | None = None
    has_more: bool = False
