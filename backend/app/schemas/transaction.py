"""Pydantic schemas for transaction ingestion and retrieval."""
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

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


class AnalystDecisionRequest(BaseModel):
    """Request body when an analyst confirms or rejects a flagged transaction."""

    label: str = Field(
        description="Must be CONFIRMED_FRAUD or CONFIRMED_LEGIT",
        pattern=r"^(CONFIRMED_FRAUD|CONFIRMED_LEGIT)$",
    )
    notes: str | None = Field(default=None, max_length=2000)
