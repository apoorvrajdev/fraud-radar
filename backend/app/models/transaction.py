"""Transaction ORM model — the core entity of the fraud detection system."""
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class Transaction(Base, TimestampMixin):
    """A payment transaction.

    Fraud scoring results are denormalized onto this row for fast reads.
    The analyst_label field captures human-in-the-loop feedback for model
    retraining.
    """

    __tablename__ = "transactions"

    # Identity
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)

    # Foreign keys
    customer_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("customers.id"), nullable=False
    )
    merchant_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("merchants.id"), nullable=False
    )

    # Money — NEVER use Float for currency
    amount: Mapped[Decimal] = mapped_column(Numeric(19, 4), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="USD"
    )

    # FX enrichment (Phase 5F) — DERIVED reporting fields, all nullable.
    # `amount` and `currency` above stay authoritative and are never
    # written by the enrichment path: these four are added beside them,
    # or left null when no rate could be resolved. `amount_base` is
    # denominated in the reporting currency (FX_BASE_CURRENCY), and
    # `fx_rate` means 1 unit of `currency` = `fx_rate` units of it.
    # See docs/FX_CONTRACT.md.
    amount_base: Mapped[Decimal | None] = mapped_column(
        Numeric(19, 4), nullable=True
    )
    fx_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 8), nullable=True
    )
    fx_rate_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    fx_source: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Transaction metadata
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(32), nullable=False)
    card_last4: Mapped[str | None] = mapped_column(String(4), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    device_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    is_card_present: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Fraud scoring results (denormalized for read performance)
    # Numeric(5, 4) hands back a Decimal at runtime — annotate it as such so
    # the readers of this column type-check against what they actually get.
    fraud_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    fraud_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rules_triggered: Mapped[str | None] = mapped_column(Text, nullable=True)
    top_features: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Human-in-the-loop analyst feedback
    analyst_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    analyst_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "customer_id",
            "idempotency_key",
            name="uq_transactions_idempotency",
        ),
        CheckConstraint(
            "status IN ('APPROVED', 'DECLINED', 'PENDING_REVIEW')",
            name="ck_transactions_status",
        ),
        CheckConstraint(
            "fraud_decision IS NULL OR fraud_decision IN "
            "('APPROVE', 'REVIEW', 'DECLINE', 'PENDING')",
            name="ck_transactions_fraud_decision",
        ),
        CheckConstraint(
            "analyst_label IS NULL OR analyst_label IN "
            "('CONFIRMED_FRAUD', 'CONFIRMED_LEGIT')",
            name="ck_transactions_analyst_label",
        ),
        CheckConstraint("amount > 0", name="ck_transactions_amount_positive"),
        CheckConstraint(
            "fx_source IS NULL OR fx_source IN "
            "('identity', 'live', 'cache', 'stale', 'unsupported', "
            "'unavailable')",
            name="ck_transactions_fx_source",
        ),
        CheckConstraint(
            "fx_rate IS NULL OR fx_rate > 0",
            name="ck_transactions_fx_rate_positive",
        ),
        # A half-converted row must not exist: either we have both the
        # derived amount and the rate that produced it, or neither.
        CheckConstraint(
            "(amount_base IS NULL AND fx_rate IS NULL) OR "
            "(amount_base IS NOT NULL AND fx_rate IS NOT NULL)",
            name="ck_transactions_fx_amount_pairing",
        ),
        Index("ix_transactions_created_at", "created_at"),
        Index("ix_transactions_customer_created", "customer_id", "created_at"),
        Index("ix_transactions_fraud_decision", "fraud_decision"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction id={self.id} amount={self.amount} "
            f"decision={self.fraud_decision}>"
        )
