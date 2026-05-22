"""Transaction ORM model — the core entity of the fraud detection system."""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    String,
    TIMESTAMP,
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
    fraud_score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
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
        Index("ix_transactions_created_at", "created_at"),
        Index("ix_transactions_customer_created", "customer_id", "created_at"),
        Index("ix_transactions_fraud_decision", "fraud_decision"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction id={self.id} amount={self.amount} "
            f"decision={self.fraud_decision}>"
        )
