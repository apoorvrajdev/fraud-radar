"""IdempotencyKey ORM model — Stripe-style request cache.

See `docs/adr/PHASE_3_DESIGN.md` "Idempotency design" for the full
semantics (hash-based replay detection, 24-hour TTL, 409 on payload
mismatch).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    TIMESTAMP,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class IdempotencyKey(Base):
    """One row per accepted POST /transactions request.

    The cache is global by `key`. Callers supply the key in the
    `Idempotency-Key` header; identical key + identical payload returns
    the original response, identical key + different payload returns 409.
    """

    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    transaction_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("transactions.id", ondelete="CASCADE"),
        nullable=False,
    )
    response_body: Mapped[str] = mapped_column(Text, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False, default=201)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "expires_at > created_at",
            name="ck_idempotency_keys_expires_after_created",
        ),
        Index("ix_idempotency_keys_request_hash", "request_hash"),
        Index("ix_idempotency_keys_expires_at", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<IdempotencyKey key={self.key} tx={self.transaction_id}>"
