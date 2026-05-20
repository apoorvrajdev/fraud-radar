"""Repository for Transaction queries — the core entity."""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.models.transaction import Transaction
from app.repositories.base import BaseRepository


class TransactionRepository(BaseRepository[Transaction]):
    """Data access methods for Transaction entities."""

    model = Transaction

    def get_by_idempotency_key(
        self, db: Session, customer_id: str, idempotency_key: str
    ) -> Transaction | None:
        """Look up a transaction by the unique (customer_id, idempotency_key) pair.

        Used to short-circuit duplicate transaction submissions.
        """
        stmt = select(Transaction).where(
            and_(
                Transaction.customer_id == customer_id,
                Transaction.idempotency_key == idempotency_key,
            )
        )
        return db.execute(stmt).scalar_one_or_none()

    def list_recent(
        self,
        db: Session,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Transaction]:
        """Return the most recent transactions, newest first."""
        stmt = (
            select(Transaction)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(db.execute(stmt).scalars().all())

    def list_pending_review(
        self, db: Session, *, limit: int = 50
    ) -> list[Transaction]:
        """Return transactions awaiting analyst review."""
        stmt = (
            select(Transaction)
            .where(Transaction.fraud_decision == "REVIEW")
            .where(Transaction.analyst_label.is_(None))
            .order_by(Transaction.created_at.desc())
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def list_by_customer(
        self,
        db: Session,
        customer_id: str,
        *,
        limit: int = 50,
    ) -> list[Transaction]:
        """List all transactions for a customer, newest first."""
        stmt = (
            select(Transaction)
            .where(Transaction.customer_id == customer_id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
        )
        return list(db.execute(stmt).scalars().all())

    def count_by_customer_since(
        self,
        db: Session,
        customer_id: str,
        since: datetime,
    ) -> int:
        """Count transactions for a customer since a given timestamp.

        Used by velocity-based fraud features.
        """
        stmt = (
            select(Transaction)
            .where(Transaction.customer_id == customer_id)
            .where(Transaction.created_at >= since)
        )
        return len(list(db.execute(stmt).scalars().all()))

    def sum_amount_by_customer_since(
        self,
        db: Session,
        customer_id: str,
        since: datetime,
    ) -> Decimal:
        """Sum of transaction amounts for a customer since a timestamp.

        Used by velocity-based fraud features.
        """
        stmt = (
            select(Transaction.amount)
            .where(Transaction.customer_id == customer_id)
            .where(Transaction.created_at >= since)
        )
        amounts = list(db.execute(stmt).scalars().all())
        return sum(amounts, start=Decimal("0"))


transaction_repository = TransactionRepository()
