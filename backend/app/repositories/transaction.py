"""Repository for Transaction queries — the core entity."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.fraud.decision import Decision
from app.models.transaction import Transaction
from app.repositories.base import BaseRepository


@dataclass(frozen=True)
class TransactionListFilters:
    """Filter set for the paginated transactions list endpoint (Phase 3F).

    Every field is optional; an empty instance returns the most recent
    rows unfiltered.
    """

    decision: Decision | None = None
    country: str | None = None
    min_amount: Decimal | None = None
    max_amount: Decimal | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    customer_id: str | None = None
    merchant_id: str | None = None


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

    def list_paginated(
        self,
        db: Session,
        *,
        filters: TransactionListFilters,
        limit: int = 50,
        cursor: tuple[datetime, str] | None = None,
    ) -> tuple[list[Transaction], tuple[datetime, str] | None]:
        """Keyset-paginated list, newest first.

        Returns ``(rows, next_cursor_or_None)``. The cursor is the
        ``(created_at, id)`` of the last row in the returned page;
        ``None`` means the caller has reached the end. Fetches
        ``limit + 1`` rows internally to detect end-of-stream without a
        second round-trip.
        """
        stmt = select(Transaction)

        if filters.decision is not None:
            stmt = stmt.where(Transaction.fraud_decision == filters.decision.value)
        if filters.country is not None:
            stmt = stmt.where(Transaction.country == filters.country)
        if filters.min_amount is not None:
            stmt = stmt.where(Transaction.amount >= filters.min_amount)
        if filters.max_amount is not None:
            stmt = stmt.where(Transaction.amount <= filters.max_amount)
        if filters.start_time is not None:
            stmt = stmt.where(Transaction.created_at >= filters.start_time)
        if filters.end_time is not None:
            stmt = stmt.where(Transaction.created_at < filters.end_time)
        if filters.customer_id is not None:
            stmt = stmt.where(Transaction.customer_id == filters.customer_id)
        if filters.merchant_id is not None:
            stmt = stmt.where(Transaction.merchant_id == filters.merchant_id)

        if cursor is not None:
            cursor_ts, cursor_id = cursor
            # Tuple comparison: rows strictly "after" the cursor in the
            # (created_at DESC, id DESC) ordering. Equivalent to
            #   created_at < ts OR (created_at = ts AND id < id)
            # but lets the planner use a composite index when present.
            stmt = stmt.where(
                or_(
                    Transaction.created_at < cursor_ts,
                    and_(
                        Transaction.created_at == cursor_ts,
                        Transaction.id < cursor_id,
                    ),
                )
            )

        stmt = stmt.order_by(
            Transaction.created_at.desc(),
            Transaction.id.desc(),
        ).limit(limit + 1)

        rows = list(db.execute(stmt).scalars().all())
        if len(rows) > limit:
            page = rows[:limit]
            last = page[-1]
            return page, (last.created_at, last.id)
        return rows, None

    # ---------------------------------------------------------------
    # Phase 3G — analyst override
    # ---------------------------------------------------------------

    def apply_analyst_decision(
        self,
        db: Session,
        *,
        tx: Transaction,
        label: str,
        notes: str | None,
        now: datetime,
    ) -> Transaction:
        """Stamp `analyst_label`, `analyst_notes`, `reviewed_at` on a row.

        Flushes inside the caller's session but does **not** commit —
        the service layer owns the transaction boundary so the column
        update and the matching audit-log insert land atomically.

        `fraud_decision` is intentionally unchanged: the model's
        verdict is preserved verbatim for evaluation and retraining,
        which is the whole reason `analyst_label` exists as a
        separate column.
        """
        tx.analyst_label = label
        tx.analyst_notes = notes
        tx.reviewed_at = now
        db.add(tx)
        db.flush()
        return tx


transaction_repository = TransactionRepository()
