"""Repository for Transaction queries — the core entity."""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import and_, case, func, or_, select
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

    # ---------------------------------------------------------------
    # Phase 3H — analyst alerts queue
    # ---------------------------------------------------------------

    def list_alerts_paginated(
        self,
        db: Session,
        *,
        filters: "AlertsListFilters",
        limit: int,
        cursor: tuple[Decimal, datetime, str] | None,
        now: datetime,
    ) -> tuple[
        list[Transaction], tuple[Decimal, datetime, str] | None
    ]:
        """Keyset-paginated pending-review queue.

        Queue predicate: ``fraud_decision = 'REVIEW' AND
        analyst_label IS NULL``. Sort key:
        ``(fraud_score DESC, created_at ASC, id ASC)`` — riskiest
        first, oldest as tiebreaker so old rows do not get stranded
        behind a flood of fresh high-score arrivals.

        Returns ``(rows, next_cursor_or_None)``. The cursor is the
        ``(fraud_score, created_at, id)`` of the last row in the
        returned page; ``None`` means end-of-stream. Fetches
        ``limit + 1`` rows internally so the caller learns whether
        more pages exist without a second round-trip.
        """
        stmt = (
            select(Transaction)
            .where(Transaction.fraud_decision == "REVIEW")
            .where(Transaction.analyst_label.is_(None))
        )

        if filters.min_score is not None:
            stmt = stmt.where(Transaction.fraud_score >= filters.min_score)
        if filters.country is not None:
            stmt = stmt.where(Transaction.country == filters.country)
        if filters.max_created_at is not None:
            # min_age_seconds → row must be older than (now - min_age)
            stmt = stmt.where(Transaction.created_at <= filters.max_created_at)
        if filters.min_created_at is not None:
            # max_age_seconds → row must be newer than (now - max_age)
            stmt = stmt.where(Transaction.created_at >= filters.min_created_at)

        if cursor is not None:
            c_score, c_ts, c_id = cursor
            # Rows strictly "after" the cursor in
            #   (fraud_score DESC, created_at ASC, id ASC)
            # which expands to:
            #   fraud_score < c_score
            #   OR (fraud_score = c_score AND created_at > c_ts)
            #   OR (fraud_score = c_score AND created_at = c_ts AND id > c_id)
            stmt = stmt.where(
                or_(
                    Transaction.fraud_score < c_score,
                    and_(
                        Transaction.fraud_score == c_score,
                        Transaction.created_at > c_ts,
                    ),
                    and_(
                        Transaction.fraud_score == c_score,
                        Transaction.created_at == c_ts,
                        Transaction.id > c_id,
                    ),
                )
            )

        stmt = stmt.order_by(
            Transaction.fraud_score.desc(),
            Transaction.created_at.asc(),
            Transaction.id.asc(),
        ).limit(limit + 1)

        rows = list(db.execute(stmt).scalars().all())
        if len(rows) > limit:
            page = rows[:limit]
            last = page[-1]
            assert last.fraud_score is not None  # guaranteed by predicate
            return page, (last.fraud_score, last.created_at, last.id)
        return rows, None

    def pending_review_summary(
        self, db: Session
    ) -> tuple[int, datetime | None, dict[str, int]]:
        """Single aggregate over the pending-review predicate.

        Returns ``(pending_count, oldest_created_at, score_buckets)``.
        Bucket boundaries are fixed (see PHASE_3H_DESIGN.md):
        ``low`` (< 0.20), ``mid`` (0.20 <= x < 0.50),
        ``high`` (>= 0.50). Counts sum exactly to ``pending_count``.
        """
        bucket = case(
            (Transaction.fraud_score < Decimal("0.20"), "low"),
            (Transaction.fraud_score < Decimal("0.50"), "mid"),
            else_="high",
        ).label("bucket")

        stmt = (
            select(
                func.count().label("total"),
                func.min(Transaction.created_at).label("oldest"),
                bucket,
                func.count().label("bucket_count"),
            )
            .where(Transaction.fraud_decision == "REVIEW")
            .where(Transaction.analyst_label.is_(None))
            .group_by(bucket)
        )

        # Per-bucket rows. Each row carries the same `total` and `oldest`
        # (they are window-like aggregates here only because SQLite is
        # lenient; we read them off the first row deliberately).
        buckets: dict[str, int] = {"low": 0, "mid": 0, "high": 0}
        total = 0
        oldest: datetime | None = None
        for row in db.execute(stmt).all():
            # `row.total` here is the count *per bucket* under strict
            # SQL, so we sum buckets to get the true total. SQLite's
            # behaviour around bare aggregates with GROUP BY is loose,
            # so we never trust `row.total` and instead recompute below.
            buckets[str(row.bucket)] = int(row.bucket_count)

        total = sum(buckets.values())

        if total > 0:
            oldest_stmt = (
                select(func.min(Transaction.created_at))
                .where(Transaction.fraud_decision == "REVIEW")
                .where(Transaction.analyst_label.is_(None))
            )
            oldest = db.execute(oldest_stmt).scalar_one_or_none()

        return total, oldest, buckets


@dataclass(frozen=True)
class AlertsListFilters:
    """Filter set for the alerts-queue keyset query.

    The router maps the public ``min_age_seconds`` / ``max_age_seconds``
    parameters into absolute ``min_created_at`` / ``max_created_at``
    timestamps at the service boundary, so the repository never has to
    know what "now" is.
    """

    min_score: Decimal | None = None
    country: str | None = None
    min_created_at: datetime | None = None
    max_created_at: datetime | None = None


transaction_repository = TransactionRepository()
