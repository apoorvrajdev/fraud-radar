"""Aggregate-read repository for dashboard endpoints.

Three pure SQLAlchemy aggregates over the `transactions` table:

- `overview_counts`  — decision counts + amount sum + avg fraud score over
  a rolling window ending `now`.
- `pending_review_total` — count of all rows currently sitting at
  `Decision.REVIEW` regardless of age (analyst queue depth).
- `hourly_buckets`   — per-hour transaction count + fraud count over the
  window, used by the timeseries endpoint.
- `country_breakdown` — top-N countries by volume over the window with
  declined counts and total amount.

The hour bucket uses ``func.strftime('%Y-%m-%d %H:00:00', created_at)``
which is SQLite-native. The equivalent Postgres form is
``date_trunc('hour', created_at)`` — swap the expression in
``_hour_bucket_expr`` when the engine moves.

No row sees `Decimal` coerced to ``float`` here; money values stay in
the SQLAlchemy result rows as ``Decimal`` and are returned untouched to
the service layer for quantisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import Integer, case, func, select
from sqlalchemy.orm import Session

from app.fraud.decision import Decision
from app.models.transaction import Transaction

# TODO(postgres): swap to ``date_trunc('hour', Transaction.created_at)``
# when the engine moves off SQLite. The string format below is what
# SQLite returns; service-layer code parses it back into ``datetime``.
_HOUR_BUCKET_FORMAT = "%Y-%m-%d %H:00:00"


def _hour_bucket_expr() -> Any:
    return func.strftime(_HOUR_BUCKET_FORMAT, Transaction.created_at)


# TODO(performance): a covering index on (created_at, fraud_decision)
# would tighten these aggregates if 24h windows ever show up in latency
# profiling. Not adding speculatively for Phase 3E.


@dataclass(frozen=True)
class OverviewRow:
    """Raw aggregate counters for the overview KPI tiles."""

    total: int
    approved: int
    declined: int
    reviewed: int
    fraud_amount: Decimal
    avg_fraud_score: float | None


@dataclass(frozen=True)
class BucketRow:
    """One hour bucket from the timeseries aggregate."""

    bucket: datetime
    transaction_count: int
    fraud_count: int


@dataclass(frozen=True)
class CountryRow:
    """One country row from the breakdown aggregate."""

    country: str
    transaction_count: int
    declined_count: int
    total_amount: Decimal


def overview_counts(
    db: Session,
    *,
    now: datetime,
    window: timedelta = timedelta(hours=24),
) -> OverviewRow:
    """Aggregate decision counts, fraud-blocked amount, and avg score.

    The fraud-blocked amount sums rows whose decision is DECLINE or
    REVIEW within the window — these are the transactions the platform
    stopped or queued for an analyst.
    """
    since = now - window

    decision_col = Transaction.fraud_decision
    amount_col = Transaction.amount
    score_col = Transaction.fraud_score

    fraud_amount_expr = func.coalesce(
        func.sum(
            case(
                (
                    decision_col.in_(
                        [Decision.DECLINE.value, Decision.REVIEW.value]
                    ),
                    amount_col,
                ),
                else_=Decimal("0"),
            )
        ),
        Decimal("0"),
    )

    stmt = select(
        func.count().label("total"),
        func.coalesce(
            func.sum(
                case(
                    (decision_col == Decision.APPROVE.value, 1),
                    else_=0,
                )
            ),
            0,
        ).label("approved"),
        func.coalesce(
            func.sum(
                case(
                    (decision_col == Decision.DECLINE.value, 1),
                    else_=0,
                )
            ),
            0,
        ).label("declined"),
        func.coalesce(
            func.sum(
                case(
                    (decision_col == Decision.REVIEW.value, 1),
                    else_=0,
                )
            ),
            0,
        ).label("reviewed"),
        fraud_amount_expr.label("fraud_amount"),
        func.avg(score_col).label("avg_score"),
    ).where(Transaction.created_at >= since)

    row = db.execute(stmt).one()

    total = int(row.total or 0)
    avg_raw = row.avg_score
    avg_fraud_score = float(avg_raw) if avg_raw is not None else None
    fraud_amount = (
        row.fraud_amount
        if isinstance(row.fraud_amount, Decimal)
        else Decimal(str(row.fraud_amount))
    )

    return OverviewRow(
        total=total,
        approved=int(row.approved or 0),
        declined=int(row.declined or 0),
        reviewed=int(row.reviewed or 0),
        fraud_amount=fraud_amount,
        avg_fraud_score=avg_fraud_score,
    )


def pending_review_total(db: Session) -> int:
    """Count of transactions currently awaiting analyst review.

    Not windowed — represents queue depth, which grows independently of
    the 24h KPI window.
    """
    stmt = select(func.count()).where(
        Transaction.fraud_decision == Decision.REVIEW.value
    )
    return int(db.execute(stmt).scalar_one() or 0)


def hourly_buckets(
    db: Session,
    *,
    now: datetime,
    window: timedelta = timedelta(hours=24),
) -> list[BucketRow]:
    """Per-hour aggregate over the window.

    Returned only for hours that actually have rows in the DB. Empty
    hours are filled in by the service layer so the chart's x-axis stays
    continuous without making SQLite generate a calendar table.
    """
    since = now - window
    bucket_expr = _hour_bucket_expr().label("bucket")

    is_fraud = case(
        (
            Transaction.fraud_decision.in_(
                [Decision.DECLINE.value, Decision.REVIEW.value]
            ),
            1,
        ),
        else_=0,
    )

    stmt = (
        select(
            bucket_expr,
            func.count().label("tx_count"),
            func.coalesce(func.sum(is_fraud), 0).cast(Integer).label("fraud_count"),
        )
        .where(Transaction.created_at >= since)
        .group_by(bucket_expr)
        .order_by(bucket_expr.asc())
    )

    rows = db.execute(stmt).all()
    out: list[BucketRow] = []
    for row in rows:
        bucket = datetime.strptime(str(row.bucket), _HOUR_BUCKET_FORMAT)
        out.append(
            BucketRow(
                bucket=bucket,
                transaction_count=int(row.tx_count or 0),
                fraud_count=int(row.fraud_count or 0),
            )
        )
    return out


def country_breakdown(
    db: Session,
    *,
    now: datetime,
    window: timedelta = timedelta(hours=24),
    limit: int = 10,
) -> list[CountryRow]:
    """Top-N countries by transaction count within the window."""
    since = now - window

    declined_expr = func.coalesce(
        func.sum(
            case(
                (Transaction.fraud_decision == Decision.DECLINE.value, 1),
                else_=0,
            )
        ),
        0,
    )

    stmt = (
        select(
            Transaction.country.label("country"),
            func.count().label("tx_count"),
            declined_expr.label("declined"),
            func.coalesce(func.sum(Transaction.amount), Decimal("0")).label(
                "total_amount"
            ),
        )
        .where(Transaction.created_at >= since)
        .group_by(Transaction.country)
        .order_by(func.count().desc())
        .limit(limit)
    )

    rows = db.execute(stmt).all()
    out: list[CountryRow] = []
    for row in rows:
        total_amount = (
            row.total_amount
            if isinstance(row.total_amount, Decimal)
            else Decimal(str(row.total_amount))
        )
        out.append(
            CountryRow(
                country=str(row.country),
                transaction_count=int(row.tx_count or 0),
                declined_count=int(row.declined or 0),
                total_amount=total_amount,
            )
        )
    return out
