"""Service layer for the dashboard stats endpoints.

Converts repository rows into the Pydantic response schemas declared in
``app.schemas.stats``. Three responsibilities live here that the
repository deliberately does not own:

1. Empty-window handling — an empty 24h window returns
   ``avg_fraud_score=None`` and ``approved_rate=0.0``, both of which are
   business decisions, not SQL details.
2. ``Decimal`` quantisation — every money value is rounded to 2 dp
   (``ROUND_HALF_UP``) before crossing the API boundary.
3. Continuous timeseries — empty hour buckets are filled with
   ``transaction_count=0`` and ``fraud_rate=0.0`` so the chart never
   shows gaps.

Routers call these three functions and return the result. No business
logic lives in the router layer.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy.orm import Session

from app.repositories import stats as stats_repo
from app.schemas.stats import (
    CategoryBreakdown,
    StatsBreakdown,
    StatsOverview,
    StatsTimeseries,
    TimeseriesPoint,
)

_WINDOW_24H = timedelta(hours=24)
_TWO_DP = Decimal("0.01")
_HOUR = timedelta(hours=1)


def _quantise(amount: Decimal) -> Decimal:
    return amount.quantize(_TWO_DP, rounding=ROUND_HALF_UP)


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def get_overview(db: Session, *, now: datetime) -> StatsOverview:
    """Top-line KPIs over the trailing 24-hour window ending ``now``."""
    row = stats_repo.overview_counts(db, now=now, window=_WINDOW_24H)
    pending_review = stats_repo.pending_review_total(db)

    approved_rate = (row.approved / row.total) if row.total > 0 else 0.0

    return StatsOverview(
        total_transactions_24h=row.total,
        approved_count_24h=row.approved,
        declined_count_24h=row.declined,
        pending_review_count=pending_review,
        approved_rate=approved_rate,
        fraud_caught_amount=_quantise(row.fraud_amount),
        avg_fraud_score=row.avg_fraud_score,
    )


def get_timeseries(db: Session, *, now: datetime) -> StatsTimeseries:
    """24 hourly buckets ending at the top of ``now``'s hour.

    Buckets without any rows are emitted with zero counts so the chart's
    x-axis stays continuous.
    """
    rows = stats_repo.hourly_buckets(db, now=now, window=_WINDOW_24H)
    by_bucket = {
        # Repository returns naive datetimes parsed from SQLite's TEXT;
        # the stored timestamps are UTC by convention.
        row.bucket.replace(tzinfo=UTC): row
        for row in rows
    }

    end_hour = _floor_to_hour(now.astimezone(UTC))
    start_hour = end_hour - _WINDOW_24H + _HOUR

    points: list[TimeseriesPoint] = []
    cursor = start_hour
    while cursor <= end_hour:
        bucket = by_bucket.get(cursor)
        if bucket is None:
            tx_count = 0
            fraud_rate = 0.0
        else:
            tx_count = bucket.transaction_count
            fraud_rate = (
                bucket.fraud_count / bucket.transaction_count
                if bucket.transaction_count > 0
                else 0.0
            )
        points.append(
            TimeseriesPoint(
                timestamp=cursor,
                transaction_count=tx_count,
                fraud_rate=fraud_rate,
            )
        )
        cursor += _HOUR

    return StatsTimeseries(window="24h", points=points)


def get_breakdown(db: Session, *, now: datetime) -> StatsBreakdown:
    """Top-10 countries by transaction count over the trailing 24h."""
    rows = stats_repo.country_breakdown(
        db, now=now, window=_WINDOW_24H, limit=10
    )
    items = [
        CategoryBreakdown(
            category=row.country,
            transaction_count=row.transaction_count,
            declined_count=row.declined_count,
            total_amount=_quantise(row.total_amount),
        )
        for row in rows
    ]
    return StatsBreakdown(dimension="country", items=items)
