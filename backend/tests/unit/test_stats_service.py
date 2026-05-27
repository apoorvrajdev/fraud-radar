"""Unit tests for the dashboard stats service.

The service layer owns three responsibilities:

1. Empty-window handling — empty 24h window returns
   ``avg_fraud_score=None`` and ``approved_rate=0.0``.
2. Decimal quantisation to 2 dp.
3. Continuous timeseries — 24 hourly buckets, empty hours filled with
   zero counts and zero fraud rate.

The repository is exercised against a real in-memory SQLite engine
seeded with deterministic fixtures so the SQL ``case``/``strftime``
expressions are real-tested, not mocked.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.fraud.decision import Decision
from app.models import Customer, Merchant
from app.models.base import Base
from app.models.transaction import Transaction
from app.services import stats as stats_service

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"
# Anchor "now" used by every test in this module — chosen on the hour so
# the timeseries bucket boundaries are easy to reason about.
NOW = datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionTesting = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False,
    )
    db = SessionTesting()
    db.add(
        Customer(
            id=CUSTOMER_ID,
            email="seed@example.com",
            full_name="Seed Customer",
            country="US",
            risk_tier="LOW",
            account_age_days=365,
        )
    )
    db.add(
        Merchant(
            id=MERCHANT_ID,
            name="Seed Merchant",
            category="RETAIL",
            mcc="5311",
            country="US",
            risk_rating="LOW",
        )
    )
    db.commit()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def _make_tx(
    *,
    tx_id: str,
    created_at: datetime,
    decision: Decision,
    amount: str = "100.00",
    country: str = "US",
    score: float = 0.05,
    **overrides: Any,
) -> Transaction:
    defaults: dict[str, Any] = {
        "id": tx_id,
        "idempotency_key": f"key-{tx_id}",
        "customer_id": CUSTOMER_ID,
        "merchant_id": MERCHANT_ID,
        "amount": Decimal(amount),
        "currency": "USD",
        "status": "APPROVED",
        "payment_method": "CARD",
        "country": country,
        "is_card_present": True,
        "fraud_score": score,
        "fraud_decision": decision.value,
        "created_at": created_at,
    }
    defaults.update(overrides)
    return Transaction(**defaults)


# ---------------------------------------------------------------------------
# get_overview
# ---------------------------------------------------------------------------


def test_overview_empty_window_returns_zeros_and_none_avg_score(
    db_session: Session,
) -> None:
    out = stats_service.get_overview(db_session, now=NOW)

    assert out.total_transactions_24h == 0
    assert out.approved_count_24h == 0
    assert out.declined_count_24h == 0
    assert out.pending_review_count == 0
    assert out.approved_rate == 0.0
    assert out.fraud_caught_amount == Decimal("0.00")
    assert out.avg_fraud_score is None


def test_overview_counts_only_within_window(db_session: Session) -> None:
    # Inside window: 1 approve, 1 decline, 1 review.
    db_session.add(
        _make_tx(
            tx_id="a",
            created_at=NOW - timedelta(hours=1),
            decision=Decision.APPROVE,
            amount="100.00",
            score=0.05,
        )
    )
    db_session.add(
        _make_tx(
            tx_id="b",
            created_at=NOW - timedelta(hours=2),
            decision=Decision.DECLINE,
            amount="500.00",
            score=0.95,
        )
    )
    db_session.add(
        _make_tx(
            tx_id="c",
            created_at=NOW - timedelta(hours=3),
            decision=Decision.REVIEW,
            amount="250.00",
            score=0.55,
        )
    )
    # Outside window — should be ignored by overview but counted by
    # pending_review (it's a REVIEW row).
    db_session.add(
        _make_tx(
            tx_id="d",
            created_at=NOW - timedelta(hours=30),
            decision=Decision.REVIEW,
            amount="999.00",
            score=0.50,
        )
    )
    db_session.commit()

    out = stats_service.get_overview(db_session, now=NOW)

    assert out.total_transactions_24h == 3
    assert out.approved_count_24h == 1
    assert out.declined_count_24h == 1
    # pending_review is queue depth, not 24h window — counts both REVIEW rows.
    assert out.pending_review_count == 2
    assert out.approved_rate == pytest.approx(1 / 3)
    assert out.fraud_caught_amount == Decimal("750.00")  # 500 + 250, 2 dp
    assert out.avg_fraud_score == pytest.approx((0.05 + 0.95 + 0.55) / 3)


def test_overview_quantises_fraud_amount_to_two_dp(db_session: Session) -> None:
    db_session.add(
        _make_tx(
            tx_id="x",
            created_at=NOW - timedelta(hours=1),
            decision=Decision.DECLINE,
            amount="123.4567",
        )
    )
    db_session.commit()

    out = stats_service.get_overview(db_session, now=NOW)

    # 123.4567 quantised half-up → 123.46
    assert out.fraud_caught_amount == Decimal("123.46")
    assert out.fraud_caught_amount.as_tuple().exponent == -2


# ---------------------------------------------------------------------------
# get_timeseries
# ---------------------------------------------------------------------------


def test_timeseries_empty_window_returns_24_zero_filled_buckets(
    db_session: Session,
) -> None:
    out = stats_service.get_timeseries(db_session, now=NOW)

    assert out.window == "24h"
    assert len(out.points) == 24
    assert all(p.transaction_count == 0 for p in out.points)
    assert all(p.fraud_rate == 0.0 for p in out.points)
    # First bucket is 23 hours before the floor of `now`'s hour; last
    # bucket is the floor of `now`'s hour. Both inclusive → 24 points.
    end = NOW.replace(minute=0, second=0, microsecond=0)
    assert out.points[0].timestamp == end - timedelta(hours=23)
    assert out.points[-1].timestamp == end


def test_timeseries_computes_fraud_rate_per_bucket(db_session: Session) -> None:
    # Bucket containing `NOW - 5h`: 4 approves + 1 decline → fraud_rate 0.2.
    bucket_anchor = (NOW - timedelta(hours=5)).replace(minute=30)
    for i in range(4):
        db_session.add(
            _make_tx(
                tx_id=f"ok-{i}",
                created_at=bucket_anchor,
                decision=Decision.APPROVE,
            )
        )
    db_session.add(
        _make_tx(
            tx_id="bad-1",
            created_at=bucket_anchor,
            decision=Decision.DECLINE,
        )
    )
    db_session.commit()

    out = stats_service.get_timeseries(db_session, now=NOW)
    target_hour = bucket_anchor.replace(minute=0, second=0, microsecond=0)
    point = next(p for p in out.points if p.timestamp == target_hour)

    assert point.transaction_count == 5
    assert point.fraud_rate == pytest.approx(0.2)


def test_timeseries_ignores_rows_outside_window(db_session: Session) -> None:
    db_session.add(
        _make_tx(
            tx_id="old",
            created_at=NOW - timedelta(hours=48),
            decision=Decision.DECLINE,
        )
    )
    db_session.commit()

    out = stats_service.get_timeseries(db_session, now=NOW)

    assert sum(p.transaction_count for p in out.points) == 0


# ---------------------------------------------------------------------------
# get_breakdown
# ---------------------------------------------------------------------------


def test_breakdown_empty_window_returns_empty_items(db_session: Session) -> None:
    out = stats_service.get_breakdown(db_session, now=NOW)

    assert out.dimension == "country"
    assert out.items == []


def test_breakdown_sorts_by_tx_count_desc_and_quantises_amount(
    db_session: Session,
) -> None:
    # US: 3 txs (1 decline). GB: 1 tx. DE: 2 txs.
    db_session.add_all(
        [
            _make_tx(
                tx_id="us-1",
                created_at=NOW - timedelta(hours=1),
                decision=Decision.APPROVE,
                country="US",
                amount="100.0050",  # quantise half-up → 100.01
            ),
            _make_tx(
                tx_id="us-2",
                created_at=NOW - timedelta(hours=1),
                decision=Decision.APPROVE,
                country="US",
                amount="50.00",
            ),
            _make_tx(
                tx_id="us-3",
                created_at=NOW - timedelta(hours=1),
                decision=Decision.DECLINE,
                country="US",
                amount="200.00",
            ),
            _make_tx(
                tx_id="de-1",
                created_at=NOW - timedelta(hours=2),
                decision=Decision.APPROVE,
                country="DE",
                amount="75.00",
            ),
            _make_tx(
                tx_id="de-2",
                created_at=NOW - timedelta(hours=2),
                decision=Decision.APPROVE,
                country="DE",
                amount="25.00",
            ),
            _make_tx(
                tx_id="gb-1",
                created_at=NOW - timedelta(hours=3),
                decision=Decision.APPROVE,
                country="GB",
                amount="10.00",
            ),
        ]
    )
    db_session.commit()

    out = stats_service.get_breakdown(db_session, now=NOW)

    assert [item.category for item in out.items] == ["US", "DE", "GB"]
    us = out.items[0]
    assert us.transaction_count == 3
    assert us.declined_count == 1
    assert us.total_amount == Decimal("350.01")  # 100.01 + 50 + 200
    assert us.total_amount.as_tuple().exponent == -2


def test_breakdown_caps_at_top_10_countries(db_session: Session) -> None:
    # 12 countries, each with a distinct row count so the order is
    # unambiguous: country i gets (i+1) rows.
    for i in range(12):
        country_code = f"X{i:02d}"[:2]  # 2-char codes: X0, X1, … X9, X1, X1
        # Force unique 2-char codes by mapping i to a letter pair.
        country_code = chr(65 + (i // 26)) + chr(65 + (i % 26))
        for j in range(i + 1):
            db_session.add(
                _make_tx(
                    tx_id=f"{country_code}-{j}",
                    created_at=NOW - timedelta(hours=1),
                    decision=Decision.APPROVE,
                    country=country_code,
                )
            )
    db_session.commit()

    out = stats_service.get_breakdown(db_session, now=NOW)

    assert len(out.items) == 10
    counts = [item.transaction_count for item in out.items]
    # Sorted descending; top of the heap should be the 12th country
    # (12 rows) then 11, 10, … down to 3.
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 12
    assert counts[-1] == 3
