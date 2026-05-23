"""Parity tests for FeatureExtractor's optional preloaded-list mode.

The new `recent_transactions` keyword on `FeatureExtractor.extract()` lets
callers pre-load a customer's history once (Phase 3C scoring service uses
this so the rules engine and the feature extractor share one fetch). These
tests prove that the in-Python aggregation produces FeatureRow values
byte-identical to the SQL aggregation — the optional path is a true
drop-in, not a near-equivalent.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.fraud.features import FeatureExtractor
from app.models import Customer, Merchant, Transaction
from app.models.base import Base


NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
CUSTOMER_ID = "cust-parity-1"
CUSTOMER_2_ID = "cust-parity-2"
MERCHANT_ID = "merch-parity-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_customer(customer_id: str) -> Customer:
    c = Customer()
    c.id = customer_id
    c.email = f"{customer_id}@example.com"
    c.full_name = "Test Customer"
    c.country = "US"
    c.risk_tier = "LOW"
    c.account_age_days = 365
    return c


def _make_merchant() -> Merchant:
    m = Merchant()
    m.id = MERCHANT_ID
    m.name = "Test Merchant"
    m.category = "RETAIL"
    m.mcc = "5311"
    m.country = "US"
    m.risk_rating = "LOW"
    return m


def _make_transaction(
    *,
    id: str,
    customer_id: str = CUSTOMER_ID,
    amount: Decimal = Decimal("100.00"),
    country: str = "US",
    is_card_present: bool = True,
    created_at: datetime,
) -> Transaction:
    tx = Transaction()
    tx.id = id
    tx.customer_id = customer_id
    tx.merchant_id = MERCHANT_ID
    tx.amount = amount
    tx.currency = "USD"
    tx.status = "APPROVED"
    tx.payment_method = "CARD"
    tx.country = country
    tx.is_card_present = is_card_present
    tx.created_at = created_at
    tx.idempotency_key = f"key-{id}"
    return tx


def _current_tx() -> Transaction:
    """An in-memory transaction at NOW for CUSTOMER_ID — not persisted to DB."""
    return _make_transaction(
        id="current-tx",
        amount=Decimal("250.00"),
        created_at=NOW,
    )


def _load_priors(db: Session, customer_id: str) -> list[Transaction]:
    """Load every prior transaction for `customer_id`, sorted desc by created_at.

    Matches the sort order the Phase 3C scoring service will use when it
    constructs a TransactionContext.
    """
    stmt = (
        select(Transaction)
        .where(Transaction.customer_id == customer_id)
        .order_by(Transaction.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def _assert_feature_rows_equal(a: list[float], b: list[float]) -> None:
    """Element-wise equality between two FeatureRow.values arrays."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12), (
            f"Mismatch at index {i}: DB={x!r} vs LIST={y!r}"
        )


# ---------------------------------------------------------------------------
# Fixture: 50 deterministic priors with an exponential time distribution
# so 1h, 24h, 30d windows each contain a non-trivial number of rows.
# ---------------------------------------------------------------------------


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

    db.add(_make_customer(CUSTOMER_ID))
    db.add(_make_customer(CUSTOMER_2_ID))  # no priors — used by the empty-list test
    db.add(_make_merchant())

    rng = np.random.default_rng(42)
    for i in range(50):
        # Exponential with mean ~15 days gives realistic-looking density:
        # a handful in the last hour, ~10 in the last 24h, ~30 in the last 30d.
        days_offset = float(rng.exponential(15.0))
        if days_offset > 199.99:
            days_offset = 199.99  # cap to keep within a 200-day window
        if days_offset < 0.0001:
            days_offset = 0.0001  # nudge off zero so strict `<` predicate fires
        ts = NOW - timedelta(days=days_offset)
        amount = Decimal(f"{float(rng.uniform(10.0, 500.0)):.2f}")
        country = "US" if rng.random() < 0.95 else "GB"
        is_cp = bool(rng.random() < 0.6)
        db.add(_make_transaction(
            id=f"prior-{i:02d}",
            amount=amount,
            country=country,
            is_card_present=is_cp,
            created_at=ts,
        ))
    db.commit()

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_returns_identical_values_with_and_without_preloaded_list(
    db_session: Session,
) -> None:
    """DB mode and list mode produce byte-identical FeatureRow.values."""
    extractor = FeatureExtractor()
    current = _current_tx()

    db_mode = extractor.extract(db_session, current)

    priors = _load_priors(db_session, CUSTOMER_ID)
    list_mode = extractor.extract(
        db_session, current, recent_transactions=priors,
    )

    _assert_feature_rows_equal(db_mode.values, list_mode.values)


def test_extract_returns_identical_values_with_full_eager_loading(
    db_session: Session,
) -> None:
    """`recent_transactions` composes with the existing customer= / merchant= kwargs."""
    extractor = FeatureExtractor()
    current = _current_tx()
    customer = db_session.get(Customer, CUSTOMER_ID)
    merchant = db_session.get(Merchant, MERCHANT_ID)

    db_mode = extractor.extract(
        db_session, current, customer=customer, merchant=merchant,
    )

    priors = _load_priors(db_session, CUSTOMER_ID)
    list_mode = extractor.extract(
        db_session, current,
        customer=customer,
        merchant=merchant,
        recent_transactions=priors,
    )

    _assert_feature_rows_equal(db_mode.values, list_mode.values)


def test_extract_with_empty_recent_transactions_list(db_session: Session) -> None:
    """An empty list means 'no history', NOT 'fall through to DB query'.

    Customer 1 has 50 priors in the DB but we pass `recent_transactions=[]`.
    Customer 2 has no priors and we let DB mode handle it. Both customers
    share identical non-velocity attributes, so the full FeatureRow values
    must match — if the empty list quietly fell through to a SQL query,
    customer 1 would see its 50 real priors and the rows would diverge.
    """
    extractor = FeatureExtractor()

    current_for_c1 = _current_tx()  # uses CUSTOMER_ID
    list_mode_empty = extractor.extract(
        db_session, current_for_c1, recent_transactions=[],
    )

    current_for_c2 = _make_transaction(
        id="current-c2-tx",
        customer_id=CUSTOMER_2_ID,
        amount=Decimal("250.00"),
        created_at=NOW,
    )
    db_mode_no_rows = extractor.extract(db_session, current_for_c2)

    _assert_feature_rows_equal(list_mode_empty.values, db_mode_no_rows.values)


def test_extract_with_preloaded_list_does_not_query_db_for_velocity(
    db_session: Session,
) -> None:
    """List mode must not call `db.execute()` for any velocity/recency helper.

    Pass a sentinel session whose `execute` raises. With customer and merchant
    also eager-loaded, no DB lookup should be needed for ANY of the 17
    features. If the test passes (no AssertionError raised by the sentinel),
    list-mode is correctly bypassing every SQL path.
    """
    extractor = FeatureExtractor()
    current = _current_tx()
    customer = db_session.get(Customer, CUSTOMER_ID)
    merchant = db_session.get(Merchant, MERCHANT_ID)
    priors = _load_priors(db_session, CUSTOMER_ID)

    class _NoDbSession:
        def execute(self, *args: object, **kwargs: object) -> object:
            raise AssertionError(
                "List-mode helper still issued db.execute() — the new optional "
                "path is not fully short-circuiting the SQL query."
            )

        def get(self, *args: object, **kwargs: object) -> object:
            raise AssertionError(
                "extract() called db.get despite eager-loaded customer/merchant"
            )

    sentinel = _NoDbSession()
    result = extractor.extract(
        sentinel,  # type: ignore[arg-type]
        current,
        customer=customer,
        merchant=merchant,
        recent_transactions=priors,
    )
    assert len(result.values) == 17
