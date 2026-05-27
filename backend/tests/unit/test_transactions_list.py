"""Unit tests for the transactions list cursor codec and keyset pagination."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.fraud.decision import Decision
from app.models import Customer, Merchant
from app.models.base import Base
from app.models.transaction import Transaction
from app.repositories.transaction import (
    TransactionListFilters,
    transaction_repository,
)
from app.services.transactions import decode_cursor, encode_cursor

CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"
OTHER_CUSTOMER_ID = "33333333-3333-3333-3333-333333333333"


# ---------------------------------------------------------------------------
# Cursor codec
# ---------------------------------------------------------------------------


def test_cursor_roundtrip_preserves_timestamp_and_id() -> None:
    ts = datetime(2026, 5, 27, 12, 34, 56, 789, tzinfo=UTC)
    tx_id = "abc-123"
    cursor = encode_cursor(ts, tx_id)
    decoded_ts, decoded_id = decode_cursor(cursor)
    assert decoded_ts == ts
    assert decoded_id == tx_id


def test_cursor_is_url_safe() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    cursor = encode_cursor(ts, "id-with-dashes")
    # urlsafe base64 uses '-' and '_', never '+' or '/'.
    assert "+" not in cursor
    assert "/" not in cursor
    # And we strip trailing padding so the token is concise.
    assert not cursor.endswith("=")


@pytest.mark.parametrize(
    "bad",
    ["not-base64!!!", "", "AAAA", "eyJ0cyI6IDEyM30"],  # last one: missing id
)
def test_decode_cursor_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        decode_cursor(bad)


# ---------------------------------------------------------------------------
# Repository keyset pagination
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> Iterator[Session]:
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
    session = SessionTesting()
    session.add(Customer(
        id=CUSTOMER_ID, email="a@example.com", full_name="A",
        country="US", risk_tier="LOW", account_age_days=100,
    ))
    session.add(Customer(
        id=OTHER_CUSTOMER_ID, email="b@example.com", full_name="B",
        country="GB", risk_tier="LOW", account_age_days=100,
    ))
    session.add(Merchant(
        id=MERCHANT_ID, name="M", category="RETAIL", mcc="5311",
        country="US", risk_rating="LOW",
    ))
    session.commit()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _make_tx(
    db: Session,
    *,
    tx_id: str,
    created_at: datetime,
    customer_id: str = CUSTOMER_ID,
    amount: str = "100.00",
    country: str = "US",
    decision: str = "APPROVE",
) -> Transaction:
    tx = Transaction(
        id=tx_id,
        idempotency_key=f"key-{tx_id}",
        customer_id=customer_id,
        merchant_id=MERCHANT_ID,
        amount=Decimal(amount),
        currency="USD",
        status="APPROVED",
        payment_method="CARD",
        country=country,
        is_card_present=True,
        fraud_score=Decimal("0.1"),
        fraud_decision=decision,
        created_at=created_at,
    )
    db.add(tx)
    db.commit()
    return tx


def test_list_paginated_returns_newest_first(db: Session) -> None:
    base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    _make_tx(db, tx_id="tx-old", created_at=base)
    _make_tx(db, tx_id="tx-mid", created_at=base + timedelta(hours=1))
    _make_tx(db, tx_id="tx-new", created_at=base + timedelta(hours=2))

    rows, next_cursor = transaction_repository.list_paginated(
        db, filters=TransactionListFilters(), limit=10,
    )
    assert [r.id for r in rows] == ["tx-new", "tx-mid", "tx-old"]
    assert next_cursor is None


def test_list_paginated_emits_cursor_when_more_rows_exist(db: Session) -> None:
    base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    for i in range(5):
        _make_tx(
            db,
            tx_id=f"tx-{i:02d}",
            created_at=base + timedelta(minutes=i),
        )

    rows, next_cursor = transaction_repository.list_paginated(
        db, filters=TransactionListFilters(), limit=2,
    )
    assert len(rows) == 2
    assert next_cursor is not None
    # Page 2
    rows2, next2 = transaction_repository.list_paginated(
        db, filters=TransactionListFilters(), limit=2, cursor=next_cursor,
    )
    assert len(rows2) == 2
    assert next2 is not None
    # Page 3 — only one row left
    rows3, next3 = transaction_repository.list_paginated(
        db, filters=TransactionListFilters(), limit=2, cursor=next2,
    )
    assert len(rows3) == 1
    assert next3 is None
    # No overlap across pages
    seen = {r.id for r in rows} | {r.id for r in rows2} | {r.id for r in rows3}
    assert len(seen) == 5


def test_list_paginated_tie_breaks_on_id_when_timestamps_equal(
    db: Session,
) -> None:
    """Same created_at across rows must still produce a stable cursor walk."""
    ts = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    for tx_id in ("tx-a", "tx-b", "tx-c"):
        _make_tx(db, tx_id=tx_id, created_at=ts)

    page1, cursor1 = transaction_repository.list_paginated(
        db, filters=TransactionListFilters(), limit=2,
    )
    assert len(page1) == 2
    assert cursor1 is not None
    page2, cursor2 = transaction_repository.list_paginated(
        db, filters=TransactionListFilters(), limit=2, cursor=cursor1,
    )
    assert len(page2) == 1
    assert cursor2 is None
    seen = {r.id for r in page1} | {r.id for r in page2}
    assert seen == {"tx-a", "tx-b", "tx-c"}


def test_list_paginated_filters_by_decision(db: Session) -> None:
    base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    _make_tx(db, tx_id="tx-approve", created_at=base, decision="APPROVE")
    _make_tx(db, tx_id="tx-review",
             created_at=base + timedelta(minutes=1), decision="REVIEW")
    _make_tx(db, tx_id="tx-decline",
             created_at=base + timedelta(minutes=2), decision="DECLINE")

    rows, _ = transaction_repository.list_paginated(
        db,
        filters=TransactionListFilters(decision=Decision.REVIEW),
        limit=10,
    )
    assert [r.id for r in rows] == ["tx-review"]


def test_list_paginated_filters_by_amount_range(db: Session) -> None:
    base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    _make_tx(db, tx_id="tx-50", created_at=base, amount="50.00")
    _make_tx(db, tx_id="tx-150",
             created_at=base + timedelta(minutes=1), amount="150.00")
    _make_tx(db, tx_id="tx-500",
             created_at=base + timedelta(minutes=2), amount="500.00")

    rows, _ = transaction_repository.list_paginated(
        db,
        filters=TransactionListFilters(
            min_amount=Decimal("100"), max_amount=Decimal("200"),
        ),
        limit=10,
    )
    assert [r.id for r in rows] == ["tx-150"]


def test_list_paginated_filters_by_country_and_customer(db: Session) -> None:
    base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    _make_tx(db, tx_id="tx-us-a", created_at=base, country="US",
             customer_id=CUSTOMER_ID)
    _make_tx(db, tx_id="tx-gb-a",
             created_at=base + timedelta(minutes=1), country="GB",
             customer_id=CUSTOMER_ID)
    _make_tx(db, tx_id="tx-us-b",
             created_at=base + timedelta(minutes=2), country="US",
             customer_id=OTHER_CUSTOMER_ID)

    rows, _ = transaction_repository.list_paginated(
        db,
        filters=TransactionListFilters(country="US", customer_id=CUSTOMER_ID),
        limit=10,
    )
    assert [r.id for r in rows] == ["tx-us-a"]


def test_list_paginated_filters_by_time_window(db: Session) -> None:
    base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    _make_tx(db, tx_id="tx-before", created_at=base - timedelta(hours=1))
    _make_tx(db, tx_id="tx-in",
             created_at=base + timedelta(minutes=30))
    _make_tx(db, tx_id="tx-after",
             created_at=base + timedelta(hours=2))

    rows, _ = transaction_repository.list_paginated(
        db,
        filters=TransactionListFilters(
            start_time=base,
            end_time=base + timedelta(hours=1),
        ),
        limit=10,
    )
    assert [r.id for r in rows] == ["tx-in"]
