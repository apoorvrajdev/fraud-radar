"""Unit tests for the Phase 3B idempotency service.

In-memory SQLite via `StaticPool` so the engine is shared across the
fixture and any threaded callers (matches the Phase 2H integration
pattern). Tests cover hashing determinism, lookup/expiry semantics, and
the 24-hour TTL on `store`.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Customer, Merchant, Transaction
from app.models.base import Base
from app.models.idempotency_key import IdempotencyKey
from app.schemas.transaction import TransactionCreate
from app.services.idempotency import TTL, hash_request, lookup, store


_SEED_CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
_SEED_MERCHANT_ID = "22222222-2222-2222-2222-222222222222"
_SEED_TX_ID = "33333333-3333-3333-3333-333333333333"


def _payload(**overrides: Any) -> TransactionCreate:
    base: dict[str, Any] = dict(
        customer_id=_SEED_CUSTOMER_ID,
        merchant_id=_SEED_MERCHANT_ID,
        amount=Decimal("100.00"),
        currency="USD",
        payment_method="CARD",
        country="US",
        is_card_present=True,
    )
    base.update(overrides)
    return TransactionCreate(**base)


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    SessionTesting = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = SessionTesting()

    db.add(Customer(
        id=_SEED_CUSTOMER_ID,
        email="seed@example.com",
        full_name="Seed Customer",
        country="US",
        risk_tier="LOW",
        account_age_days=365,
    ))
    db.add(Merchant(
        id=_SEED_MERCHANT_ID,
        name="Seed Merchant",
        category="RETAIL",
        mcc="5311",
        country="US",
        risk_rating="LOW",
    ))
    db.add(Transaction(
        id=_SEED_TX_ID,
        idempotency_key="seed-key",
        customer_id=_SEED_CUSTOMER_ID,
        merchant_id=_SEED_MERCHANT_ID,
        amount=Decimal("100.00"),
        currency="USD",
        status="PENDING_REVIEW",
        payment_method="CARD",
        country="US",
        is_card_present=True,
    ))
    db.commit()

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# hash_request
# ---------------------------------------------------------------------------


def test_hash_request_is_deterministic() -> None:
    assert hash_request(_payload()) == hash_request(_payload())


def test_hash_request_differs_for_different_payloads() -> None:
    h1 = hash_request(_payload(amount=Decimal("100.00")))
    h2 = hash_request(_payload(amount=Decimal("200.00")))
    assert h1 != h2


def test_hash_request_stable_across_field_reordering() -> None:
    """Pydantic dumps in declaration order, so kwarg insertion order doesn't matter."""
    a = TransactionCreate(
        customer_id=_SEED_CUSTOMER_ID,
        merchant_id=_SEED_MERCHANT_ID,
        amount=Decimal("50.00"),
        currency="USD",
        payment_method="CARD",
        country="US",
    )
    b = TransactionCreate(
        country="US",
        payment_method="CARD",
        currency="USD",
        amount=Decimal("50.00"),
        merchant_id=_SEED_MERCHANT_ID,
        customer_id=_SEED_CUSTOMER_ID,
    )
    assert hash_request(a) == hash_request(b)


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


def test_lookup_returns_none_for_missing_key(db_session: Session) -> None:
    assert lookup(db_session, "does-not-exist") is None


def test_lookup_returns_none_for_expired_key(db_session: Session) -> None:
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.add(IdempotencyKey(
        key="expired-key",
        request_hash="abc",
        transaction_id=_SEED_TX_ID,
        response_body="{}",
        status_code=201,
        created_at=past - timedelta(hours=24),
        expires_at=past,
    ))
    db_session.commit()
    assert lookup(db_session, "expired-key") is None


def test_lookup_returns_entry_for_valid_key(db_session: Session) -> None:
    now = datetime.now(timezone.utc)
    db_session.add(IdempotencyKey(
        key="live-key",
        request_hash="xyz",
        transaction_id=_SEED_TX_ID,
        response_body='{"x":1}',
        status_code=201,
        created_at=now,
        expires_at=now + timedelta(hours=24),
    ))
    db_session.commit()
    result = lookup(db_session, "live-key")
    assert result is not None
    assert result.key == "live-key"
    assert result.request_hash == "xyz"
    assert result.response_body == '{"x":1}'


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


def test_store_sets_24h_ttl(db_session: Session) -> None:
    entry = store(
        db_session,
        key="new-key",
        request_hash="hash-x",
        transaction_id=_SEED_TX_ID,
        response_body='{"ok":true}',
        status_code=201,
    )
    assert entry.expires_at - entry.created_at == TTL


def test_store_persists_and_is_retrievable(db_session: Session) -> None:
    store(
        db_session,
        key="round-trip-key",
        request_hash="hash-r",
        transaction_id=_SEED_TX_ID,
        response_body='{"r":1}',
        status_code=201,
    )
    fetched = lookup(db_session, "round-trip-key")
    assert fetched is not None
    assert fetched.response_body == '{"r":1}'
    assert fetched.status_code == 201
