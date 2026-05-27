"""Integration tests for GET /api/v1/transactions (Phase 3F)."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.main import app
from app.models import Customer, Merchant
from app.models.base import Base
from app.models.transaction import Transaction

CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"
OTHER_CUSTOMER_ID = "33333333-3333-3333-3333-333333333333"


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
    db.add(Customer(
        id=CUSTOMER_ID, email="a@example.com", full_name="A",
        country="US", risk_tier="LOW", account_age_days=100,
    ))
    db.add(Customer(
        id=OTHER_CUSTOMER_ID, email="b@example.com", full_name="B",
        country="GB", risk_tier="LOW", account_age_days=100,
    ))
    db.add(Merchant(
        id=MERCHANT_ID, name="M", category="RETAIL", mcc="5311",
        country="US", risk_rating="LOW",
    ))
    db.commit()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        yield TestClient(app, raise_server_exceptions=True)
    finally:
        app.dependency_overrides.clear()


def _seed(
    db: Session,
    *,
    count: int,
    base: datetime | None = None,
    country: str = "US",
    decision: str = "APPROVE",
    amount: str = "100.00",
    customer_id: str = CUSTOMER_ID,
    prefix: str | None = None,
) -> list[str]:
    """Insert `count` transactions one minute apart, return their ids."""
    if base is None:
        base = datetime(2026, 5, 27, 10, 0, 0, tzinfo=UTC)
    tag = prefix or f"{decision.lower()}-{base.strftime('%H%M')}"
    ids: list[str] = []
    for i in range(count):
        tx_id = f"tx-{tag}-{i:03d}"
        db.add(Transaction(
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
            created_at=base + timedelta(minutes=i),
        ))
        ids.append(tx_id)
    db.commit()
    return ids


def test_list_returns_empty_envelope_when_no_rows(client: TestClient) -> None:
    resp = client.get("/api/v1/transactions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"items": [], "next_cursor": None, "has_more": False}


def test_list_returns_rows_newest_first(
    client: TestClient, db_session: Session,
) -> None:
    ids = _seed(db_session, count=3)
    resp = client.get("/api/v1/transactions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    returned = [item["id"] for item in body["items"]]
    assert returned == list(reversed(ids))
    assert body["has_more"] is False
    assert body["next_cursor"] is None


def test_list_paginates_via_next_cursor(
    client: TestClient, db_session: Session,
) -> None:
    ids = _seed(db_session, count=5)
    expected_order = list(reversed(ids))

    resp1 = client.get("/api/v1/transactions", params={"limit": 2})
    assert resp1.status_code == 200
    page1 = resp1.json()
    assert [i["id"] for i in page1["items"]] == expected_order[:2]
    assert page1["has_more"] is True
    cursor = page1["next_cursor"]
    assert cursor

    resp2 = client.get(
        "/api/v1/transactions", params={"limit": 2, "cursor": cursor},
    )
    page2 = resp2.json()
    assert [i["id"] for i in page2["items"]] == expected_order[2:4]
    assert page2["has_more"] is True

    resp3 = client.get(
        "/api/v1/transactions",
        params={"limit": 2, "cursor": page2["next_cursor"]},
    )
    page3 = resp3.json()
    assert [i["id"] for i in page3["items"]] == expected_order[4:5]
    assert page3["has_more"] is False
    assert page3["next_cursor"] is None


def test_list_filters_by_decision(
    client: TestClient, db_session: Session,
) -> None:
    _seed(db_session, count=2, decision="APPROVE")
    _seed(db_session, count=3, decision="REVIEW",
          base=datetime(2026, 5, 27, 11, 0, 0, tzinfo=UTC))
    resp = client.get(
        "/api/v1/transactions", params={"decision": "REVIEW"},
    )
    body = resp.json()
    assert len(body["items"]) == 3
    assert all(i["fraud_decision"] == "REVIEW" for i in body["items"])


def test_list_filters_by_country_and_amount(
    client: TestClient, db_session: Session,
) -> None:
    _seed(db_session, count=1, country="US", amount="50.00")
    _seed(db_session, count=1, country="US", amount="150.00",
          base=datetime(2026, 5, 27, 11, 0, 0, tzinfo=UTC))
    _seed(db_session, count=1, country="GB", amount="200.00",
          base=datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC))
    resp = client.get(
        "/api/v1/transactions",
        params={
            "country": "US",
            "min_amount": "100",
            "max_amount": "200",
        },
    )
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["country"] == "US"
    assert Decimal(body["items"][0]["amount"]) == Decimal("150.00")


def test_list_rejects_min_amount_greater_than_max(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions",
        params={"min_amount": "500", "max_amount": "100"},
    )
    assert resp.status_code == 422


def test_list_rejects_start_time_after_end_time(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions",
        params={
            "start_time": "2026-05-27T12:00:00Z",
            "end_time": "2026-05-27T10:00:00Z",
        },
    )
    assert resp.status_code == 422


def test_list_rejects_limit_above_cap(client: TestClient) -> None:
    resp = client.get("/api/v1/transactions", params={"limit": 500})
    assert resp.status_code == 422


def test_list_rejects_invalid_decision(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions", params={"decision": "MAYBE"},
    )
    assert resp.status_code == 422


def test_list_rejects_malformed_cursor(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions", params={"cursor": "not-a-real-cursor!!!"},
    )
    assert resp.status_code == 422
