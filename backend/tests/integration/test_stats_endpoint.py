"""Integration tests for the Phase 3E /stats endpoints.

In-memory SQLite via the same StaticPool pattern used by
test_transactions_endpoint.py — the FastAPI ``lifespan`` is bypassed
because the stats endpoints do not touch the SHAP explainer and we
don't want to couple this suite to Phase 2G artifacts being on disk.

Covers:
- Happy-path 200 + schema shape for all three endpoints.
- Empty-database edge case for each endpoint.
- Literal-query 422 rejection for ``window=7d`` and
  ``dimension=category``.
- CORS preflight for the configured dev origin.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.fraud.decision import Decision
from app.main import app
from app.models import Customer, Merchant
from app.models.base import Base
from app.models.transaction import Transaction

CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"
DASHBOARD_ORIGIN = "http://localhost:5173"


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


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    # Lifespan is intentionally not entered; instantiate TestClient
    # without a context manager so the SHAP explainer is not loaded.
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed_one_tx(db: Session) -> None:
    from datetime import datetime

    db.add(
        Transaction(
            id="seed-tx-1",
            idempotency_key="seed-key-1",
            customer_id=CUSTOMER_ID,
            merchant_id=MERCHANT_ID,
            amount=Decimal("250.00"),
            currency="USD",
            status="DECLINED",
            payment_method="CARD",
            country="US",
            is_card_present=True,
            fraud_score=0.91,
            fraud_decision=Decision.DECLINE.value,
            created_at=datetime.now(UTC),
        )
    )
    db.commit()


# ---------------------------------------------------------------------------
# /overview
# ---------------------------------------------------------------------------


def test_overview_empty_db_returns_200_and_zeros(client: TestClient) -> None:
    resp = client.get("/api/v1/stats/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions_24h"] == 0
    assert body["approved_count_24h"] == 0
    assert body["declined_count_24h"] == 0
    assert body["pending_review_count"] == 0
    assert body["approved_rate"] == 0.0
    assert body["fraud_caught_amount"] == "0.00"
    assert body["avg_fraud_score"] is None


def test_overview_reflects_seeded_decline(
    client: TestClient, db_session: Session,
) -> None:
    _seed_one_tx(db_session)
    resp = client.get("/api/v1/stats/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_transactions_24h"] == 1
    assert body["declined_count_24h"] == 1
    assert body["fraud_caught_amount"] == "250.00"
    assert body["avg_fraud_score"] == pytest.approx(0.91, rel=1e-3)


# ---------------------------------------------------------------------------
# /timeseries
# ---------------------------------------------------------------------------


def test_timeseries_returns_24_buckets(client: TestClient) -> None:
    resp = client.get("/api/v1/stats/timeseries")
    assert resp.status_code == 200
    body = resp.json()
    assert body["window"] == "24h"
    assert len(body["points"]) == 24
    for point in body["points"]:
        assert point["transaction_count"] == 0
        assert point["fraud_rate"] == 0.0
        assert "timestamp" in point


def test_timeseries_rejects_unsupported_window(client: TestClient) -> None:
    resp = client.get("/api/v1/stats/timeseries", params={"window": "7d"})
    assert resp.status_code == 422


def test_timeseries_rejects_unsupported_bucket(client: TestClient) -> None:
    resp = client.get("/api/v1/stats/timeseries", params={"bucket": "5m"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /breakdown
# ---------------------------------------------------------------------------


def test_breakdown_empty_db_returns_empty_items(client: TestClient) -> None:
    resp = client.get("/api/v1/stats/breakdown")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimension"] == "country"
    assert body["items"] == []


def test_breakdown_reflects_seeded_country(
    client: TestClient, db_session: Session,
) -> None:
    _seed_one_tx(db_session)
    resp = client.get("/api/v1/stats/breakdown")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["category"] == "US"
    assert item["transaction_count"] == 1
    assert item["declined_count"] == 1
    assert item["total_amount"] == "250.00"


def test_breakdown_rejects_unsupported_dimension(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/stats/breakdown", params={"dimension": "category"}
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_preflight_from_dashboard_origin_is_allowed(
    client: TestClient,
) -> None:
    resp = client.options(
        "/api/v1/stats/overview",
        headers={
            "Origin": DASHBOARD_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    # Starlette's CORSMiddleware returns 200 for a valid preflight.
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == DASHBOARD_ORIGIN
    allowed_methods = resp.headers.get("access-control-allow-methods", "")
    assert "GET" in allowed_methods


def test_cors_response_includes_origin_header_on_simple_get(
    client: TestClient,
) -> None:
    resp = client.get(
        "/api/v1/stats/overview", headers={"Origin": DASHBOARD_ORIGIN}
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == DASHBOARD_ORIGIN
