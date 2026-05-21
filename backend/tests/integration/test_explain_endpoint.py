"""Integration tests for the /explain endpoint.

These tests construct the FastAPI app with dependency overrides: an
in-memory SQLite + a stub explainer that returns deterministic SHAP values.
That keeps the suite passing on a fresh clone without first running
`ml.train`.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.fraud.explainer import (
    FraudExplainer,
    get_explainer,
    reset_explainer_for_tests,
)
from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES
from app.main import app
from app.models import Customer, Merchant, Transaction
from app.models.base import Base


KNOWN_TX_ID = "11111111-1111-1111-1111-111111111111"
KNOWN_CUSTOMER_ID = "22222222-2222-2222-2222-222222222222"
KNOWN_MERCHANT_ID = "33333333-3333-3333-3333-333333333333"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session() -> Iterator[Session]:
    # `:memory:` databases live inside a SQLite connection, not an engine.
    # SQLAlchemy's default pool (`SingletonThreadPool`) hands each thread its
    # own connection — which means the threadpool that Starlette runs sync
    # route handlers on would see a brand-new empty database. `StaticPool`
    # forces a single shared connection across all threads, so the route and
    # the fixture see the same tables and rows.
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
        id=KNOWN_CUSTOMER_ID,
        email="x@example.com",
        full_name="Test User",
        country="US",
        risk_tier="LOW",
        account_age_days=365,
    ))
    db.add(Merchant(
        id=KNOWN_MERCHANT_ID,
        name="Test Co",
        category="RETAIL",
        mcc="5311",
        country="US",
        risk_rating="LOW",
    ))
    db.add(Transaction(
        id=KNOWN_TX_ID,
        idempotency_key="key-1",
        customer_id=KNOWN_CUSTOMER_ID,
        merchant_id=KNOWN_MERCHANT_ID,
        amount=Decimal("100.00"),
        currency="USD",
        status="APPROVED",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
    ))
    db.commit()

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Stub explainer — deterministic, no shap dependency for these tests
# ---------------------------------------------------------------------------

class _StubExplainer:
    """Minimal stub that matches the FraudExplainer surface used by the route."""

    threshold = 0.6

    def explain_local(self, x_row: np.ndarray):  # type: ignore[no-untyped-def]
        from app.fraud.explainer import LocalExplanation
        shap_values = np.linspace(-0.2, 0.4, N_FEATURES)
        return LocalExplanation(
            fraud_score=0.75,  # > threshold → DECLINE
            shap_values=shap_values,
            base_value=0.1,
        )

    def classify(self, fraud_score: float) -> str:
        if fraud_score >= self.threshold:
            return "DECLINE"
        if fraud_score >= self.threshold * 0.5:
            return "REVIEW"
        return "APPROVE"


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    reset_explainer_for_tests()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_explainer] = lambda: _StubExplainer()
    try:
        # Don't run lifespan — it would try to load real artifacts.
        with TestClient(app, raise_server_exceptions=True) as _client:
            yield _client
    finally:
        app.dependency_overrides.clear()
        reset_explainer_for_tests()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_explain_returns_json_schema(client: TestClient) -> None:
    resp = client.get(f"/api/v1/transactions/{KNOWN_TX_ID}/explain")
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert payload["transaction_id"] == KNOWN_TX_ID
    assert payload["fraud_score"] == pytest.approx(0.75)
    assert payload["decision"] == "DECLINE"
    assert payload["threshold"] == pytest.approx(0.6)
    assert payload["base_value"] == pytest.approx(0.1)
    assert len(payload["top_contributors"]) == 5
    assert set(payload["all_shap_values"].keys()) == set(FEATURE_NAMES)

    # Top contributors should be sorted by |shap_value| descending
    abs_values = [abs(c["shap_value"]) for c in payload["top_contributors"]]
    assert abs_values == sorted(abs_values, reverse=True)


def test_explain_force_returns_png(client: TestClient) -> None:
    resp = client.get(
        f"/api/v1/transactions/{KNOWN_TX_ID}/explain",
        params={"format": "force"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(PNG_MAGIC)


def test_explain_waterfall_returns_png(client: TestClient) -> None:
    resp = client.get(
        f"/api/v1/transactions/{KNOWN_TX_ID}/explain",
        params={"format": "waterfall"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(PNG_MAGIC)


def test_explain_unknown_transaction_returns_404(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions/00000000-0000-0000-0000-000000000000/explain"
    )
    assert resp.status_code == 404


def test_explain_invalid_format_returns_422(client: TestClient) -> None:
    resp = client.get(
        f"/api/v1/transactions/{KNOWN_TX_ID}/explain",
        params={"format": "garbage"},
    )
    assert resp.status_code == 422
