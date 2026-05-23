"""Integration tests for the Phase 3B ingestion endpoint.

In-memory SQLite (`StaticPool` so the shared connection survives across
Starlette's request threadpool — see test_explain_endpoint.py for the
rationale). The FastAPI `lifespan` is intentionally not invoked: loading
the SHAP explainer would couple this suite to Phase 2G artifacts being
on disk. Instead, `app.services.scoring.get_explainer` is monkeypatched
to return a deterministic stub so Phase 3C-2's POST endpoint scores
without touching the trained model.

The truly artifact-dependent integration tests live in
test_scoring_endpoint.py; this file owns the contract surface (status
codes, idempotency, validation).
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import UUID

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.fraud.explainer import LocalExplanation
from app.main import app
from app.models import Customer, Merchant
from app.models.base import Base


KNOWN_CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
KNOWN_MERCHANT_ID = "22222222-2222-2222-2222-222222222222"


def _payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid TransactionCreate body as a plain dict for TestClient."""
    base: dict[str, Any] = {
        "customer_id": KNOWN_CUSTOMER_ID,
        "merchant_id": KNOWN_MERCHANT_ID,
        "amount": "100.00",
        "currency": "USD",
        "payment_method": "CARD",
        "country": "US",
        "is_card_present": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
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
    db.add(Customer(
        id=KNOWN_CUSTOMER_ID,
        email="seed@example.com",
        full_name="Seed Customer",
        country="US",
        risk_tier="LOW",
        account_age_days=365,
    ))
    db.add(Merchant(
        id=KNOWN_MERCHANT_ID,
        name="Seed Merchant",
        category="RETAIL",
        mcc="5311",
        country="US",
        risk_rating="LOW",
    ))
    db.commit()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


class _StubExplainer:
    """Minimal FraudExplainer surface for Phase 3C-2 scoring.

    Returns a low fraud_score so the default seeded payload (US $100
    card-present, no recent history) consistently classifies as APPROVE.
    Tests that need a specific decision (REVIEW/DECLINE) trigger a rule
    that overrides the model output.
    """

    threshold = 0.5

    def explain_local(self, x_row: np.ndarray) -> LocalExplanation:
        return LocalExplanation(
            fraud_score=0.05,
            shap_values=np.linspace(-0.05, 0.05, 17),
            base_value=0.02,
        )


@pytest.fixture
def client(
    db_session: Session, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    """TestClient without lifespan; stub explainer injected for scoring."""
    app.dependency_overrides[get_db] = lambda: db_session
    import app.services.scoring as scoring_module
    monkeypatch.setattr(scoring_module, "get_explainer", lambda: _StubExplainer())
    try:
        yield TestClient(app, raise_server_exceptions=True)
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /transactions — happy path
# ---------------------------------------------------------------------------


def test_post_returns_201_with_real_decision_for_new_key(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-new-1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()

    # Phase 3C-2: the scoring service now resolves the real decision on POST.
    # The stub explainer returns a low fraud_score, so the clean seeded
    # payload classifies as APPROVE; we assert membership in the valid set
    # so the test stays robust if the stub is retuned later.
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert isinstance(body["fraud_score"], float)
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert isinstance(body["threshold"], float)
    assert body["rules_triggered"] == []
    assert len(body["top_contributors"]) == 5

    # transaction_id must be a valid UUID string
    UUID(body["transaction_id"])  # raises ValueError if malformed

    # computed_at must parse as an ISO 8601 timestamp
    from datetime import datetime
    datetime.fromisoformat(body["computed_at"].replace("Z", "+00:00"))

    # First-time POST must not carry the replay header
    assert resp.headers.get("X-Idempotency-Replay") in (None, "false")


def test_post_persists_transaction_with_real_decision(client: TestClient) -> None:
    post_resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-persist-1"},
    )
    assert post_resp.status_code == 201
    posted = post_resp.json()
    tx_id = posted["transaction_id"]

    get_resp = client.get(f"/api/v1/transactions/{tx_id}")
    assert get_resp.status_code == 200, get_resp.text
    fetched = get_resp.json()

    assert fetched["transaction_id"] == tx_id
    assert fetched["decision"] == posted["decision"]
    assert fetched["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    # `fraud_score` round-trips through Numeric(5, 4) so precision differs
    # by up to a half-ulp at the 4th decimal. approx handles it.
    assert fetched["fraud_score"] == pytest.approx(posted["fraud_score"], abs=1e-4)
    assert fetched["rules_triggered"] == posted["rules_triggered"]
    assert fetched["top_contributors"] == posted["top_contributors"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_post_returns_replay_with_header_for_repeated_key(client: TestClient) -> None:
    body = _payload()
    key = "k-replay-1"

    first = client.post(
        "/api/v1/transactions", json=body,
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201, first.text

    second = client.post(
        "/api/v1/transactions", json=body,
        headers={"Idempotency-Key": key},
    )
    assert second.status_code == first.status_code == 201
    assert second.headers.get("X-Idempotency-Replay") == "true"
    # Compare parsed JSON, not raw bytes — Pydantic serialisation order can
    # vary across FastAPI minor versions, but the logical body must match.
    assert second.json() == first.json()


def test_post_returns_409_for_same_key_different_body(client: TestClient) -> None:
    key = "k-conflict-1"
    first = client.post(
        "/api/v1/transactions",
        json=_payload(amount="100.00"),
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201

    second = client.post(
        "/api/v1/transactions",
        json=_payload(amount="200.00"),  # mutated payload, same key
        headers={"Idempotency-Key": key},
    )
    assert second.status_code == 409
    detail = second.json()["detail"].lower()
    assert "idempotency key" in detail
    assert "different payload" in detail


# ---------------------------------------------------------------------------
# 422 validation — Idempotency-Key header
# ---------------------------------------------------------------------------


def test_post_returns_422_when_idempotency_key_missing(client: TestClient) -> None:
    resp = client.post("/api/v1/transactions", json=_payload())
    assert resp.status_code == 422


def test_post_returns_422_when_idempotency_key_empty(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": ""},
    )
    assert resp.status_code == 422


def test_post_returns_422_when_idempotency_key_too_long(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "x" * 65},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 422 validation — request body
# ---------------------------------------------------------------------------


def test_post_returns_422_for_invalid_transaction_body(client: TestClient) -> None:
    """A 4-character currency code violates the ISO 4217 length constraint."""
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(currency="USDD"),
        headers={"Idempotency-Key": "k-bad-body"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /transactions/{id}
# ---------------------------------------------------------------------------


def test_get_unknown_transaction_returns_404(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions/99999999-9999-9999-9999-999999999999",
    )
    assert resp.status_code == 404


def test_get_returns_persisted_transaction(client: TestClient) -> None:
    post_resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-get-flow"},
    )
    assert post_resp.status_code == 201
    posted = post_resp.json()
    tx_id = posted["transaction_id"]

    get_resp = client.get(f"/api/v1/transactions/{tx_id}")
    assert get_resp.status_code == 200, get_resp.text
    fetched = get_resp.json()

    assert fetched["transaction_id"] == tx_id
    assert fetched["decision"] == posted["decision"]
    assert fetched["fraud_score"] == pytest.approx(posted["fraud_score"], abs=1e-4)
    # GET reads from the transactions row, which does not persist the
    # operating threshold (it lives in the explainer artifact). The POST
    # response includes threshold; the GET response carries None. This
    # asymmetry is documented in `_scored_from_transaction`.
    assert fetched["threshold"] is None
    assert fetched["rules_triggered"] == posted["rules_triggered"]
    assert fetched["top_contributors"] == posted["top_contributors"]
