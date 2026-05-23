"""Phase 3C-2 integration tests — POST /transactions end-to-end.

Unlike test_transactions_endpoint.py (which monkeypatches the explainer
to stay artifact-free), these tests run the real FastAPI `lifespan` so
the live SHAP `TreeExplainer` is loaded from `backend/ml/artifacts/`.
That makes them real integration tests; it also means they SKIP cleanly
if the Phase 2G model artifacts are not on disk yet.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.fraud.explainer import reset_explainer_for_tests
from app.main import app
from app.models import AuditLog, Customer, Merchant, Transaction
from app.models.base import Base


KNOWN_CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
KNOWN_MERCHANT_ID = "22222222-2222-2222-2222-222222222222"

_ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"
_MODEL_PATH = _ARTIFACTS / "model.json"

pytestmark = pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason=(
        f"Real model artifact not at {_MODEL_PATH}. Run "
        "`uv run python -m ml.train` to enable Phase 3C integration tests."
    ),
)


def _payload(**overrides: Any) -> dict[str, Any]:
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


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    """TestClient WITH lifespan — loads the real SHAP explainer once."""
    reset_explainer_for_tests()
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        reset_explainer_for_tests()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_burst(db: Session, count: int = 3) -> None:
    """Insert `count` recent transactions to trigger velocity_burst."""
    from datetime import datetime, timedelta, timezone
    from decimal import Decimal
    anchor = datetime.now(timezone.utc)
    for i in range(count):
        db.add(Transaction(
            id=f"burst-seed-{i}",
            idempotency_key=f"key-burst-seed-{i}",
            customer_id=KNOWN_CUSTOMER_ID,
            merchant_id=KNOWN_MERCHANT_ID,
            amount=Decimal("5.00"),
            currency="USD",
            status="APPROVED",
            payment_method="CARD",
            country="US",
            is_card_present=True,
            created_at=anchor - timedelta(seconds=10 * (i + 1)),
        ))
    db.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_transaction_returns_approve(client: TestClient) -> None:
    """Small US card-present transaction with no recent history → APPROVE."""
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-clean-1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    # The seeded data + low amount should almost certainly be APPROVE on the
    # trained model, but we allow REVIEW to keep the test non-flaky if the
    # model is recalibrated later. DECLINE on this input would be a real
    # regression signal.
    assert body["decision"] != "DECLINE", (
        f"Trained model declined a clean US $100 card-present tx: {body}"
    )
    assert isinstance(body["fraud_score"], float)
    assert 0.0 <= body["fraud_score"] <= 1.0


def test_velocity_burst_triggers_hard_decline(
    client: TestClient,
    db_session: Session,
) -> None:
    _seed_burst(db_session, count=3)
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-burst-1"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["decision"] == "DECLINE"
    assert "velocity_burst" in body["rules_triggered"]
    assert body["fraud_score"] is None
    assert body["top_contributors"] == []


def test_high_amount_triggers_review(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(amount="6000.00"),
        headers={"Idempotency-Key": "k-high-amount"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # amount_ceiling fires → at minimum REVIEW; model may push to DECLINE.
    assert body["decision"] in {"REVIEW", "DECLINE"}
    assert "amount_ceiling" in body["rules_triggered"]


def test_idempotent_replay_returns_same_decision(client: TestClient) -> None:
    key = "k-replay-real"
    first = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": key},
    )
    second = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.headers.get("X-Idempotency-Replay") == "true"
    assert second.json()["decision"] == first.json()["decision"]
    assert second.json()["fraud_score"] == first.json()["fraud_score"]


def test_response_includes_top_contributors_for_clean_transaction(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-contribs"},
    )
    body = resp.json()
    if body["decision"] == "DECLINE":
        pytest.skip("Hard-block path returns empty contributors by design.")
    assert len(body["top_contributors"]) == 5
    for entry in body["top_contributors"]:
        assert "feature" in entry
        assert "shap_value" in entry
        assert "feature_value" in entry


def test_audit_log_written_on_post(
    client: TestClient,
    db_session: Session,
) -> None:
    resp = client.post(
        "/api/v1/transactions",
        json=_payload(),
        headers={"Idempotency-Key": "k-audit-1"},
    )
    assert resp.status_code == 201
    tx_id = resp.json()["transaction_id"]

    rows = list(
        db_session.execute(
            select(AuditLog).where(AuditLog.resource_id == tx_id)
        ).scalars().all()
    )
    assert len(rows) == 1
    assert rows[0].action.startswith("scored.")
    assert rows[0].actor.startswith("scorer:")
    assert rows[0].resource_type == "transaction"
