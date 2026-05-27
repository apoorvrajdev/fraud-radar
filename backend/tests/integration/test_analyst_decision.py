"""Integration tests for the Phase 3G analyst override endpoint.

Covers POST /api/v1/transactions/{id}/decision and the
TransactionDetail envelope returned by GET /api/v1/transactions/{id}
in the Phase 3G shape.

Same in-memory SQLite + stub explainer pattern as
``test_transactions_endpoint.py`` — the stub returns a low fraud
score so a clean payload classifies as APPROVE. REVIEW rows in this
suite are forced by triggering a REVIEW-severity rule directly via
db fixtures (no need to coerce the model).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.fraud.decision import Decision
from app.fraud.explainer import LocalExplanation
from app.main import app
from app.models import Customer, Merchant
from app.models.base import Base
from app.models.transaction import Transaction

KNOWN_CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
KNOWN_MERCHANT_ID = "22222222-2222-2222-2222-222222222222"


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
        id=KNOWN_CUSTOMER_ID, email="a@example.com", full_name="A",
        country="US", risk_tier="LOW", account_age_days=100,
    ))
    db.add(Merchant(
        id=KNOWN_MERCHANT_ID, name="M", category="RETAIL", mcc="5311",
        country="US", risk_rating="LOW",
    ))
    db.commit()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


class _StubExplainer:
    threshold = 0.7

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
    app.dependency_overrides[get_db] = lambda: db_session
    import app.services.scoring as scoring_module
    import app.services.transaction_detail as detail_module
    stub = _StubExplainer()
    monkeypatch.setattr(scoring_module, "get_explainer", lambda: stub)
    monkeypatch.setattr(detail_module, "get_explainer", lambda: stub)
    try:
        yield TestClient(app, raise_server_exceptions=True)
    finally:
        app.dependency_overrides.clear()


def _seed_tx(
    db: Session,
    *,
    fraud_decision: str | None = Decision.REVIEW.value,
    analyst_label: str | None = None,
    top_features: list[dict[str, Any]] | None = None,
    rules_triggered: list[str] | None = None,
) -> str:
    """Insert a Transaction directly so tests can target any decision state."""
    tx_id = str(uuid4())
    if top_features is None:
        top_features = [
            {"feature": "amount", "feature_value": 9421.0, "shap_value": 1.83},
            {"feature": "geo_velocity_kmh", "feature_value": 0.0, "shap_value": -0.42},
            {"feature": "off_hours_flag", "feature_value": 1.0, "shap_value": 0.0},
        ]
    if rules_triggered is None:
        rules_triggered = ["high_amount", "off_hours"]

    db.add(Transaction(
        id=tx_id,
        idempotency_key=f"seed-{tx_id[:8]}",
        customer_id=KNOWN_CUSTOMER_ID,
        merchant_id=KNOWN_MERCHANT_ID,
        amount=Decimal("100.00"),
        currency="USD",
        status="PENDING_REVIEW" if fraud_decision == "REVIEW" else "APPROVED",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        fraud_score=Decimal("0.4217"),
        fraud_decision=fraud_decision,
        rules_triggered=json.dumps(rules_triggered),
        top_features=json.dumps(top_features),
        analyst_label=analyst_label,
        created_at=datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC),
    ))
    db.commit()
    return tx_id


# ---------------------------------------------------------------------------
# GET /transactions/{id} — Phase 3G detail envelope shape
# ---------------------------------------------------------------------------


def test_get_detail_envelope_shape(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)

    resp = client.get(f"/api/v1/transactions/{tx_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Row fields
    assert body["id"] == tx_id
    assert body["customer_id"] == KNOWN_CUSTOMER_ID
    assert body["merchant_id"] == KNOWN_MERCHANT_ID
    assert body["country"] == "US"
    assert body["is_card_present"] is True

    # Scoring fields
    assert body["fraud_decision"] == "REVIEW"
    assert body["effective_decision"] == "REVIEW"
    assert body["rules_triggered"] == ["high_amount", "off_hours"]
    assert body["threshold"] == pytest.approx(0.7)

    # Contributors carry a computed `direction` field
    directions = {c["direction"] for c in body["top_contributors"]}
    assert directions <= {"fraud", "legit"}
    # shap > 0 → fraud, shap <= 0 → legit
    by_feature = {c["feature"]: c for c in body["top_contributors"]}
    assert by_feature["amount"]["direction"] == "fraud"
    assert by_feature["geo_velocity_kmh"]["direction"] == "legit"
    # Zero contribution ties toward legit (documented tie-break).
    assert by_feature["off_hours_flag"]["direction"] == "legit"

    # Analyst fields default to null on a fresh row.
    assert body["analyst_label"] is None
    assert body["analyst_notes"] is None
    assert body["reviewed_at"] is None

    # Audit is an empty list for rows that were not scored via the
    # scoring service (this seed bypasses it).
    assert body["audit"] == []


def test_get_unknown_transaction_returns_404(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/transactions/99999999-9999-9999-9999-999999999999",
    )
    assert resp.status_code == 404


def test_effective_decision_flips_with_confirmed_fraud_label(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(
        db_session,
        fraud_decision=Decision.REVIEW.value,
        analyst_label="CONFIRMED_FRAUD",
    )
    body = client.get(f"/api/v1/transactions/{tx_id}").json()
    # Original ML verdict preserved …
    assert body["fraud_decision"] == "REVIEW"
    # … but effective decision reflects the analyst's call.
    assert body["effective_decision"] == "DECLINE"


def test_effective_decision_flips_with_confirmed_legit_label(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(
        db_session,
        fraud_decision=Decision.REVIEW.value,
        analyst_label="CONFIRMED_LEGIT",
    )
    body = client.get(f"/api/v1/transactions/{tx_id}").json()
    assert body["fraud_decision"] == "REVIEW"
    assert body["effective_decision"] == "APPROVE"


# ---------------------------------------------------------------------------
# POST /transactions/{id}/decision — happy paths
# ---------------------------------------------------------------------------


def test_post_decision_confirms_fraud_on_review_row(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)

    resp = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_FRAUD", "notes": "stolen-card report 4421"},
        headers={"X-Analyst-Id": "analyst-1"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["analyst_label"] == "CONFIRMED_FRAUD"
    assert body["analyst_notes"] == "stolen-card report 4421"
    assert body["reviewed_at"] is not None
    assert body["effective_decision"] == "DECLINE"
    # Original model verdict preserved verbatim.
    assert body["fraud_decision"] == "REVIEW"

    # Exactly one audit row, action=ANALYST_DECISION, actor=analyst-1.
    audit = body["audit"]
    assert len(audit) == 1
    assert audit[0]["actor"] == "analyst-1"
    assert audit[0]["action"] == "ANALYST_DECISION"
    assert audit[0]["payload"]["label"] == "CONFIRMED_FRAUD"
    assert audit[0]["payload"]["prev_label"] is None


def test_post_decision_idempotent_on_identical_resubmit(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)
    body = {"label": "CONFIRMED_LEGIT", "notes": "false alarm"}
    headers = {"X-Analyst-Id": "analyst-2"}

    first = client.post(
        f"/api/v1/transactions/{tx_id}/decision", json=body, headers=headers,
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/v1/transactions/{tx_id}/decision", json=body, headers=headers,
    )
    assert second.status_code == 200
    # Audit length unchanged — identical resubmit must not append.
    assert len(second.json()["audit"]) == 1


def test_post_decision_revision_writes_revised_audit_row(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)
    headers = {"X-Analyst-Id": "analyst-3"}

    first = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_LEGIT", "notes": "looks fine"},
        headers=headers,
    )
    assert first.status_code == 200

    second = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_FRAUD", "notes": "second look says no"},
        headers=headers,
    )
    assert second.status_code == 200
    body = second.json()

    # Final state reflects the revision.
    assert body["analyst_label"] == "CONFIRMED_FRAUD"
    assert body["effective_decision"] == "DECLINE"

    # Audit history has two rows: the original ANALYST_DECISION plus
    # a new ANALYST_DECISION_REVISED entry. Newest first.
    actions = [row["action"] for row in body["audit"]]
    assert actions == ["ANALYST_DECISION_REVISED", "ANALYST_DECISION"]
    assert body["audit"][0]["payload"]["prev_label"] == "CONFIRMED_LEGIT"


# ---------------------------------------------------------------------------
# POST /transactions/{id}/decision — error paths
# ---------------------------------------------------------------------------


def test_post_decision_returns_404_for_unknown_id(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/transactions/99999999-9999-9999-9999-999999999999/decision",
        json={"label": "CONFIRMED_FRAUD"},
        headers={"X-Analyst-Id": "analyst-x"},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "fraud_decision", ["APPROVE", "DECLINE", "PENDING"],
)
def test_post_decision_returns_409_for_non_review_rows(
    client: TestClient, db_session: Session, fraud_decision: str,
) -> None:
    tx_id = _seed_tx(db_session, fraud_decision=fraud_decision)
    resp = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_FRAUD"},
        headers={"X-Analyst-Id": "analyst-x"},
    )
    assert resp.status_code == 409
    assert "REVIEW" in resp.json()["detail"]


def test_post_decision_returns_422_when_analyst_header_missing(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)
    resp = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_FRAUD"},
    )
    assert resp.status_code == 422


def test_post_decision_returns_422_for_invalid_label(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)
    resp = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "NOT_A_LABEL"},
        headers={"X-Analyst-Id": "analyst-x"},
    )
    assert resp.status_code == 422


def test_post_decision_returns_422_for_oversize_notes(
    client: TestClient, db_session: Session,
) -> None:
    tx_id = _seed_tx(db_session)
    resp = client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_FRAUD", "notes": "x" * 2001},
        headers={"X-Analyst-Id": "analyst-x"},
    )
    assert resp.status_code == 422


def test_post_decision_preserves_original_fraud_decision(
    client: TestClient, db_session: Session,
) -> None:
    """The model's verdict is preserved verbatim across every override."""
    tx_id = _seed_tx(db_session)
    client.post(
        f"/api/v1/transactions/{tx_id}/decision",
        json={"label": "CONFIRMED_FRAUD"},
        headers={"X-Analyst-Id": "analyst-x"},
    )

    db_session.expire_all()
    tx = db_session.get(Transaction, tx_id)
    assert tx is not None
    assert tx.fraud_decision == "REVIEW"  # untouched
    assert tx.analyst_label == "CONFIRMED_FRAUD"
