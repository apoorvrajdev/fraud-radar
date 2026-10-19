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
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.v1.transactions as transactions_module
import app.services.idempotency as idempotency_module
from app.db import get_db
from app.fraud.explainer import LocalExplanation
from app.main import app
from app.models import Customer, Merchant, Transaction
from app.models.base import Base
from app.models.idempotency_key import IdempotencyKey

KNOWN_CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
OTHER_CUSTOMER_ID = "44444444-4444-4444-4444-444444444444"
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
    db.add(Customer(
        id=OTHER_CUSTOMER_ID,
        email="other@example.com",
        full_name="Other Customer",
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
    import app.services.transaction_detail as detail_module
    stub = _StubExplainer()
    monkeypatch.setattr(scoring_module, "get_explainer", lambda: stub)
    # The Phase 3G detail builder also reads the live operating threshold
    # from `get_explainer()` — patch the symbol in that module too so the
    # detail envelope reports the stub's threshold instead of trying to
    # load the trained artifact.
    monkeypatch.setattr(detail_module, "get_explainer", lambda: stub)
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

    # GET now returns the Phase 3G TransactionDetail envelope; assert on
    # the fields that round-trip from the POST response.
    get_resp = client.get(f"/api/v1/transactions/{tx_id}")
    assert get_resp.status_code == 200, get_resp.text
    fetched = get_resp.json()

    assert fetched["id"] == tx_id
    assert fetched["fraud_decision"] == posted["decision"]
    assert fetched["fraud_decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    # `fraud_score` round-trips through Numeric(5, 4) so precision differs
    # by up to a half-ulp at the 4th decimal. approx handles it.
    assert float(fetched["fraud_score"]) == pytest.approx(
        posted["fraud_score"], abs=1e-4,
    )
    assert fetched["rules_triggered"] == posted["rules_triggered"]


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

    # The Phase 3G envelope renames `transaction_id` → `id` and exposes
    # the row fields directly. `effective_decision` falls through to
    # `fraud_decision` when no analyst has reviewed.
    assert fetched["id"] == tx_id
    assert fetched["fraud_decision"] == posted["decision"]
    assert fetched["effective_decision"] == posted["decision"]
    assert float(fetched["fraud_score"]) == pytest.approx(
        posted["fraud_score"], abs=1e-4,
    )
    # The detail envelope now surfaces the live operating threshold
    # from the explainer artifact (the stub fixture sets it to 0.7).
    assert fetched["threshold"] is not None
    assert fetched["rules_triggered"] == posted["rules_triggered"]
    # Top contributors round-trip with a computed `direction` field added.
    assert len(fetched["top_contributors"]) == len(posted["top_contributors"])
    for entry in fetched["top_contributors"]:
        assert entry["direction"] in {"fraud", "legit"}
    # No analyst has reviewed yet.
    assert fetched["analyst_label"] is None
    assert fetched["audit"]  # scoring writes at least one audit row


# ---------------------------------------------------------------------------
# Idempotency failure modes — expired keys and lost races must never 500
# ---------------------------------------------------------------------------


def _transaction_count(db_session: Session) -> int:
    return int(db_session.execute(select(func.count(Transaction.id))).scalar_one())


def test_post_returns_409_for_expired_key(
    client: TestClient, db_session: Session,
) -> None:
    """After the 24-hour TTL the key is neither replayed nor recycled.

    Without the expiry-aware resolve the lookup misses, the endpoint
    tries a fresh insert, and the `(customer_id, idempotency_key)`
    constraint turns the request into a 500.
    """
    key = "k-expired-1"
    first = client.post(
        "/api/v1/transactions", json=_payload(),
        headers={"Idempotency-Key": key},
    )
    assert first.status_code == 201, first.text

    entry = db_session.get(IdempotencyKey, key)
    assert entry is not None
    # Age the row past its TTL. `created_at` moves too: the table checks
    # `expires_at > created_at`.
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    entry.created_at = expired_at - timedelta(hours=24)
    entry.expires_at = expired_at
    db_session.commit()

    second = client.post(
        "/api/v1/transactions", json=_payload(),
        headers={"Idempotency-Key": key},
    )
    assert second.status_code == 409, second.text
    detail = second.json()["detail"].lower()
    assert "expired" in detail
    assert "new key" in detail
    assert first.json()["transaction_id"] in detail
    assert second.headers.get("X-Idempotency-Replay") is None
    assert _transaction_count(db_session) == 1


def _lose_the_race(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the endpoint's first `resolve` miss, as a racing request's would.

    The winner has already committed. The loser read the cache before
    that commit (stale miss), inserts, and trips the unique constraint;
    its recovery `resolve` is the real one and must find the winner.
    """
    real_resolve = transactions_module.resolve
    calls = {"n": 0}

    def stale_then_real(db: Session, *, key: str, request_hash: str) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real_resolve(db, key=key, request_hash=request_hash)

    monkeypatch.setattr(transactions_module, "resolve", stale_then_real)


def test_post_replays_after_losing_the_race_on_the_same_key(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same key, same body, same customer: the transactions constraint
    fires at flush; the loser rolls back and serves the winner's response."""
    key = "k-race-replay"
    winner = client.post(
        "/api/v1/transactions", json=_payload(),
        headers={"Idempotency-Key": key},
    )
    assert winner.status_code == 201, winner.text

    _lose_the_race(monkeypatch)
    loser = client.post(
        "/api/v1/transactions", json=_payload(),
        headers={"Idempotency-Key": key},
    )
    assert loser.status_code == 201, loser.text
    assert loser.headers.get("X-Idempotency-Replay") == "true"
    assert loser.json() == winner.json()
    assert _transaction_count(db_session) == 1


def test_post_returns_409_after_losing_the_race_with_a_different_body(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same key, different customer: the transaction row inserts fine and
    the `idempotency_keys` primary key fires at commit instead. The loser's
    scored row must be rolled back, not left orphaned without a cache entry."""
    key = "k-race-conflict"
    winner = client.post(
        "/api/v1/transactions", json=_payload(),
        headers={"Idempotency-Key": key},
    )
    assert winner.status_code == 201, winner.text

    _lose_the_race(monkeypatch)
    loser = client.post(
        "/api/v1/transactions",
        json=_payload(customer_id=OTHER_CUSTOMER_ID),
        headers={"Idempotency-Key": key},
    )
    assert loser.status_code == 409, loser.text
    assert "different payload" in loser.json()["detail"].lower()
    assert _transaction_count(db_session) == 1
    orphan = db_session.execute(
        select(Transaction).where(Transaction.customer_id == OTHER_CUSTOMER_ID)
    ).scalar_one_or_none()
    assert orphan is None


def test_post_propagates_integrity_errors_that_are_not_idempotency_collisions(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a collision on the key is recovered. Any other constraint
    failure (no row under that key afterwards) is re-raised unchanged
    rather than masked as a replay or a 409.

    A negative TTL makes the real `store` violate the table's own
    `expires_at > created_at` check at commit, so the handler runs
    against a session the database genuinely failed, not a synthetic
    exception raised before any statement."""
    monkeypatch.setattr(idempotency_module, "TTL", timedelta(hours=-1))
    with pytest.raises(IntegrityError, match="expires_after_created"):
        client.post(
            "/api/v1/transactions", json=_payload(),
            headers={"Idempotency-Key": "k-unrelated-integrity"},
        )
    assert _transaction_count(db_session) == 0
    assert db_session.get(IdempotencyKey, "k-unrelated-integrity") is None
