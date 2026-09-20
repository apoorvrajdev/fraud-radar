"""Phase 5F — FX enrichment through the real ingestion endpoint.

The unit suites prove the FX rules in isolation. This file proves the
property that actually matters in production: **the external dependency
cannot take the fraud pipeline down with it.** A transaction posted
while the FX provider is unreachable is still scored, still decided,
still persisted and still audited — it simply has no converted amount.

The FX service is injected through the app's dependency override, so no
test here opens a socket. The explainer is stubbed the same way the
other endpoint tests stub it, so no model artifact is needed either.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import get_db
from app.enrichment.fx import FxService, get_fx_service
from app.enrichment.provider import (
    FxProviderError,
    FxTimeoutError,
    FxUnsupportedCurrencyError,
    ProviderRate,
)
from app.fraud.explainer import LocalExplanation
from app.main import app
from app.models import AuditLog, Customer, Merchant, Transaction
from app.models.base import Base
from app.repositories.fx_rate import fx_rate_repository

KNOWN_CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
KNOWN_MERCHANT_ID = "22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubExplainer:
    """Low, deterministic score so a clean payload lands on APPROVE."""

    threshold = 0.5

    def explain_local(self, x_row: np.ndarray) -> LocalExplanation:
        return LocalExplanation(
            fraud_score=0.05,
            shap_values=np.linspace(-0.05, 0.05, 17),
            base_value=0.02,
        )


class _Provider:
    """Answers with a fixed rate, or raises a fixed failure."""

    def __init__(
        self, *, rate: str | None = "1.1022", raises: Exception | None = None
    ) -> None:
        self.rate = Decimal(rate) if rate is not None else None
        self.raises = raises
        self.calls: list[tuple[str, str, date]] = []

    def fetch_rate(
        self, *, base: str, quote: str, on: date
    ) -> ProviderRate | None:
        self.calls.append((base, quote, on))
        if self.raises is not None:
            raise self.raises
        if self.rate is None:
            return None
        return ProviderRate(base=base, quote=quote, rate_date=on, rate=self.rate)


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
def make_client(
    db_session: Session, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """Build a TestClient wired to a given FX provider.

    Returns a factory rather than a client, because each test needs its
    own provider behaviour. The FX service is supplied through the
    dependency override the router declares, which is exactly the seam
    that exists so production can reach the network and tests cannot.
    """
    app.dependency_overrides[get_db] = lambda: db_session

    import app.services.scoring as scoring_module
    import app.services.transaction_detail as detail_module

    stub = _StubExplainer()
    monkeypatch.setattr(scoring_module, "get_explainer", lambda: stub)
    monkeypatch.setattr(detail_module, "get_explainer", lambda: stub)

    clients: list[TestClient] = []

    def factory(provider: _Provider, **kwargs: object) -> TestClient:
        defaults: dict[str, object] = {
            "provider": provider,
            "base_currency": "USD",
            "cache_max_age_days": 7,
            "enabled": True,
        }
        defaults.update(kwargs)
        service = FxService(**defaults)  # type: ignore[arg-type]
        app.dependency_overrides[get_fx_service] = lambda: service
        client = TestClient(app, raise_server_exceptions=True)
        clients.append(client)
        return client

    try:
        yield factory
    finally:
        for client in clients:
            client.close()
        app.dependency_overrides.clear()


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


def _post(client: TestClient, key: str, **overrides: Any) -> Any:
    return client.post(
        "/api/v1/transactions",
        json=_payload(**overrides),
        headers={"Idempotency-Key": key},
    )


def _row(db: Session, tx_id: str) -> Transaction:
    db.expire_all()
    row = db.get(Transaction, tx_id)
    assert row is not None
    return row


def _scoring_payload(db: Session, tx_id: str) -> dict[str, Any]:
    entries = list(
        db.execute(
            select(AuditLog).where(AuditLog.resource_id == tx_id)
        ).scalars().all()
    )
    assert entries, "expected a scoring audit row"
    assert entries[-1].payload is not None
    parsed = json.loads(entries[-1].payload)
    assert isinstance(parsed, dict)
    return parsed


# ---------------------------------------------------------------------------
# The happy paths
# ---------------------------------------------------------------------------


def test_usd_ingestion_takes_the_identity_path(
    db_session: Session, make_client: Any
) -> None:
    """The default traffic shape: no lookup, deterministic 1:1."""
    provider = _Provider()
    client = make_client(provider)

    response = _post(client, "fx-usd-1")

    assert response.status_code == 201
    row = _row(db_session, response.json()["transaction_id"])
    assert row.amount == Decimal("100.0000")
    assert row.currency == "USD"
    assert row.amount_base == Decimal("100.0000")
    assert row.fx_rate == Decimal("1.00000000")
    assert row.fx_source == "identity"
    assert provider.calls == []


def test_non_usd_ingestion_stores_a_converted_amount(
    db_session: Session, make_client: Any
) -> None:
    provider = _Provider(rate="1.1022")
    client = make_client(provider)

    response = _post(client, "fx-eur-1", currency="EUR", amount="250.00")

    assert response.status_code == 201
    row = _row(db_session, response.json()["transaction_id"])
    assert row.amount == Decimal("250.0000")
    assert row.currency == "EUR"
    assert row.amount_base == Decimal("275.5500")
    assert row.fx_rate == Decimal("1.10220000")
    assert row.fx_source == "live"
    assert len(provider.calls) == 1


def test_the_fetched_rate_is_committed_with_the_transaction(
    db_session: Session, make_client: Any
) -> None:
    """The cache write rides the same commit as the row that caused it."""
    client = make_client(_Provider(rate="1.1022"))

    _post(client, "fx-eur-cache", currency="EUR")

    db_session.expire_all()
    cached = fx_rate_repository.latest_on_or_before(
        db_session,
        base="USD",
        quote="EUR",
        on=datetime.now(UTC).date(),
    )
    assert cached is not None
    assert cached.rate == Decimal("1.10220000")


def test_a_second_transaction_reuses_the_cached_rate(
    db_session: Session, make_client: Any
) -> None:
    provider = _Provider(rate="1.1022")
    client = make_client(provider)

    first = _post(client, "fx-eur-a", currency="EUR")
    second = _post(client, "fx-eur-b", currency="EUR", amount="50.00")

    assert first.status_code == second.status_code == 201
    assert len(provider.calls) == 1
    row = _row(db_session, second.json()["transaction_id"])
    assert row.fx_source == "cache"
    assert row.amount_base == Decimal("55.1100")


# ---------------------------------------------------------------------------
# The property this file exists for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(FxTimeoutError("timed out"), id="timeout"),
        pytest.param(FxProviderError("connection refused"), id="network-down"),
        pytest.param(RuntimeError("unforeseen"), id="unforeseen"),
    ],
)
def test_a_transaction_is_still_scored_when_fx_fails(
    db_session: Session, make_client: Any, failure: Exception
) -> None:
    """The external dependency cannot take the fraud pipeline with it."""
    client = make_client(_Provider(raises=failure))

    response = _post(client, "fx-down-1", currency="EUR", amount="250.00")

    assert response.status_code == 201
    body = response.json()
    assert body["decision"] == "APPROVE"
    assert body["fraud_score"] is not None
    assert body["threshold"] == 0.5

    row = _row(db_session, body["transaction_id"])
    assert row.fraud_decision == "APPROVE"
    assert row.status == "APPROVED"
    # The original values survive untouched...
    assert row.amount == Decimal("250.0000")
    assert row.currency == "EUR"
    # ...and no amount is invented for the one we could not convert.
    assert row.amount_base is None
    assert row.fx_rate is None
    assert row.fx_rate_date is None
    assert row.fx_source == "unavailable"


def test_the_audit_row_records_that_fx_was_unavailable(
    db_session: Session, make_client: Any
) -> None:
    """A decision taken without a converted amount says so in the trail."""
    client = make_client(_Provider(raises=FxTimeoutError("timed out")))

    response = _post(client, "fx-down-2", currency="EUR")

    payload = _scoring_payload(db_session, response.json()["transaction_id"])
    assert payload["fx_source"] == "unavailable"
    assert payload["decision"] == "APPROVE"


def test_the_audit_row_records_a_stale_rate(
    db_session: Session, make_client: Any
) -> None:
    """The case an operator most needs to see after the fact."""
    # Well before the server-stamped `created_at`, so it qualifies as an
    # older rate but is far outside the freshness window.
    stale_date = date(2020, 3, 9)
    fx_rate_repository.upsert(
        db_session,
        base="USD",
        quote="EUR",
        rate_date=stale_date,
        rate=Decimal("1.05"),
        fetched_at=datetime.now(UTC),
    )
    db_session.commit()
    client = make_client(_Provider(raises=FxProviderError("down")))

    response = _post(client, "fx-stale-1", currency="EUR", amount="100.00")

    body = response.json()
    row = _row(db_session, body["transaction_id"])
    assert row.fx_source == "stale"
    assert row.amount_base == Decimal("105.0000")

    payload = _scoring_payload(db_session, body["transaction_id"])
    assert payload["fx_source"] == "stale"
    assert payload["fx_rate_date"] == "2020-03-09"


def test_an_unsupported_currency_is_still_ingested(
    db_session: Session, make_client: Any
) -> None:
    """A currency the provider will not price is still a real transaction."""
    client = make_client(
        _Provider(raises=FxUnsupportedCurrencyError("no such pair"))
    )

    response = _post(client, "fx-zzz-1", currency="ZZZ", amount="77.00")

    assert response.status_code == 201
    row = _row(db_session, response.json()["transaction_id"])
    assert row.currency == "ZZZ"
    assert row.amount == Decimal("77.0000")
    assert row.amount_base is None
    assert row.fx_source == "unsupported"


def test_a_usd_transaction_is_unaffected_by_a_dead_provider(
    db_session: Session, make_client: Any
) -> None:
    """Backward compatibility: the existing traffic shape never calls out."""
    provider = _Provider(raises=FxProviderError("down"))
    client = make_client(provider)

    response = _post(client, "fx-usd-2")

    assert response.status_code == 201
    row = _row(db_session, response.json()["transaction_id"])
    assert row.amount_base == Decimal("100.0000")
    assert row.fx_source == "identity"
    assert provider.calls == []


def test_fx_disabled_still_ingests_and_scores(
    db_session: Session, make_client: Any
) -> None:
    provider = _Provider(rate="1.1022")
    client = make_client(provider, enabled=False)

    response = _post(client, "fx-off-1", currency="EUR")

    assert response.status_code == 201
    row = _row(db_session, response.json()["transaction_id"])
    assert row.fraud_decision == "APPROVE"
    assert row.fx_source == "unavailable"
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Interaction with idempotency
# ---------------------------------------------------------------------------


def test_a_replayed_request_does_not_refetch_the_rate(
    db_session: Session, make_client: Any
) -> None:
    """The replay returns the cached response before anything is enriched."""
    provider = _Provider(rate="1.1022")
    client = make_client(provider)

    first = _post(client, "fx-replay-1", currency="EUR")
    second = _post(client, "fx-replay-1", currency="EUR")

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.headers.get("X-Idempotency-Replay") == "true"
    assert len(provider.calls) == 1
