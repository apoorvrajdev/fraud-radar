"""Phase 5F — the FX fields as the API actually serves them.

Three read surfaces carry FX, and each one has a different job:

* `GET /transactions` and `GET /transactions/{id}` present the original
  amount and the derived one **side by side**, so a client can never
  mistake one for the other and can always tell why a converted figure
  is missing.
* `GET /stats/*` aggregates in the reporting currency, because summing
  a raw amount column across mixed currencies adds euros to yen.

The backward-compatibility assertions are as important as the new ones:
a USD-only deployment — which is every row this system has served —
must see exactly the numbers it saw before.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

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
from app.models import Customer, Merchant, Transaction
from app.models.base import Base

CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"


class _StubExplainer:
    threshold = 0.5

    def explain_local(self, x_row: np.ndarray) -> LocalExplanation:
        return LocalExplanation(
            fraud_score=0.05,
            shap_values=np.linspace(-0.05, 0.05, 17),
            base_value=0.02,
        )


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
        id=CUSTOMER_ID,
        email="seed@example.com",
        full_name="Seed Customer",
        country="US",
        risk_tier="LOW",
        account_age_days=365,
    ))
    db.add(Merchant(
        id=MERCHANT_ID,
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
def client(
    db_session: Session, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    import app.services.transaction_detail as detail_module

    # No lifespan — nothing under test here loads a model, and the
    # detail builder's threshold read is patched out. Constructing the
    # client without the context manager keeps this file runnable with
    # no artifacts on disk, as the other read-path suites are.
    monkeypatch.setattr(detail_module, "get_explainer", lambda: _StubExplainer())
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _add_tx(
    db: Session,
    *,
    tx_id: str,
    amount: str,
    currency: str = "USD",
    amount_base: str | None = None,
    fx_rate: str | None = None,
    fx_rate_date: date | None = None,
    fx_source: str | None = None,
    decision: Decision = Decision.APPROVE,
    country: str = "US",
    created_at: datetime | None = None,
) -> Transaction:
    tx = Transaction(
        id=tx_id,
        idempotency_key=f"key-{tx_id}",
        customer_id=CUSTOMER_ID,
        merchant_id=MERCHANT_ID,
        amount=Decimal(amount),
        currency=currency,
        amount_base=Decimal(amount_base) if amount_base is not None else None,
        fx_rate=Decimal(fx_rate) if fx_rate is not None else None,
        fx_rate_date=fx_rate_date,
        fx_source=fx_source,
        status="APPROVED",
        payment_method="CARD",
        country=country,
        is_card_present=True,
        fraud_score=Decimal("0.0500"),
        fraud_decision=decision.value,
        created_at=created_at or datetime.now(UTC) - timedelta(minutes=5),
    )
    db.add(tx)
    db.commit()
    return tx


# ---------------------------------------------------------------------------
# Transaction list
# ---------------------------------------------------------------------------


def test_list_presents_original_and_derived_amounts_side_by_side(
    db_session: Session, client: TestClient
) -> None:
    _add_tx(
        db_session,
        tx_id="tx-eur",
        amount="250.00",
        currency="EUR",
        amount_base="275.5500",
        fx_rate="1.10220000",
        fx_rate_date=date(2024, 1, 2),
        fx_source="live",
    )

    body = client.get("/api/v1/transactions").json()

    item = body["items"][0]
    # What the cardholder was charged...
    assert item["amount"] == "250.0000"
    assert item["currency"] == "EUR"
    # ...and the derived reporting figure, clearly distinct.
    assert item["amount_base"] == "275.5500"
    assert item["fx_rate"] == "1.10220000"
    assert item["fx_rate_date"] == "2024-01-02"
    assert item["fx_source"] == "live"


def test_list_renders_a_row_that_could_not_be_converted(
    db_session: Session, client: TestClient
) -> None:
    """The row is present and honest, not hidden and not guessed at."""
    _add_tx(
        db_session,
        tx_id="tx-noconv",
        amount="250.00",
        currency="EUR",
        fx_source="unavailable",
    )

    item = client.get("/api/v1/transactions").json()["items"][0]

    assert item["amount"] == "250.0000"
    assert item["currency"] == "EUR"
    assert item["amount_base"] is None
    assert item["fx_rate"] is None
    assert item["fx_rate_date"] is None
    assert item["fx_source"] == "unavailable"


def test_list_of_a_usd_row_reports_the_identity_conversion(
    db_session: Session, client: TestClient
) -> None:
    _add_tx(
        db_session,
        tx_id="tx-usd",
        amount="100.00",
        amount_base="100.0000",
        fx_rate="1.00000000",
        fx_source="identity",
    )

    item = client.get("/api/v1/transactions").json()["items"][0]

    assert item["amount"] == item["amount_base"] == "100.0000"
    assert item["fx_source"] == "identity"


def test_a_row_written_before_enrichment_still_serialises(
    db_session: Session, client: TestClient
) -> None:
    """Backward compatibility: pre-5F rows have four null FX columns."""
    _add_tx(db_session, tx_id="tx-legacy", amount="100.00")

    item = client.get("/api/v1/transactions").json()["items"][0]

    assert item["amount"] == "100.0000"
    assert item["amount_base"] is None
    assert item["fx_source"] is None


# ---------------------------------------------------------------------------
# Transaction detail
# ---------------------------------------------------------------------------


def test_detail_carries_the_fx_fields(
    db_session: Session, client: TestClient
) -> None:
    _add_tx(
        db_session,
        tx_id="tx-detail",
        amount="250.00",
        currency="EUR",
        amount_base="275.5500",
        fx_rate="1.10220000",
        fx_rate_date=date(2024, 1, 2),
        fx_source="stale",
    )

    body = client.get("/api/v1/transactions/tx-detail").json()

    assert body["amount"] == "250.0000"
    assert body["currency"] == "EUR"
    assert body["amount_base"] == "275.5500"
    assert body["fx_rate_date"] == "2024-01-02"
    # The provenance a reviewer needs to judge the converted figure.
    assert body["fx_source"] == "stale"


def test_detail_reads_the_row_rather_than_re_pricing_it(
    db_session: Session, client: TestClient
) -> None:
    """A stored rate from years ago is served as stored.

    The detail page backs an audit trail: it must show the rate the
    decision was taken with, not what the provider would say today.
    """
    _add_tx(
        db_session,
        tx_id="tx-old",
        amount="100.00",
        currency="EUR",
        amount_base="105.0000",
        fx_rate="1.05000000",
        fx_rate_date=date(2020, 3, 9),
        fx_source="stale",
    )

    body = client.get("/api/v1/transactions/tx-old").json()

    assert body["fx_rate"] == "1.05000000"
    assert body["fx_rate_date"] == "2020-03-09"


def test_detail_of_an_unconverted_row_is_all_nulls(
    db_session: Session, client: TestClient
) -> None:
    _add_tx(
        db_session,
        tx_id="tx-detail-null",
        amount="250.00",
        currency="ZZZ",
        fx_source="unsupported",
    )

    body = client.get("/api/v1/transactions/tx-detail-null").json()

    assert body["amount_base"] is None
    assert body["fx_rate"] is None
    assert body["fx_source"] == "unsupported"


# ---------------------------------------------------------------------------
# Stats — aggregation in the reporting currency
# ---------------------------------------------------------------------------


def test_overview_sums_the_converted_amount(
    db_session: Session, client: TestClient
) -> None:
    """Two declines: one EUR converted to 275.55, one USD at 100."""
    _add_tx(
        db_session,
        tx_id="tx-s1",
        amount="250.00",
        currency="EUR",
        amount_base="275.5500",
        fx_rate="1.10220000",
        fx_source="live",
        decision=Decision.DECLINE,
    )
    _add_tx(
        db_session,
        tx_id="tx-s2",
        amount="100.00",
        amount_base="100.0000",
        fx_rate="1.00000000",
        fx_source="identity",
        decision=Decision.DECLINE,
    )

    body = client.get("/api/v1/stats/overview").json()

    # 275.55 + 100.00 — not 250 + 100, which would have added euros to
    # dollars and reported 350.
    assert body["fraud_caught_amount"] == "375.55"


def test_overview_falls_back_to_the_original_amount_when_unconverted(
    db_session: Session, client: TestClient
) -> None:
    """An unconvertible row still counts toward the volume it contributed."""
    _add_tx(
        db_session,
        tx_id="tx-s3",
        amount="250.00",
        currency="EUR",
        fx_source="unavailable",
        decision=Decision.DECLINE,
    )

    body = client.get("/api/v1/stats/overview").json()

    assert body["fraud_caught_amount"] == "250.00"


def test_usd_only_traffic_reports_exactly_what_it_did_before(
    db_session: Session, client: TestClient
) -> None:
    """The backward-compatibility guarantee, stated as a test.

    On identity-converted rows the two COALESCE branches are equal, so
    the aggregate is unchanged by Phase 5F.
    """
    for i, amount in enumerate(("100.00", "250.50", "99.49")):
        _add_tx(
            db_session,
            tx_id=f"tx-usd-{i}",
            amount=amount,
            amount_base=amount,
            fx_rate="1.00000000",
            fx_source="identity",
            decision=Decision.DECLINE,
        )

    body = client.get("/api/v1/stats/overview").json()

    assert body["fraud_caught_amount"] == "449.99"


def test_breakdown_sums_the_converted_amount(
    db_session: Session, client: TestClient
) -> None:
    _add_tx(
        db_session,
        tx_id="tx-b1",
        amount="250.00",
        currency="EUR",
        amount_base="275.5500",
        fx_rate="1.10220000",
        fx_source="live",
        country="DE",
    )
    _add_tx(
        db_session,
        tx_id="tx-b2",
        amount="100.00",
        amount_base="100.0000",
        fx_rate="1.00000000",
        fx_source="identity",
        country="US",
    )

    items = client.get("/api/v1/stats/breakdown").json()["items"]

    by_country = {item["category"]: item["total_amount"] for item in items}
    assert by_country["DE"] == "275.55"
    assert by_country["US"] == "100.00"


def test_breakdown_falls_back_for_an_unconverted_row(
    db_session: Session, client: TestClient
) -> None:
    _add_tx(
        db_session,
        tx_id="tx-b3",
        amount="250.00",
        currency="EUR",
        fx_source="unavailable",
        country="DE",
    )

    items = client.get("/api/v1/stats/breakdown").json()["items"]

    assert items[0]["total_amount"] == "250.00"


def test_amount_filters_still_filter_on_the_original_amount(
    db_session: Session, client: TestClient
) -> None:
    """A deliberate asymmetry with the aggregates, documented in the contract.

    Aggregates are corrected because summing mixed currencies is wrong.
    Filters are not, because "show me transactions over 250" is a
    question about what the cardholder was charged.
    """
    _add_tx(
        db_session,
        tx_id="tx-f1",
        amount="250.00",
        currency="EUR",
        amount_base="275.5500",
        fx_rate="1.10220000",
        fx_source="live",
    )

    over_260 = client.get(
        "/api/v1/transactions", params={"min_amount": "260"}
    ).json()
    over_240 = client.get(
        "/api/v1/transactions", params={"min_amount": "240"}
    ).json()

    # 250 EUR is not matched by min_amount=260 even though its converted
    # figure is 275.55.
    assert over_260["items"] == []
    assert [item["id"] for item in over_240["items"]] == ["tx-f1"]


def test_timeseries_is_unaffected_by_fx(
    db_session: Session, client: TestClient
) -> None:
    """It counts rows and fraud rates — there is no money in it to convert."""
    _add_tx(
        db_session,
        tx_id="tx-t1",
        amount="250.00",
        currency="EUR",
        amount_base="275.5500",
        fx_rate="1.10220000",
        fx_source="live",
    )

    body = client.get("/api/v1/stats/timeseries").json()

    assert sum(point["transaction_count"] for point in body["points"]) == 1


def test_stats_tolerate_a_mix_of_enriched_and_legacy_rows(
    db_session: Session, client: TestClient
) -> None:
    """The realistic state of a database mid-migration."""
    _add_tx(db_session, tx_id="tx-m1", amount="100.00", decision=Decision.DECLINE)
    _add_tx(
        db_session,
        tx_id="tx-m2",
        amount="50.00",
        currency="EUR",
        amount_base="55.1100",
        fx_rate="1.10220000",
        fx_source="live",
        decision=Decision.DECLINE,
    )

    body = client.get("/api/v1/stats/overview").json()

    assert body["fraud_caught_amount"] == "155.11"
    assert body["total_transactions_24h"] == 2
