"""Integration tests for GET /api/v1/alerts (Phase 3H)."""
from __future__ import annotations

import json
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

# Reference "now" used by every fixture and assertion. The service
# layer reads its clock via dependency injection in unit tests; the
# integration tests use the real clock and assert ranges instead.
FIXED_NOW = datetime(2026, 5, 27, 15, 0, 0, tzinfo=UTC)


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


def _add(
    db: Session,
    *,
    tx_id: str,
    score: str,
    decision: str = "REVIEW",
    analyst_label: str | None = None,
    country: str = "US",
    created_at: datetime | None = None,
    rules: list[str] | None = None,
) -> str:
    """Insert one transaction and return its id."""
    db.add(Transaction(
        id=tx_id,
        idempotency_key=f"key-{tx_id}",
        customer_id=CUSTOMER_ID,
        merchant_id=MERCHANT_ID,
        amount=Decimal("100.00"),
        currency="USD",
        status="PENDING_REVIEW" if decision == "REVIEW" else "APPROVED",
        payment_method="CARD",
        country=country,
        is_card_present=True,
        fraud_score=Decimal(score),
        fraud_decision=decision,
        analyst_label=analyst_label,
        rules_triggered=json.dumps(rules) if rules else None,
        created_at=created_at or (FIXED_NOW - timedelta(minutes=10)),
    ))
    db.commit()
    return tx_id


# ---------------------------------------------------------------------------
# Empty-queue behavior
# ---------------------------------------------------------------------------


def test_empty_queue_returns_zero_summary_and_empty_items(
    client: TestClient,
) -> None:
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"] == []
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert body["summary"]["pending_count"] == 0
    assert body["summary"]["oldest_pending_seconds"] is None
    assert body["summary"]["score_buckets"] == {"low": 0, "mid": 0, "high": 0}


# ---------------------------------------------------------------------------
# Queue predicate
# ---------------------------------------------------------------------------


def test_queue_excludes_non_review_decisions(
    client: TestClient, db_session: Session,
) -> None:
    _add(db_session, tx_id="r1", score="0.5", decision="REVIEW")
    _add(db_session, tx_id="a1", score="0.1", decision="APPROVE")
    _add(db_session, tx_id="d1", score="0.9", decision="DECLINE")
    _add(db_session, tx_id="p1", score="0.4", decision="PENDING")

    resp = client.get("/api/v1/alerts")
    body = resp.json()
    ids = [item["id"] for item in body["items"]]
    assert ids == ["r1"]
    assert body["summary"]["pending_count"] == 1


def test_queue_excludes_already_labeled_review_rows(
    client: TestClient, db_session: Session,
) -> None:
    _add(db_session, tx_id="unlabeled", score="0.5", decision="REVIEW")
    _add(
        db_session,
        tx_id="labeled",
        score="0.6",
        decision="REVIEW",
        analyst_label="CONFIRMED_FRAUD",
    )

    body = client.get("/api/v1/alerts").json()
    ids = [item["id"] for item in body["items"]]
    assert ids == ["unlabeled"]
    assert body["summary"]["pending_count"] == 1


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------


def test_sort_is_score_desc_then_created_at_asc_then_id_asc(
    client: TestClient, db_session: Session,
) -> None:
    # Equal scores → older created_at wins; equal ts → smaller id wins.
    older = FIXED_NOW - timedelta(hours=2)
    newer = FIXED_NOW - timedelta(hours=1)
    _add(db_session, tx_id="tx-mid-new", score="0.5", created_at=newer)
    _add(db_session, tx_id="tx-mid-old", score="0.5", created_at=older)
    _add(db_session, tx_id="tx-high", score="0.7", created_at=newer)
    _add(db_session, tx_id="tx-low-b", score="0.1", created_at=older)
    _add(db_session, tx_id="tx-low-a", score="0.1", created_at=older)

    body = client.get("/api/v1/alerts").json()
    ids = [item["id"] for item in body["items"]]
    assert ids == [
        "tx-high",       # highest score
        "tx-mid-old",    # tied score, older
        "tx-mid-new",    # tied score, newer
        "tx-low-a",      # tied score+ts, id "a" < "b"
        "tx-low-b",
    ]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_round_trips_with_no_duplicates(
    client: TestClient, db_session: Session,
) -> None:
    # Five rows with distinct scores so the cursor is unambiguous.
    for i, score in enumerate(["0.9", "0.7", "0.5", "0.3", "0.1"]):
        _add(db_session, tx_id=f"tx-{i}", score=score)

    page1 = client.get("/api/v1/alerts", params={"limit": 2}).json()
    assert [i["id"] for i in page1["items"]] == ["tx-0", "tx-1"]
    assert page1["has_more"] is True

    page2 = client.get(
        "/api/v1/alerts",
        params={"limit": 2, "cursor": page1["next_cursor"]},
    ).json()
    assert [i["id"] for i in page2["items"]] == ["tx-2", "tx-3"]
    assert page2["has_more"] is True

    page3 = client.get(
        "/api/v1/alerts",
        params={"limit": 2, "cursor": page2["next_cursor"]},
    ).json()
    assert [i["id"] for i in page3["items"]] == ["tx-4"]
    assert page3["has_more"] is False
    assert page3["next_cursor"] is None


def test_malformed_cursor_returns_422(client: TestClient) -> None:
    resp = client.get("/api/v1/alerts", params={"cursor": "not-a-cursor"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_min_score_filter_constrains_items_but_not_summary(
    client: TestClient, db_session: Session,
) -> None:
    _add(db_session, tx_id="lo", score="0.1")
    _add(db_session, tx_id="md", score="0.3")
    _add(db_session, tx_id="hi", score="0.6")

    body = client.get("/api/v1/alerts", params={"min_score": "0.5"}).json()
    assert [i["id"] for i in body["items"]] == ["hi"]
    # Summary ignores the filter — it reports queue-wide health.
    assert body["summary"]["pending_count"] == 3


def test_country_filter(client: TestClient, db_session: Session) -> None:
    _add(db_session, tx_id="us1", score="0.5", country="US")
    _add(db_session, tx_id="gb1", score="0.5", country="GB")

    body = client.get("/api/v1/alerts", params={"country": "GB"}).json()
    assert [i["id"] for i in body["items"]] == ["gb1"]
    assert body["summary"]["pending_count"] == 2


def test_age_filters_respect_min_and_max(
    client: TestClient, db_session: Session,
) -> None:
    now = datetime.now(UTC)
    _add(db_session, tx_id="fresh", score="0.5", created_at=now - timedelta(minutes=1))
    _add(db_session, tx_id="hour", score="0.5", created_at=now - timedelta(hours=1))
    _add(db_session, tx_id="day", score="0.5", created_at=now - timedelta(days=1))

    # Older than 30 minutes — drops "fresh".
    body = client.get(
        "/api/v1/alerts", params={"min_age_seconds": 1800},
    ).json()
    ids = {i["id"] for i in body["items"]}
    assert ids == {"hour", "day"}

    # In the last 2 hours — drops "day".
    body = client.get(
        "/api/v1/alerts", params={"max_age_seconds": 7200},
    ).json()
    ids = {i["id"] for i in body["items"]}
    assert ids == {"fresh", "hour"}


def test_age_filter_cross_validation_returns_422(client: TestClient) -> None:
    resp = client.get(
        "/api/v1/alerts",
        params={"min_age_seconds": 100, "max_age_seconds": 50},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Per-row hydration
# ---------------------------------------------------------------------------


def test_age_seconds_is_non_negative_and_monotonic_with_created_at(
    client: TestClient, db_session: Session,
) -> None:
    now = datetime.now(UTC)
    _add(db_session, tx_id="old", score="0.5", created_at=now - timedelta(hours=2))
    _add(db_session, tx_id="new", score="0.5", created_at=now - timedelta(minutes=1))

    body = client.get("/api/v1/alerts").json()
    by_id = {i["id"]: i for i in body["items"]}
    assert by_id["old"]["age_seconds"] > by_id["new"]["age_seconds"] > 0


def test_rules_triggered_round_trips_from_persisted_json(
    client: TestClient, db_session: Session,
) -> None:
    _add(
        db_session,
        tx_id="rules",
        score="0.5",
        rules=["high_amount", "off_hours"],
    )

    body = client.get("/api/v1/alerts").json()
    assert body["items"][0]["rules_triggered"] == ["high_amount", "off_hours"]


def test_rules_triggered_missing_collapses_to_empty_list(
    client: TestClient, db_session: Session,
) -> None:
    _add(db_session, tx_id="no-rules", score="0.5", rules=None)

    body = client.get("/api/v1/alerts").json()
    assert body["items"][0]["rules_triggered"] == []


# ---------------------------------------------------------------------------
# Summary math
# ---------------------------------------------------------------------------


def test_summary_bucket_boundaries_are_low_lt_0_20_mid_lt_0_50_else_high(
    client: TestClient, db_session: Session,
) -> None:
    # Boundary cases land in the documented bucket.
    _add(db_session, tx_id="lo-edge", score="0.1999")
    _add(db_session, tx_id="mid-low-edge", score="0.20")
    _add(db_session, tx_id="mid-mid", score="0.35")
    _add(db_session, tx_id="mid-high-edge", score="0.4999")
    _add(db_session, tx_id="hi-edge", score="0.50")
    _add(db_session, tx_id="hi-high", score="0.68")

    body = client.get("/api/v1/alerts").json()
    assert body["summary"]["score_buckets"] == {
        "low": 1,
        "mid": 3,
        "high": 2,
    }
    # Buckets sum exactly to pending_count — no overflow category.
    assert sum(body["summary"]["score_buckets"].values()) == (
        body["summary"]["pending_count"]
    )


def test_summary_oldest_matches_oldest_pending_row(
    client: TestClient, db_session: Session,
) -> None:
    now = datetime.now(UTC)
    _add(db_session, tx_id="newer", score="0.5", created_at=now - timedelta(minutes=5))
    _add(db_session, tx_id="oldest", score="0.5", created_at=now - timedelta(hours=3))
    _add(db_session, tx_id="middle", score="0.5", created_at=now - timedelta(hours=1))

    body = client.get("/api/v1/alerts").json()
    oldest = body["summary"]["oldest_pending_seconds"]
    assert oldest is not None
    # Should be close to 3 hours in seconds, allowing for clock drift.
    assert 3 * 3600 - 10 <= oldest <= 3 * 3600 + 60
