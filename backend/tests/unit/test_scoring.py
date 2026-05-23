"""Unit tests for the Phase 3C-2 scoring service.

Uses in-memory SQLite + StaticPool (same pattern as the explain endpoint
tests). The FraudExplainer is monkeypatched to a deterministic stub so
the tests don't require model artifacts on disk and can force specific
fraud scores to exercise every branch of the decision matrix.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.fraud.decision import Decision
from app.fraud.explainer import LocalExplanation
from app.fraud.rules import RuleResult, Severity
from app.models import AuditLog, Customer, Merchant, Transaction
from app.models.base import Base
from app.services.scoring import (
    ScoringResult,
    _compose_decision,
    score_transaction,
)


CUSTOMER_ID = "11111111-1111-1111-1111-111111111111"
MERCHANT_ID = "22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# Stub explainer — deterministic, no SHAP artifact dependency
# ---------------------------------------------------------------------------


class _StubExplainer:
    """Minimal FraudExplainer surface the scoring service touches."""

    def __init__(self, fraud_score: float = 0.1, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._fraud_score = fraud_score

    def explain_local(self, x_row: np.ndarray) -> LocalExplanation:
        return LocalExplanation(
            fraud_score=self._fraud_score,
            shap_values=np.linspace(-0.05, 0.05, 17),
            base_value=0.02,
        )


@pytest.fixture
def stub_explainer(monkeypatch: pytest.MonkeyPatch) -> _StubExplainer:
    """Default stub returning a low fraud score (model says APPROVE)."""
    stub = _StubExplainer(fraud_score=0.1, threshold=0.5)
    import app.services.scoring as scoring_module
    monkeypatch.setattr(scoring_module, "get_explainer", lambda: stub)
    return stub


@pytest.fixture
def stub_explainer_high(monkeypatch: pytest.MonkeyPatch) -> _StubExplainer:
    """Stub returning a high fraud score (model says DECLINE) for matrix testing."""
    stub = _StubExplainer(fraud_score=0.9, threshold=0.5)
    import app.services.scoring as scoring_module
    monkeypatch.setattr(scoring_module, "get_explainer", lambda: stub)
    return stub


# ---------------------------------------------------------------------------
# DB fixture
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
        id=CUSTOMER_ID,
        email="x@example.com",
        full_name="Test Customer",
        country="US",
        risk_tier="LOW",
        account_age_days=365,
    ))
    db.add(Merchant(
        id=MERCHANT_ID,
        name="Test Merchant",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tx(
    db: Session,
    *,
    amount: Decimal = Decimal("100.00"),
    country: str = "US",
    is_card_present: bool = True,
    created_at: datetime | None = None,
    tx_id: str | None = None,
    flush: bool = True,
) -> Transaction:
    tx = Transaction(
        id=tx_id or "current-tx-001",
        idempotency_key=f"key-{tx_id or 'current-tx-001'}",
        customer_id=CUSTOMER_ID,
        merchant_id=MERCHANT_ID,
        amount=amount,
        currency="USD",
        status="PENDING_REVIEW",
        payment_method="CARD",
        country=country,
        is_card_present=is_card_present,
        created_at=created_at or datetime.now(timezone.utc),
    )
    db.add(tx)
    if flush:
        db.flush()
    return tx


def _seed_recent_burst(db: Session, anchor: datetime, count: int = 3) -> None:
    """Seed `count` recent transactions in the 60 seconds before `anchor`."""
    for i in range(count):
        db.add(Transaction(
            id=f"burst-{i}",
            idempotency_key=f"key-burst-{i}",
            customer_id=CUSTOMER_ID,
            merchant_id=MERCHANT_ID,
            amount=Decimal("5.00"),
            currency="USD",
            status="APPROVED",
            payment_method="CARD",
            country="US",
            is_card_present=True,
            created_at=anchor - timedelta(seconds=10 * (i + 1)),
        ))
    db.flush()


def _audit_count(db: Session) -> int:
    return len(list(db.execute(select(AuditLog)).scalars().all()))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_transaction_returns_approve_with_low_score(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    tx = _make_tx(db_session)
    result = score_transaction(db_session, tx)
    assert isinstance(result, ScoringResult)
    assert result.decision == Decision.APPROVE
    assert result.fraud_score is not None
    assert 0.0 <= result.fraud_score < stub_explainer.threshold * 0.5
    assert result.rules_triggered == []
    assert len(result.top_contributors) == 5
    assert result.all_shap_values is not None and len(result.all_shap_values) == 17


def test_hard_block_rule_short_circuits_model(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    """velocity_burst fires → DECLINE; explain_local must NOT be called."""
    anchor = datetime.now(timezone.utc)
    _seed_recent_burst(db_session, anchor=anchor, count=3)
    tx = _make_tx(db_session, created_at=anchor)

    result = score_transaction(db_session, tx)
    assert result.decision == Decision.DECLINE
    assert result.fraud_score is None
    assert result.top_contributors == []
    assert result.all_shap_values is None
    assert "velocity_burst" in result.rules_triggered


def test_review_rule_with_approve_model_returns_review(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    """amount_ceiling fires + model says APPROVE → REVIEW."""
    tx = _make_tx(db_session, amount=Decimal("5001.00"))
    result = score_transaction(db_session, tx)
    assert result.decision == Decision.REVIEW
    assert "amount_ceiling" in result.rules_triggered


def test_review_rule_with_decline_model_returns_decline(
    db_session: Session,
    stub_explainer_high: _StubExplainer,
) -> None:
    """amount_ceiling fires + model says DECLINE → DECLINE (model wins, more conservative).

    Uses the high-score stub fixture because constructing real features
    that score above threshold is non-trivial in a unit test.
    """
    tx = _make_tx(db_session, amount=Decimal("5001.00"))
    result = score_transaction(db_session, tx)
    assert result.decision == Decision.DECLINE
    assert "amount_ceiling" in result.rules_triggered
    assert result.fraud_score is not None and result.fraud_score >= 0.5


def test_no_rules_no_model_signal_returns_approve(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    tx = _make_tx(db_session)
    result = score_transaction(db_session, tx)
    assert result.decision == Decision.APPROVE
    assert result.rules_triggered == []


def test_audit_log_written_on_hard_block(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    anchor = datetime.now(timezone.utc)
    _seed_recent_burst(db_session, anchor=anchor, count=3)
    before = _audit_count(db_session)
    tx = _make_tx(db_session, created_at=anchor)
    score_transaction(db_session, tx)
    after = _audit_count(db_session)
    assert after - before == 1

    entry = list(db_session.execute(select(AuditLog)).scalars().all())[-1]
    assert entry.action == "scored.hard_block"
    assert entry.actor.startswith("scorer:")  # may be "scorer:unknown" in CI
    assert entry.resource_type == "transaction"
    assert entry.resource_id == tx.id


def test_audit_log_written_on_normal_path(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    before = _audit_count(db_session)
    tx = _make_tx(db_session)
    score_transaction(db_session, tx)
    after = _audit_count(db_session)
    assert after - before == 1

    entry = list(db_session.execute(select(AuditLog)).scalars().all())[-1]
    assert entry.action == "scored.approve"
    assert entry.actor.startswith("scorer:")


def test_audit_log_skipped_when_write_audit_false(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    before = _audit_count(db_session)
    tx = _make_tx(db_session)
    score_transaction(db_session, tx, write_audit=False)
    after = _audit_count(db_session)
    assert after == before


@pytest.mark.parametrize(
    ("rules", "model_decision", "expected"),
    [
        # (1) Hard rule fires → DECLINE regardless of model
        (
            [RuleResult("velocity_burst", True, Severity.HARD_BLOCK, "burst")],
            Decision.APPROVE,
            Decision.DECLINE,
        ),
        # (2) REVIEW rule + model APPROVE → REVIEW
        (
            [RuleResult("amount_ceiling", True, Severity.REVIEW, "big")],
            Decision.APPROVE,
            Decision.REVIEW,
        ),
        # (3) REVIEW rule + model REVIEW → REVIEW
        (
            [RuleResult("amount_ceiling", True, Severity.REVIEW, "big")],
            Decision.REVIEW,
            Decision.REVIEW,
        ),
        # (4) REVIEW rule + model DECLINE → DECLINE (model wins)
        (
            [RuleResult("amount_ceiling", True, Severity.REVIEW, "big")],
            Decision.DECLINE,
            Decision.DECLINE,
        ),
        # (5) No rules → model decides
        (
            [RuleResult("amount_ceiling", False, Severity.REVIEW, None)],
            Decision.APPROVE,
            Decision.APPROVE,
        ),
    ],
)
def test_compose_decision_matrix_exhaustive(
    rules: list[RuleResult],
    model_decision: Decision,
    expected: Decision,
) -> None:
    final, _names = _compose_decision(rules, model_decision)
    assert final == expected


def test_score_transaction_does_not_commit_session(
    db_session: Session,
    stub_explainer: _StubExplainer,
) -> None:
    """The endpoint relies on the session staying uncommitted so it can
    bundle the Transaction, audit log, and idempotency cache into one
    atomic commit via `idempotency.store`."""
    tx = _make_tx(db_session)
    score_transaction(db_session, tx)
    # `new` carries pending inserts; if commit had happened, it'd be empty.
    # The audit log row written by the scorer is still pending.
    assert db_session.in_transaction(), "session must remain in an open transaction"
