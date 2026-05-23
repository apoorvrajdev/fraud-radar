"""Phase 3C-2 — End-to-end scoring service.

Orchestrates rules + XGBoost + SHAP + decision composition + audit log
for one transaction. Called by the POST /transactions endpoint and by
the latency benchmark.

The service does NOT commit the session. The caller (endpoint or
benchmark) is responsible for the commit — typically as part of the
idempotency cache write, so both the new Transaction row and its audit
log entry land atomically.

See `docs/adr/PHASE_3_DESIGN.md` for the architecture and the
conservative-wins decision matrix; `docs/adr/PHASE_3C_INTEGRATION.md`
for the six implementation decisions this module enacts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.fraud import FEATURE_NAMES, FeatureExtractor
from app.fraud.decision import Decision
from app.fraud.explainer import get_explainer, top_contributors
from app.fraud.rules import RuleResult, Severity, evaluate_all
from app.fraud.transaction_context import TransactionContext
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from app.repositories.audit import audit_repository


# 180 days = the longest rule lookback window (rule_dormant_account_high_value).
# One context load covers every rule (ADR decision #5).
_RECENT_HISTORY_WINDOW = timedelta(days=180)

_extractor = FeatureExtractor()  # stateless, safe to share


def _load_model_version() -> str:
    """Read `trained_at_utc` from training_metadata.json once at module load.

    Falls back to "unknown" if the file is missing or malformed — test
    environments without trained artifacts (e.g. CI bootstrapping) should
    not crash on import.
    """
    try:
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "ml" / "artifacts" / "training_metadata.json"
        )
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("trained_at_utc", "unknown"))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
        return "unknown"


_MODEL_VERSION = _load_model_version()


@dataclass(frozen=True)
class ScoringResult:
    """Output of `score_transaction`.

    `fraud_score` and `all_shap_values` are None when a HARD_BLOCK rule
    fired — the model wasn't invoked. `threshold` is always populated
    (from the explainer singleton) so the caller can report it for context
    even on hard-blocked rows.
    """

    fraud_score: float | None
    decision: Decision
    threshold: float
    rules_triggered: list[str]
    top_contributors: list[dict[str, Any]]
    all_shap_values: dict[str, float] | None
    computed_at: datetime


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_context(db: Session, tx: Transaction) -> TransactionContext:
    """Load customer, merchant, and last 180 days of customer history.

    One indexed SELECT per join. The 180-day window matches the longest
    rule lookback (dormant account) so this single fetch serves both the
    rules engine and the feature extractor — per ADR decision #5.

    Raises ValueError if customer or merchant is missing (would indicate
    a corrupt write; FK constraints should prevent this in practice).
    """
    customer = db.get(Customer, tx.customer_id)
    merchant = db.get(Merchant, tx.merchant_id)
    if customer is None or merchant is None:
        raise ValueError(
            f"Cannot score tx {tx.id}: customer or merchant missing"
        )

    cutoff = tx.created_at - _RECENT_HISTORY_WINDOW
    stmt = (
        select(Transaction)
        .where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at < tx.created_at,
                Transaction.created_at >= cutoff,
            )
        )
        .order_by(Transaction.created_at.desc())
    )
    recent = list(db.execute(stmt).scalars().all())

    return TransactionContext(
        transaction=tx,
        customer=customer,
        merchant=merchant,
        recent_transactions=recent,
    )


def _model_decision_from_score(score: float, threshold: float) -> Decision:
    """Map a probability to APPROVE / REVIEW / DECLINE.

    Mirrors `FraudExplainer.classify` but returns the typed enum
    (the explainer returns bare strings despite its annotation; wrapping
    here keeps the type system honest).
    """
    if score >= threshold:
        return Decision.DECLINE
    if score >= threshold * 0.5:
        return Decision.REVIEW
    return Decision.APPROVE


def _compose_decision(
    rules: list[RuleResult],
    model_decision: Decision,
) -> tuple[Decision, list[str]]:
    """Apply the conservative-wins decision matrix.

    Matrix (from PHASE_3_DESIGN.md):

      Hard rule | REVIEW rule | Model output    | Final
      ----------|-------------|------------------|--------
      fired     | —           | (irrelevant)     | DECLINE
      —         | fired       | APPROVE          | REVIEW
      —         | fired       | REVIEW           | REVIEW
      —         | fired       | DECLINE          | DECLINE
      —         | —           | APPROVE/REVIEW/DECLINE | as model says

    Returns the final decision plus the names of every triggered rule
    (preserving `evaluate_all` order), regardless of which signal won.
    """
    triggered = [r for r in rules if r.triggered]
    rule_names = [r.rule_name for r in triggered]

    has_hard_block = any(r.severity == Severity.HARD_BLOCK for r in triggered)
    has_review = any(r.severity == Severity.REVIEW for r in triggered)

    if has_hard_block:
        return Decision.DECLINE, rule_names
    if has_review:
        if model_decision == Decision.DECLINE:
            return Decision.DECLINE, rule_names  # model wins; more conservative
        return Decision.REVIEW, rule_names
    return model_decision, rule_names


def _write_audit(
    db: Session,
    *,
    tx_id: str,
    decision: Decision,
    payload: dict[str, Any],
    is_hard_block: bool,
) -> None:
    """Record one audit_log row for the scoring decision.

    `action` is `scored.hard_block` for HARD_BLOCK paths so post-hoc
    analysis can separate rule-driven declines from model-driven ones.
    """
    action = "scored.hard_block" if is_hard_block else f"scored.{decision.value.lower()}"
    audit_repository.record(
        db,
        actor=f"scorer:{_MODEL_VERSION}",
        action=action,
        resource_type="transaction",
        resource_id=tx_id,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def score_transaction(
    db: Session,
    tx: Transaction,
    *,
    write_audit: bool = True,
) -> ScoringResult:
    """Score one transaction end-to-end.

    Does NOT commit the session — the caller is responsible. The endpoint
    commits via `idempotency.store()`; the latency benchmark rolls back.

    Pass `write_audit=False` to skip the audit log row (the benchmark
    uses this to keep synthetic load from polluting `audit_log`).
    """
    computed_at = datetime.now(timezone.utc)
    ctx = _load_context(db, tx)
    rule_results = evaluate_all(ctx)
    triggered_rules = [r for r in rule_results if r.triggered]

    explainer = get_explainer()
    threshold = float(explainer.threshold)

    # HARD_BLOCK short-circuit — skip the model entirely.
    if any(r.severity == Severity.HARD_BLOCK for r in triggered_rules):
        rule_names = [r.rule_name for r in triggered_rules]
        if write_audit:
            _write_audit(
                db,
                tx_id=tx.id,
                decision=Decision.DECLINE,
                payload={
                    "decision": Decision.DECLINE.value,
                    "threshold": threshold,
                    "rules_triggered": rule_names,
                    "rule_reasons": [r.reason for r in triggered_rules],
                },
                is_hard_block=True,
            )
        return ScoringResult(
            fraud_score=None,
            decision=Decision.DECLINE,
            threshold=threshold,
            rules_triggered=rule_names,
            top_contributors=[],
            all_shap_values=None,
            computed_at=computed_at,
        )

    # Normal path: extract features, score, attribute, compose.
    features = _extractor.extract(
        db,
        tx,
        customer=ctx.customer,
        merchant=ctx.merchant,
        recent_transactions=ctx.recent_transactions,
    )
    features_array = np.asarray(features.values, dtype=np.float64)
    local = explainer.explain_local(features_array)

    model_decision = _model_decision_from_score(local.fraud_score, threshold)
    composed_decision, rules_triggered_names = _compose_decision(
        rule_results, model_decision
    )

    top_contribs = top_contributors(
        FEATURE_NAMES, features_array, local.shap_values, k=5,
    )
    all_shap = {
        name: float(local.shap_values[i])
        for i, name in enumerate(FEATURE_NAMES)
    }

    if write_audit:
        _write_audit(
            db,
            tx_id=tx.id,
            decision=composed_decision,
            payload={
                "fraud_score": float(local.fraud_score),
                "decision": composed_decision.value,
                "threshold": threshold,
                "rules_triggered": rules_triggered_names,
                "top_contributors": top_contribs,
            },
            is_hard_block=False,
        )

    return ScoringResult(
        fraud_score=float(local.fraud_score),
        decision=composed_decision,
        threshold=threshold,
        rules_triggered=rules_triggered_names,
        top_contributors=top_contribs,
        all_shap_values=all_shap,
        computed_at=computed_at,
    )
