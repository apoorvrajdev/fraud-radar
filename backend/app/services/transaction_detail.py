"""Phase 3G — composite read of one transaction's detail envelope.

Owns the read path for `GET /api/v1/transactions/{id}`. Hydrates the
SHAP contributors from the persisted `top_features` JSON column,
classifies each contributor's direction, computes the
`effective_decision` from `analyst_label` (falling through to
`fraud_decision` when no analyst has reviewed), and bundles the
trailing audit-log entries.

The explainer is NOT re-invoked here — that is what `/explain` is
for. The detail page is a read of what was decided at scoring time,
which is what an audit log fundamentally requires.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, cast

from sqlalchemy.orm import Session

from app.fraud.decision import Decision
from app.fraud.explainer import get_explainer
from app.models.audit_log import AuditLog
from app.models.transaction import Transaction
from app.repositories.audit import audit_repository
from app.schemas.common import PaymentMethod, TransactionStatus
from app.schemas.explanation import ContributorEntry
from app.schemas.transaction import (
    AuditEntry,
    TransactionDetail,
)


def effective_decision(
    fraud_decision: str | None,
    analyst_label: str | None,
) -> Decision:
    """Map (`fraud_decision`, `analyst_label`) → final business decision.

    The analyst's verdict overrides the model's when present:
    `CONFIRMED_FRAUD` projects to DECLINE, `CONFIRMED_LEGIT` to APPROVE.
    A null label falls through to `fraud_decision`; a null
    `fraud_decision` defaults to PENDING.
    """
    if analyst_label == "CONFIRMED_FRAUD":
        return Decision.DECLINE
    if analyst_label == "CONFIRMED_LEGIT":
        return Decision.APPROVE
    if fraud_decision is None:
        return Decision.PENDING
    return Decision(fraud_decision)


def _parse_top_contributors(raw: str | None) -> list[ContributorEntry]:
    """Parse the persisted `top_features` JSON column into ContributorEntry rows.

    Returns an empty list when the column is null or malformed —
    legacy rows from the Phase 3B stub era (before scoring was wired)
    have no attribution. The detail page handles that gracefully.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    out: list[ContributorEntry] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        try:
            feature = str(entry["feature"])
            feature_value = float(entry["feature_value"])
            shap_value = float(entry["shap_value"])
        except (KeyError, TypeError, ValueError):
            continue
        # Prefer the direction baked in by the explainer at scoring
        # time; fall back to recomputing if the persisted row is from
        # an older schema. `shap_value == 0` ties toward legit so a
        # zero-contribution feature never visually accuses anyone.
        raw_direction = entry.get("direction")
        if raw_direction in ("fraud", "legit"):
            direction = raw_direction
        else:
            direction = "fraud" if shap_value > 0 else "legit"
        out.append(
            ContributorEntry(
                feature=feature,
                feature_value=feature_value,
                shap_value=shap_value,
                direction=direction,
            )
        )
    return out


def _parse_rules(raw: str | None) -> list[str]:
    """Parse the persisted `rules_triggered` JSON column into a list of names."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(r) for r in parsed]


def _parse_audit_payload(raw: str | None) -> dict[str, Any] | None:
    """Parse one audit-log row's payload column; tolerate null/malformed."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _audit_entry(row: AuditLog) -> AuditEntry:
    return AuditEntry(
        id=row.id,
        actor=row.actor,
        action=row.action,
        payload=_parse_audit_payload(row.payload),
        created_at=row.created_at,
    )


def build_detail(db: Session, tx: Transaction) -> TransactionDetail:
    """Assemble the full TransactionDetail envelope for one transaction."""
    audit_rows = audit_repository.recent_for_resource(
        db,
        resource_type="transaction",
        resource_id=tx.id,
        limit=20,
    )

    # `explainer.threshold` is the live operating threshold from the
    # trained artifacts; it is stable across the process and safe to
    # surface on every detail read.
    try:
        threshold: float | None = float(get_explainer().threshold)
    except Exception:  # fail open if artifacts unavailable
        threshold = None

    # Hydrate the row fields explicitly. The Literal-typed columns
    # (status, payment_method) are constrained at the DB layer by
    # CHECK constraints, so a cast is safe here; Pydantic will still
    # raise if the value is somehow out of range.
    return TransactionDetail(
        id=tx.id,
        customer_id=tx.customer_id,
        merchant_id=tx.merchant_id,
        amount=tx.amount,
        currency=tx.currency,
        status=cast(TransactionStatus, tx.status),
        payment_method=cast(PaymentMethod, tx.payment_method),
        country=tx.country,
        card_last4=tx.card_last4,
        ip_address=tx.ip_address,
        device_id=tx.device_id,
        is_card_present=tx.is_card_present,
        fraud_score=(
            None if tx.fraud_score is None else Decimal(str(tx.fraud_score))
        ),
        fraud_decision=tx.fraud_decision,
        threshold=threshold,
        rules_triggered=_parse_rules(tx.rules_triggered),
        top_contributors=_parse_top_contributors(tx.top_features),
        effective_decision=effective_decision(
            tx.fraud_decision, tx.analyst_label,
        ),
        analyst_label=tx.analyst_label,
        analyst_notes=tx.analyst_notes,
        reviewed_at=tx.reviewed_at,
        created_at=tx.created_at,
        audit=[_audit_entry(row) for row in audit_rows],
    )
