"""Phase 3G — analyst review / override write path.

Owns the mutation surface for `POST /api/v1/transactions/{id}/decision`.
Stamps `analyst_label`, `analyst_notes`, `reviewed_at` on the target
row and appends an audit-log entry inside one atomic commit.

`fraud_decision` is never mutated here. The model's verdict is
preserved verbatim so it can be measured against the analyst's call
during evaluation and retraining.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.fraud.decision import Decision
from app.models.transaction import Transaction
from app.repositories.audit import audit_repository
from app.repositories.transaction import transaction_repository
from app.schemas.transaction import AnalystDecisionRequest, TransactionDetail
from app.services.transaction_detail import build_detail


class ReviewConflictError(Exception):
    """Raised when the target row is not in a reviewable state.

    Mapped to 409 by the router. The detail message explains *why* a
    given row can't be reviewed (already-terminal decision, missing
    score, etc.) so the frontend can surface it directly.
    """


def _is_same_decision(
    tx: Transaction, request: AnalystDecisionRequest,
) -> bool:
    """True when the requested label+notes match what's already stored."""
    return (
        tx.analyst_label == request.label
        and (tx.analyst_notes or None) == (request.notes or None)
    )


def apply_analyst_decision(
    db: Session,
    *,
    transaction_id: str,
    analyst_id: str,
    request: AnalystDecisionRequest,
    now: datetime | None = None,
) -> TransactionDetail:
    """Commit (or idempotently no-op) an analyst override.

    Raises ``LookupError`` if the transaction does not exist (router
    maps to 404) and ``ReviewConflictError`` if the row is not in
    REVIEW (router maps to 409).

    Idempotent on identical resubmit: when the (label, notes) pair
    matches the persisted state, no columns are written and no
    audit-log row is appended. A *different* (label, notes) on an
    already-reviewed row is allowed and records a new
    ``ANALYST_DECISION_REVISED`` audit entry, matching how real
    review tools handle "I changed my mind".
    """
    if now is None:
        now = datetime.now(UTC)

    tx = db.get(Transaction, transaction_id)
    if tx is None:
        raise LookupError(f"Transaction {transaction_id} not found")

    if tx.fraud_decision != Decision.REVIEW.value:
        raise ReviewConflictError(
            f"Transaction {transaction_id} is not in REVIEW "
            f"(current fraud_decision={tx.fraud_decision!r}); "
            "only REVIEW rows are analyst-actionable."
        )

    # Idempotent no-op: same decision already on record.
    if _is_same_decision(tx, request):
        return build_detail(db, tx)

    prev_label = tx.analyst_label
    action = (
        "ANALYST_DECISION_REVISED"
        if prev_label is not None
        else "ANALYST_DECISION"
    )

    transaction_repository.apply_analyst_decision(
        db,
        tx=tx,
        label=request.label,
        notes=request.notes,
        now=now,
    )
    audit_repository.record(
        db,
        actor=analyst_id,
        action=action,
        resource_type="transaction",
        resource_id=tx.id,
        payload={
            "label": request.label,
            "notes": request.notes,
            "prev_label": prev_label,
        },
    )
    db.commit()
    db.refresh(tx)

    return build_detail(db, tx)
