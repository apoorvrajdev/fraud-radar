"""Transaction-scoped API endpoints.

Phase 2H added the SHAP /explain endpoint. Phase 3B adds POST /transactions
(ingestion with Stripe-style idempotency) and GET /transactions/{id}
(read-back of a persisted decision). Real rules-plus-ML scoring is
deferred to Phase 3C; this layer persists a stub with `decision=PENDING`
and caches the response.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.fraud import FEATURE_NAMES, FeatureExtractor, FraudExplainer
from app.fraud.decision import Decision
from app.fraud.explainer import get_explainer, top_contributors
from app.fraud.plots import render_force_plot, render_waterfall_plot
from app.models.transaction import Transaction
from app.schemas.explanation import (
    ContributorEntry,
    ExplanationFormat,
    ExplanationResponse,
)
from app.schemas.transaction import (
    AnalystDecisionRequest,
    TransactionCreate,
    TransactionDetail,
    TransactionList,
    TransactionListQuery,
    TransactionScored,
)
from app.services.idempotency import hash_request, lookup, store
from app.services.review import ReviewConflictError, apply_analyst_decision
from app.services.scoring import score_transaction
from app.services.transaction_detail import build_detail
from app.services.transactions import list_transactions as list_transactions_service

router = APIRouter(prefix="/transactions", tags=["transactions"])

_TOP_K = 5
_extractor = FeatureExtractor()  # stateless — safe to share


def _extract_feature_vector(db: Session, tx: Transaction) -> np.ndarray:
    """Run the production FeatureExtractor and return a numpy row vector."""
    row = _extractor.extract(db, tx)
    return np.asarray(row.values, dtype=np.float64)


@router.get(
    "/{transaction_id}/explain",
    response_model=None,  # JSON path returns the schema; PNG paths return Response
    summary="SHAP explanation for a transaction (JSON, force plot, or waterfall plot)",
)
def explain_transaction(
    transaction_id: str,
    format: ExplanationFormat = Query(  # noqa: A002  shadowing builtin acceptable for FastAPI query name
        default="json",
        description="json (default), force, or waterfall",
    ),
    db: Session = Depends(get_db),
    explainer: FraudExplainer = Depends(get_explainer),
) -> ExplanationResponse | Response:
    """Return a per-transaction explanation in the requested format.

    `format=json` returns the structured `ExplanationResponse` with the top 5
    SHAP contributors plus the full per-feature value map. `format=force` and
    `format=waterfall` return rendered PNGs of the canonical SHAP visualisations.
    """
    tx = db.get(Transaction, transaction_id)
    if tx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )

    features = _extract_feature_vector(db, tx)
    local = explainer.explain_local(features)
    decision = explainer.classify(local.fraud_score)

    if format == "json":
        contributors = [
            ContributorEntry(**c)
            for c in top_contributors(
                FEATURE_NAMES,
                features,
                local.shap_values,
                k=_TOP_K,
            )
        ]
        all_shap = {
            name: float(local.shap_values[i])
            for i, name in enumerate(FEATURE_NAMES)
        }
        return ExplanationResponse(
            transaction_id=transaction_id,
            fraud_score=float(local.fraud_score),
            decision=decision,
            threshold=explainer.threshold,
            base_value=local.base_value,
            top_contributors=contributors,
            all_shap_values=all_shap,
            computed_at=datetime.now(timezone.utc),
        )

    if format == "force":
        png = render_force_plot(
            transaction_id=transaction_id,
            feature_names=FEATURE_NAMES,
            feature_values=features,
            shap_values=local.shap_values,
            base_value=local.base_value,
        )
        return Response(content=png, media_type="image/png")

    if format == "waterfall":
        png = render_waterfall_plot(
            transaction_id=transaction_id,
            feature_names=FEATURE_NAMES,
            feature_values=features,
            shap_values=local.shap_values,
            base_value=local.base_value,
        )
        return Response(content=png, media_type="image/png")

    # Pydantic's Literal type validation rejects anything else before we get
    # here — this is purely a defensive belt-and-braces guard.
    raise HTTPException(  # pragma: no cover
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Unsupported format: {format!r}",
    )


# ---------------------------------------------------------------------------
# POST /api/v1/transactions — Phase 3B ingestion
# ---------------------------------------------------------------------------


def _scored_from_transaction(tx: Transaction) -> TransactionScored:
    """Build a TransactionScored response from a persisted Transaction row.

    Reads SHAP attribution and triggered rules from the `top_features` and
    `rules_triggered` TEXT columns — the explainer is NOT re-invoked. SHAP
    is computed and persisted on POST (in Phase 3C; in 3B both columns are
    NULL and the response carries empty lists).
    """
    top_contributors_list: list[dict[str, Any]] = []
    if tx.top_features:
        try:
            parsed = json.loads(tx.top_features)
            if isinstance(parsed, list):
                top_contributors_list = parsed
        except json.JSONDecodeError:
            top_contributors_list = []

    rules_list: list[str] = []
    if tx.rules_triggered:
        try:
            parsed = json.loads(tx.rules_triggered)
            if isinstance(parsed, list):
                rules_list = [str(r) for r in parsed]
        except json.JSONDecodeError:
            rules_list = []

    return TransactionScored(
        transaction_id=tx.id,
        fraud_score=float(tx.fraud_score) if tx.fraud_score is not None else None,
        decision=(
            Decision(tx.fraud_decision) if tx.fraud_decision else Decision.PENDING
        ),
        threshold=None,  # Phase 3B does not persist threshold; 3C will.
        rules_triggered=rules_list,
        top_contributors=top_contributors_list,
        computed_at=tx.created_at,
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=TransactionScored,
    summary="Ingest a new transaction (Phase 3B stub — scoring lands in 3C)",
)
def create_transaction(
    payload: TransactionCreate,
    response: Response,
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=1,
        max_length=64,
        description="Stripe-style request key; required, 1-64 chars.",
    ),
    db: Session = Depends(get_db),
) -> TransactionScored:
    """Persist a transaction and cache the response for replay.

    See `docs/adr/PHASE_3_DESIGN.md` "API surface" and "Idempotency design".
    Phase 3B does not invoke the rules engine, the ML scorer, or the SHAP
    explainer — those land in Phase 3C. The response carries
    `decision=PENDING`, `fraud_score=None`, and empty
    `rules_triggered` / `top_contributors`.
    """
    request_hash = hash_request(payload)
    existing = lookup(db, idempotency_key)

    if existing is not None:
        if existing.request_hash != request_hash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="idempotency key reused with different payload",
            )
        response.headers["X-Idempotency-Replay"] = "true"
        response.status_code = existing.status_code
        return TransactionScored.model_validate_json(existing.response_body)

    tx_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    tx = Transaction(
        id=tx_id,
        idempotency_key=idempotency_key,
        customer_id=payload.customer_id,
        merchant_id=payload.merchant_id,
        amount=payload.amount,
        currency=payload.currency,
        # `status` CHECK allows {APPROVED, DECLINED, PENDING_REVIEW}. The
        # initial insert uses PENDING_REVIEW; `score_transaction` resolves
        # the real decision below and we overwrite the row before commit.
        status="PENDING_REVIEW",
        payment_method=payload.payment_method,
        card_last4=payload.card_last4,
        ip_address=payload.ip_address,
        device_id=payload.device_id,
        country=payload.country,
        is_card_present=payload.is_card_present,
        fraud_score=None,
        fraud_decision=Decision.PENDING.value,
        rules_triggered=None,
        top_features=None,
        created_at=now,
    )
    db.add(tx)
    db.flush()  # surface FK / CHECK violations before scoring

    # Phase 3C-2: end-to-end scoring. `score_transaction` reads the just-
    # added Transaction back through the session, runs rules + ML + SHAP,
    # writes one audit_log row, and returns the composed decision. It does
    # NOT commit — `idempotency.store` below commits both the Transaction
    # and the audit log row atomically.
    result = score_transaction(db, tx, write_audit=True)
    tx.fraud_score = (
        Decimal(str(result.fraud_score)) if result.fraud_score is not None else None
    )
    tx.fraud_decision = result.decision.value
    tx.top_features = (
        json.dumps(result.top_contributors) if result.top_contributors else None
    )
    tx.rules_triggered = (
        json.dumps(result.rules_triggered) if result.rules_triggered else None
    )
    tx.status = (
        "APPROVED" if result.decision == Decision.APPROVE
        else "DECLINED" if result.decision == Decision.DECLINE
        else "PENDING_REVIEW"  # for REVIEW
    )

    scored = TransactionScored(
        transaction_id=tx_id,
        fraud_score=result.fraud_score,
        decision=result.decision,
        threshold=result.threshold,
        rules_triggered=result.rules_triggered,
        top_contributors=result.top_contributors,
        computed_at=result.computed_at,
    )
    response_body = scored.model_dump_json()

    # `store` commits the session, which atomically flushes both the new
    # Transaction (staged via `db.add` above) and the IdempotencyKey row.
    store(
        db,
        key=idempotency_key,
        request_hash=request_hash,
        transaction_id=tx_id,
        response_body=response_body,
        status_code=status.HTTP_201_CREATED,
    )
    return scored


@router.get(
    "/{transaction_id}",
    response_model=TransactionDetail,
    summary="Read the full detail envelope for one transaction (Phase 3G)",
)
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
) -> TransactionDetail:
    """Return the composite TransactionDetail envelope.

    Reads from the persisted `transactions.top_features` and
    `rules_triggered` columns — the explainer is NOT re-invoked. The
    detail page is a read of what was decided at scoring time, which
    is what an audit log fundamentally requires.

    See `docs/adr/PHASE_3G_DESIGN.md` for the envelope shape and the
    `effective_decision` mapping.
    """
    tx = db.get(Transaction, transaction_id)
    if tx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {transaction_id} not found",
        )
    return build_detail(db, tx)


# ---------------------------------------------------------------------------
# POST /api/v1/transactions/{id}/decision — Phase 3G analyst override
# ---------------------------------------------------------------------------


@router.post(
    "/{transaction_id}/decision",
    response_model=TransactionDetail,
    summary="Commit an analyst CONFIRMED_FRAUD/CONFIRMED_LEGIT label (Phase 3G)",
)
def record_analyst_decision(
    transaction_id: str,
    payload: AnalystDecisionRequest,
    analyst_id: str = Header(
        ...,
        alias="X-Analyst-Id",
        min_length=1,
        max_length=64,
        description="Identifies the analyst for the audit log; trusted as declared.",
    ),
    db: Session = Depends(get_db),
) -> TransactionDetail:
    """Apply an analyst override to a REVIEW transaction.

    Returns the full updated TransactionDetail envelope. Idempotent on
    identical resubmit. Revisions (different label/notes on an
    already-reviewed row) record an `ANALYST_DECISION_REVISED` audit
    entry.

    Status codes:

      * 200 — override accepted (or idempotent no-op).
      * 404 — no transaction with that id.
      * 409 — transaction is not in REVIEW.
      * 422 — missing/invalid header or body.
    """
    try:
        return apply_analyst_decision(
            db,
            transaction_id=transaction_id,
            analyst_id=analyst_id,
            request=payload,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ReviewConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------------
# GET /api/v1/transactions — Phase 3F paginated list
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=TransactionList,
    summary="List transactions with filters and keyset pagination",
)
def list_transactions(
    query: Annotated[TransactionListQuery, Query()],
    db: Session = Depends(get_db),
) -> TransactionList:
    """Return a page of transactions, newest first.

    See `docs/adr/PHASE_3F_DESIGN.md` for the cursor format, filter
    semantics, and pagination contract. Malformed cursors return 422.
    """
    try:
        return list_transactions_service(db, query)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
