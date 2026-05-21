"""Transaction-scoped API endpoints.

Currently exposes the SHAP explanation endpoint added in Phase 2H. The
transaction CRUD endpoints will land in Phase 3 alongside the rules engine
and ingestion pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.fraud import FEATURE_NAMES, FeatureExtractor, FraudExplainer
from app.fraud.explainer import get_explainer, top_contributors
from app.fraud.plots import render_force_plot, render_waterfall_plot
from app.models.transaction import Transaction
from app.schemas.explanation import (
    ContributorEntry,
    ExplanationFormat,
    ExplanationResponse,
)

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
