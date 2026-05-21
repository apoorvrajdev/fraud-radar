"""Pydantic schemas for the per-transaction SHAP explanation endpoint."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ExplanationFormat = Literal["json", "force", "waterfall"]
Decision = Literal["APPROVE", "REVIEW", "DECLINE"]
ContributionDirection = Literal["fraud", "legit"]


class ContributorEntry(BaseModel):
    """One feature's SHAP contribution to a single transaction's score."""

    model_config = ConfigDict(frozen=True)

    feature: str
    shap_value: float = Field(description="Signed SHAP contribution")
    feature_value: float = Field(description="Observed feature value at inference time")
    direction: ContributionDirection = Field(
        description="'fraud' if shap_value > 0, otherwise 'legit'"
    )


class ExplanationResponse(BaseModel):
    """Full per-transaction explanation payload."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    fraud_score: float = Field(ge=0.0, le=1.0)
    decision: Decision
    threshold: float = Field(ge=0.0, le=1.0)
    base_value: float = Field(description="explainer.expected_value (mean model output)")
    top_contributors: list[ContributorEntry]
    all_shap_values: dict[str, float]
    computed_at: datetime
