"""Pydantic schemas for fraud scoring results."""
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import FraudDecision


class FraudFeature(BaseModel):
    """A single feature contributing to a fraud score, with SHAP value."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Feature name, e.g. 'amount_zscore_30d'")
    value: float | int | bool | str = Field(
        description="Observed feature value for this transaction"
    )
    contribution: float = Field(
        description="SHAP contribution to the final score (signed)"
    )


class FraudScoreResult(BaseModel):
    """Output of the fraud detection pipeline for a single transaction."""

    model_config = ConfigDict(frozen=True)

    score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
        decimal_places=4,
        description="Calibrated fraud probability in [0, 1]",
    )
    decision: FraudDecision = Field(
        description="Final decision after thresholds applied"
    )
    rules_triggered: list[str] = Field(
        default_factory=list,
        description="Names of rules engine rules that triggered for this tx",
    )
    top_features: list[FraudFeature] = Field(
        default_factory=list,
        description="Top contributing features for explainability",
    )
    model_version: str = Field(
        default="0.1.0",
        description="Version of the model that produced this score",
    )
