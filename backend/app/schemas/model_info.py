"""Pydantic schemas for GET /api/v1/model (Phase 5G).

What the API is actually serving, and — just as importantly — what it
is *not*. This project trains on three data layers that are never
merged (see the README's "Three data layers, kept apart"), and a
dashboard that showed a headline number without saying which layer it
came from would undo that separation in one line of UI.

So the envelope has two halves. `serving` describes the artifacts the
running process loaded. `benchmarks` describes the evaluation tracks
that exist in the repository, every one of them marked `served: false`,
because none has been promoted — `ml/promote.py` is built and tested
and has deliberately never been run.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ServingMetrics(BaseModel):
    """The headline numbers from the served run's own test fold."""

    model_config = ConfigDict(frozen=True)

    pr_auc: float
    roc_auc: float
    recall_at_1pct_fpr: float
    test_size: int | None = None
    test_fraud_rate: float | None = None


class ServingModel(BaseModel):
    """The model artifacts this process has loaded.

    Fields that can be read from the artifacts are read from them;
    `dataset_name` and `dataset_kind` describe a lineage the current
    artifacts predate carrying, and are sourced from the constant in
    ``app.services.model_info`` with its provenance documented there.
    """

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(
        description="Which dataset the served model was trained on",
    )
    dataset_kind: str = Field(
        description=(
            "The nature of that data — 'synthetic' here. Never 'real': "
            "no model trained on real data has been promoted."
        ),
    )
    featureset_version: str = Field(
        description="Featureset the served model expects, from the registry",
    )
    feature_count: int = Field(ge=0)
    trained_at_utc: str | None = Field(
        default=None,
        description="From training_metadata.json; null if unreadable",
    )
    threshold: float | None = Field(
        default=None,
        description="Operating threshold in force, selected on the validation fold",
    )
    metrics: ServingMetrics | None = Field(
        default=None,
        description=(
            "Held-out test-fold metrics for the *training generator*. These "
            "measure how learnable that generator is, not how detectable "
            "fraud is — see `metrics_caveat`."
        ),
    )
    metrics_caveat: str = Field(
        description="One line stating what the metrics above do and do not say",
    )


class BenchmarkTrack(BaseModel):
    """One evaluation track, and its relationship to the served model."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str = Field(
        description="'synthetic' (in-house), 'synthetic-external', or 'real-anonymised'",
    )
    description: str
    served: bool = Field(
        description="Whether this track's model is the one answering requests",
    )
    card_path: str | None = Field(
        default=None,
        description="Repository path to the card holding this track's results",
    )


class ModelInfo(BaseModel):
    """Envelope for GET /api/v1/model."""

    model_config = ConfigDict(frozen=True)

    serving: ServingModel
    benchmarks: list[BenchmarkTrack]
    reporting_currency: str = Field(
        description="Currency every /stats money field is denominated in",
    )
