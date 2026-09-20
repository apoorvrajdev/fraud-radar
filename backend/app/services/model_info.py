"""Phase 5G — assemble the model/dataset identity envelope.

Reads what the loaded artifacts can prove, and states the rest from
constants whose provenance is documented here rather than guessed.

The distinction matters. `trained_at_utc`, the threshold, the feature
count and the test metrics are all read from files on disk, so they
cannot drift from what is being served. The *lineage* — which dataset
produced those artifacts — is not recorded in the artifacts this
repository currently ships, and inventing a value for it would be
exactly the kind of unfounded claim a model badge must not make.

It is nonetheless a known fact, not a guess: the served artifacts are
written only by ``ml.train`` with no ``--dataset`` argument, which is
the in-house synthetic generator, or by ``ml.promote``, which has never
been run (README: "Promotion is built and tested, and deliberately has
not been run"). So the lineage is stated as a constant below, and if a
future run records it in the artifacts the reader should prefer that.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.fraud.feature_spec import DEFAULT_FEATURESET, feature_names
from app.schemas.model_info import (
    BenchmarkTrack,
    ModelInfo,
    ServingMetrics,
    ServingModel,
)

log = logging.getLogger(__name__)

# Lineage of the served artifacts — see the module docstring for why
# this is a constant rather than a read.
_SERVED_DATASET_NAME = "synthetic_v1"
_SERVED_DATASET_KIND = "synthetic"

# Deliberately not a performance claim. It says what the numbers
# measure, which is the one thing a headline metric on a dashboard
# routinely fails to say.
_METRICS_CAVEAT = (
    "Measured on a held-out chronological fold of this repository's own "
    "synthetic generator. It measures how learnable that generator is, "
    "not how detectable real fraud is."
)

# The evaluation tracks that exist in the repository. Wording follows
# the benchmark cards and the README; every entry is served=False
# because no benchmark run has been promoted.
_BENCHMARKS: tuple[BenchmarkTrack, ...] = (
    BenchmarkTrack(
        name="synthetic_v1",
        kind="synthetic",
        description=(
            "This repository's own generator — 50,010 transactions with six "
            "injected fraud patterns. The development baseline, and the data "
            "the served model is trained on."
        ),
        served=True,
        card_path="backend/ml/MODEL_CARD.md",
    ),
    BenchmarkTrack(
        name="sparkov_v1",
        kind="synthetic-external",
        description=(
            "Simulated data from an independent generator (Sparkov). Not real "
            "card transactions. Tests whether features designed against this "
            "project's own simulator transfer at all."
        ),
        served=False,
        card_path="backend/ml/BENCHMARK_CARD.md",
    ),
    BenchmarkTrack(
        name="ulb_pca_v1",
        kind="real-anonymised",
        description=(
            "Real, publisher-anonymised card data (ULB), on its own featureset "
            "of published PCA components. Runs on an isolated track and cannot "
            "be promoted or served."
        ),
        served=False,
        card_path="backend/ml/ULB_BENCHMARK_CARD.md",
    ),
)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read one artifact file, or None if it is absent or unreadable.

    A missing artifact degrades the badge rather than failing the
    request: the dashboard showing "unknown" is better than a 500 on a
    page whose other tiles are fine.
    """
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        log.warning("Model info: could not read %s (%s)", path.name, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _metrics_from(
    metrics: dict[str, Any] | None, metadata: dict[str, Any] | None
) -> ServingMetrics | None:
    """Build the metrics block, or None if the core numbers are absent."""
    if metrics is None:
        return None
    try:
        return ServingMetrics(
            pr_auc=float(metrics["test_pr_auc"]),
            roc_auc=float(metrics["test_roc_auc"]),
            recall_at_1pct_fpr=float(metrics["recall_at_1pct_fpr"]),
            test_size=(
                int(metadata["test_size"])
                if metadata and "test_size" in metadata
                else None
            ),
            test_fraud_rate=(
                float(metadata["test_fraud_rate"])
                if metadata and "test_fraud_rate" in metadata
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Model info: metrics.json missing expected fields (%s)", exc)
        return None


def get_model_info(artifacts_dir: Path | str | None = None) -> ModelInfo:
    """Assemble the identity envelope from the served artifacts."""
    settings = get_settings()
    root = Path(artifacts_dir or settings.model_artifacts_dir)

    metadata = _read_json(root / "training_metadata.json")
    metrics = _read_json(root / "metrics.json")
    threshold_payload = _read_json(root / "threshold.json")
    feature_payload = _read_json(root / "feature_list.json")

    # Prefer a featureset version recorded in the artifacts; fall back
    # to the registry default, which is what the extractor uses.
    featureset_version = DEFAULT_FEATURESET
    feature_count = len(feature_names(featureset_version))
    if feature_payload is not None:
        recorded = feature_payload.get("featureset_version")
        if isinstance(recorded, str) and recorded:
            featureset_version = recorded
        features = feature_payload.get("features")
        if isinstance(features, list):
            feature_count = len(features)

    threshold: float | None = None
    if threshold_payload is not None:
        try:
            threshold = float(threshold_payload["value"])
        except (KeyError, TypeError, ValueError):
            log.warning("Model info: threshold.json has no usable 'value'")

    trained_at = None
    if metadata is not None:
        raw = metadata.get("trained_at_utc")
        trained_at = str(raw) if raw else None

    serving = ServingModel(
        dataset_name=_SERVED_DATASET_NAME,
        dataset_kind=_SERVED_DATASET_KIND,
        featureset_version=featureset_version,
        feature_count=feature_count,
        trained_at_utc=trained_at,
        threshold=threshold,
        metrics=_metrics_from(metrics, metadata),
        metrics_caveat=_METRICS_CAVEAT,
    )

    return ModelInfo(
        serving=serving,
        benchmarks=list(_BENCHMARKS),
        reporting_currency=settings.fx_base_currency,
    )
