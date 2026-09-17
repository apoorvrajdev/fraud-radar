"""Artifact persistence for the trained fraud model.

XGBoost's native JSON format is used instead of pickle — it's portable across
Python and XGBoost versions, smaller, and human-readable enough to diff.

The artifact directory layout is:
    artifacts/
        model.json                XGBoost native, the rounds the model predicts with (gitignored)
        feature_list.json         Canonical feature order (committed)
        threshold.json            Decision threshold, its FPR target, whether it fell back
        metrics.json              Test-set evaluation results (committed)
        training_metadata.json    The fit: folds, hyperparameters, fit settings, live features,
                                  and the SHA-256 of model.json (committed)
        pr_curve.png              Plot of test-set PR curve (gitignored)
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from ml.datasets.manifest import sha256_file

MODEL_FILENAME = "model.json"


@dataclass(frozen=True)
class ThresholdRecord:
    """The operating threshold chosen at training time.

    `fallback_used` is True when no val-fold threshold met `target_fpr`, so
    `value` is the fixed fallback rather than a threshold selected for that
    target, and `realised_fpr_on_val` may exceed it. None means the record was
    written before the flag existed: whether it fell back was not recorded.
    """

    value: float
    target_fpr: float
    realised_fpr_on_val: float
    fallback_used: bool | None


@dataclass(frozen=True)
class TrainingMetadata:
    """The record of a single model fit.

    Fold sizes and fraud rates, the chosen hyperparameters, and the settings
    the fit ran with. What went into a run is recorded separately, in
    `run.json` (see `ml/runs.py`).

    `random_state` is the fit's only random state: it seeds the hyperparameter
    search, the shuffle of that search's cross-validation folds, and the final
    model. `scale_pos_weight` is computed from the training fold's labels.
    `best_iteration` is the zero-based boosting round with the best val-fold
    score; boosting stops once `early_stopping_rounds` rounds pass without
    improving on it.

    `live_feature_count` counts the features that take more than one distinct
    value in the training fold, and `constant_features` names the others in
    feature order. The `_whole_matrix` pair is the same count over every row
    of the dataset (see `ml/reporting.py`).

    `model_sha256` is the SHA-256 of the `model.json` saved beside this record.
    model.json is never committed, so the digest is the only durable link
    between the record and the model it describes. `save_artifacts` computes
    it from the bytes it wrote, replacing any value passed in; it is None only
    in a record that has not been saved, or was saved before digests were.
    """

    trained_at_utc: str
    dataset_size: int
    train_size: int
    val_size: int
    test_size: int
    train_fraud_rate: float
    val_fraud_rate: float
    test_fraud_rate: float
    best_hyperparameters: dict[str, Any]
    library_versions: dict[str, str]
    random_state: int
    tuning_iterations: int
    tuning_cv_folds: int
    scale_pos_weight: float
    early_stopping_rounds: int
    best_iteration: int
    live_feature_count: int
    constant_features: list[str]
    live_feature_count_whole_matrix: int
    constant_features_whole_matrix: list[str]
    model_sha256: str | None = None


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def save_artifacts(
    artifact_dir: Path,
    *,
    model: xgb.XGBClassifier,
    feature_names: list[str],
    threshold: ThresholdRecord,
    metrics: dict[str, Any],
    metadata: TrainingMetadata,
) -> None:
    """Write all six artifact files to `artifact_dir`.

    The fit record is written last, with the digest of the model file as
    written, so it describes the exact bytes on disk.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # XGBoost native JSON — load_model() reads this back exactly
    model_path = artifact_dir / MODEL_FILENAME
    _predicting_booster(model).save_model(str(model_path))

    _json_dump(artifact_dir / "feature_list.json", {"features": feature_names})
    _json_dump(artifact_dir / "threshold.json", asdict(threshold))
    _json_dump(artifact_dir / "metrics.json", metrics)
    saved = replace(metadata, model_sha256=sha256_file(model_path))
    _json_dump(artifact_dir / "training_metadata.json", asdict(saved))


def _predicting_booster(model: xgb.XGBClassifier) -> xgb.Booster:
    """The booster holding exactly the rounds `model.predict_proba` scores with.

    Early stopping records its best round as a booster attribute and keeps the
    rounds boosted after it. The classifier scores with the rounds up to the
    best one, which is what the threshold and test metrics are computed from.
    A reloaded `Booster.predict`, which serving calls, scores with every round
    it holds, while SHAP limits itself to the recorded best round. Saving only
    the rounds the classifier uses, without the attribute, gives every reader
    that one function.

    The slice drops the attribute; the best round is recorded in
    training_metadata.json.
    """
    booster = model.get_booster()
    try:
        best_iteration = model.best_iteration
    except AttributeError:
        # No early stopping, so the classifier scores with every round.
        return booster
    # best_iteration is zero-based, so the stop is at least 1. A stop of 0
    # would select the whole model rather than none of it.
    return booster[: best_iteration + 1]


def load_model(artifact_dir: Path) -> xgb.XGBClassifier:
    """Reload the trained model from its native JSON artifact."""
    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "model.json"))
    clf = xgb.XGBClassifier()
    clf._Booster = booster
    return clf


def load_threshold(artifact_dir: Path) -> ThresholdRecord:
    """Reload the operating threshold."""
    with (artifact_dir / "threshold.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    # Absent from records that predate the flag: unknown, never assumed False.
    fallback_used = payload.get("fallback_used")
    return ThresholdRecord(
        value=float(payload["value"]),
        target_fpr=float(payload["target_fpr"]),
        realised_fpr_on_val=float(payload["realised_fpr_on_val"]),
        fallback_used=None if fallback_used is None else bool(fallback_used),
    )


def load_feature_list(artifact_dir: Path) -> list[str]:
    """Reload the canonical feature ordering."""
    with (artifact_dir / "feature_list.json").open(encoding="utf-8") as f:
        return list(json.load(f)["features"])


def predict_proba(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(fraud=1) using a Booster reloaded from native JSON.

    `XGBClassifier.predict_proba` requires sklearn-side attributes that the
    native loader doesn't restore, so we call the underlying booster directly
    on a DMatrix and return the raw probability column.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X)
    return np.asarray(booster.predict(dmat))


def collect_library_versions() -> dict[str, str]:
    """Capture versions of the libraries that affect training reproducibility."""
    import sklearn  # noqa: PLC0415

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "xgboost": xgb.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
    }


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, no microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
