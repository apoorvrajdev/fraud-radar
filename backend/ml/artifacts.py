"""Artifact persistence for the trained fraud model.

XGBoost's native JSON format is used instead of pickle — it's portable across
Python and XGBoost versions, smaller, and human-readable enough to diff.

The artifact directory layout is:
    artifacts/
        model.json                XGBoost native (gitignored)
        feature_list.json         Canonical feature order (committed)
        threshold.json            Decision threshold + the FPR target it serves
        metrics.json              Test-set evaluation results (committed)
        training_metadata.json    Run provenance (committed)
        pr_curve.png              Plot of test-set PR curve (gitignored)
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb


@dataclass(frozen=True)
class ThresholdRecord:
    """The operating threshold chosen at training time."""

    value: float
    target_fpr: float
    realised_fpr_on_val: float


@dataclass(frozen=True)
class TrainingMetadata:
    """Provenance for a single training run."""

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
    """Write all six artifact files to `artifact_dir`."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # XGBoost native JSON — load_model() reads this back exactly
    model.get_booster().save_model(str(artifact_dir / "model.json"))

    _json_dump(artifact_dir / "feature_list.json", {"features": feature_names})
    _json_dump(artifact_dir / "threshold.json", asdict(threshold))
    _json_dump(artifact_dir / "metrics.json", metrics)
    _json_dump(artifact_dir / "training_metadata.json", asdict(metadata))


def load_model(artifact_dir: Path) -> xgb.XGBClassifier:
    """Reload the trained model from its native JSON artifact."""
    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "model.json"))
    clf = xgb.XGBClassifier()
    clf._Booster = booster  # type: ignore[attr-defined]
    return clf


def load_threshold(artifact_dir: Path) -> ThresholdRecord:
    """Reload the operating threshold."""
    with (artifact_dir / "threshold.json").open(encoding="utf-8") as f:
        payload = json.load(f)
    return ThresholdRecord(
        value=float(payload["value"]),
        target_fpr=float(payload["target_fpr"]),
        realised_fpr_on_val=float(payload["realised_fpr_on_val"]),
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
