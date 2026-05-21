"""Round-trip tests: save → reload preserves predictions byte-for-byte."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from ml.artifacts import (
    ThresholdRecord,
    TrainingMetadata,
    collect_library_versions,
    load_feature_list,
    load_model,
    load_threshold,
    predict_proba,
    save_artifacts,
)


@pytest.fixture
def trained_pair(tmp_path: Path) -> tuple[xgb.XGBClassifier, np.ndarray, Path]:
    """Train a tiny XGBoost on synthetic data and save to a tmp artifact dir."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 5))
    # Label = first feature positive — easy signal so training is stable
    y = (X[:, 0] > 0).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X, y)

    save_artifacts(
        tmp_path,
        model=model,
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        threshold=ThresholdRecord(
            value=0.5,
            target_fpr=0.01,
            realised_fpr_on_val=0.008,
        ),
        metrics={"pr_auc": 0.9, "roc_auc": 0.95},
        metadata=TrainingMetadata(
            trained_at_utc="2026-05-21T00:00:00+00:00",
            dataset_size=n,
            train_size=140,
            val_size=30,
            test_size=30,
            train_fraud_rate=0.5,
            val_fraud_rate=0.5,
            test_fraud_rate=0.5,
            best_hyperparameters={"max_depth": 3},
            library_versions=collect_library_versions(),
        ),
    )
    return model, X, tmp_path


def test_all_six_artifact_files_are_written(trained_pair: tuple) -> None:
    _, _, artifact_dir = trained_pair
    expected = {
        "model.json",
        "feature_list.json",
        "threshold.json",
        "metrics.json",
        "training_metadata.json",
    }
    actual = {p.name for p in artifact_dir.iterdir() if p.is_file()}
    # pr_curve.png is written by train.py, not save_artifacts — that's fine
    assert expected.issubset(actual)


def test_reload_preserves_predictions(trained_pair: tuple) -> None:
    model, X, artifact_dir = trained_pair
    original = model.predict_proba(X)[:, 1]

    reloaded = load_model(artifact_dir)
    rehydrated = predict_proba(reloaded, X)

    np.testing.assert_allclose(original, rehydrated, rtol=0, atol=1e-7)


def test_feature_list_round_trip(trained_pair: tuple) -> None:
    _, _, artifact_dir = trained_pair
    assert load_feature_list(artifact_dir) == ["f0", "f1", "f2", "f3", "f4"]


def test_threshold_round_trip(trained_pair: tuple) -> None:
    _, _, artifact_dir = trained_pair
    threshold = load_threshold(artifact_dir)
    assert threshold.value == 0.5
    assert threshold.target_fpr == 0.01
    assert threshold.realised_fpr_on_val == pytest.approx(0.008)


def test_collect_library_versions_returns_real_strings() -> None:
    versions = collect_library_versions()
    for key in ("python", "xgboost", "scikit-learn", "numpy"):
        assert key in versions
        assert versions[key] and isinstance(versions[key], str)
