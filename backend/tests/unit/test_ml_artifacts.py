"""Round-trip tests: save → reload preserves predictions byte-for-byte."""
from __future__ import annotations

import json
from dataclasses import asdict
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


def _save(artifact_dir: Path, model: xgb.XGBClassifier, n_features: int) -> None:
    """Save `model` with fixed threshold, metrics and metadata records."""
    save_artifacts(
        artifact_dir,
        model=model,
        feature_names=[f"f{index}" for index in range(n_features)],
        threshold=ThresholdRecord(
            value=0.5,
            target_fpr=0.01,
            realised_fpr_on_val=0.008,
            fallback_used=False,
        ),
        metrics={"pr_auc": 0.9, "roc_auc": 0.95},
        metadata=TrainingMetadata(
            trained_at_utc="2026-05-21T00:00:00+00:00",
            dataset_size=200,
            train_size=140,
            val_size=30,
            test_size=30,
            train_fraud_rate=0.5,
            val_fraud_rate=0.5,
            test_fraud_rate=0.5,
            best_hyperparameters={"max_depth": 3},
            library_versions=collect_library_versions(),
            random_state=42,
            tuning_iterations=25,
            tuning_cv_folds=4,
            scale_pos_weight=1.0,
            early_stopping_rounds=50,
            best_iteration=12,
            live_feature_count=n_features - 1,
            constant_features=["f0"],
            live_feature_count_whole_matrix=n_features,
            constant_features_whole_matrix=[],
        ),
    )


def _saved_booster(artifact_dir: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "model.json"))
    return booster


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

    _save(tmp_path, model, n_features=5)
    return model, X, tmp_path


@pytest.fixture
def early_stopped() -> tuple[xgb.XGBClassifier, np.ndarray]:
    """A model that kept boosting for ten rounds after its best one.

    Noisy labels make the val score peak early: the best round is 3 of 14.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(size=300) > 1).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.3,
        eval_metric="aucpr",
        random_state=42,
        early_stopping_rounds=10,
    )
    model.fit(X[:200], y[:200], eval_set=[(X[200:], y[200:])], verbose=False)

    # Only meaningful if rounds after the best one exist and change the scores.
    assert model.best_iteration + 1 < model.get_booster().num_boosted_rounds()
    every_round = model.get_booster().predict(xgb.DMatrix(X))
    assert not np.array_equal(every_round, model.predict_proba(X)[:, 1])
    return model, X


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


def test_an_early_stopped_model_is_saved_with_only_the_rounds_it_predicts_with(
    early_stopped: tuple[xgb.XGBClassifier, np.ndarray], tmp_path: Path
) -> None:
    model, X = early_stopped

    _save(tmp_path, model, n_features=4)

    saved = _saved_booster(tmp_path)
    assert saved.num_boosted_rounds() == model.best_iteration + 1
    # No early-stopping state survives, so no reader can apply it differently.
    assert saved.attr("best_iteration") is None
    assert saved.attr("best_score") is None
    np.testing.assert_array_equal(
        predict_proba(load_model(tmp_path), X), model.predict_proba(X)[:, 1]
    )


def test_a_model_without_early_stopping_is_saved_with_every_round(trained_pair: tuple) -> None:
    model, X, artifact_dir = trained_pair
    assert not hasattr(model, "best_iteration")

    saved = _saved_booster(artifact_dir)

    assert saved.num_boosted_rounds() == model.get_booster().num_boosted_rounds() == 20
    np.testing.assert_array_equal(
        predict_proba(load_model(artifact_dir), X), model.predict_proba(X)[:, 1]
    )


def test_a_best_round_of_zero_saves_one_round_rather_than_the_whole_model(tmp_path: Path) -> None:
    """A zero-based best round of 0 is a stop of 1; a stop of 0 would select every round."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 4))
    y = (rng.random(200) < 0.3).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.3,
        eval_metric="aucpr",
        random_state=42,
        early_stopping_rounds=5,
    )
    model.fit(X[:120], y[:120], eval_set=[(X[120:], y[120:])], verbose=False)
    assert model.best_iteration == 0
    assert model.get_booster().num_boosted_rounds() > 1

    _save(tmp_path, model, n_features=4)

    assert _saved_booster(tmp_path).num_boosted_rounds() == 1
    np.testing.assert_array_equal(
        predict_proba(load_model(tmp_path), X), model.predict_proba(X)[:, 1]
    )


def test_feature_list_round_trip(trained_pair: tuple) -> None:
    _, _, artifact_dir = trained_pair
    assert load_feature_list(artifact_dir) == ["f0", "f1", "f2", "f3", "f4"]


def test_threshold_round_trip(trained_pair: tuple) -> None:
    _, _, artifact_dir = trained_pair
    threshold = load_threshold(artifact_dir)
    assert threshold.value == 0.5
    assert threshold.target_fpr == 0.01
    assert threshold.realised_fpr_on_val == pytest.approx(0.008)
    assert threshold.fallback_used is False


def test_a_fallback_threshold_reads_back_as_a_fallback(tmp_path: Path) -> None:
    record = ThresholdRecord(
        value=0.5, target_fpr=0.01, realised_fpr_on_val=0.2, fallback_used=True
    )
    (tmp_path / "threshold.json").write_text(json.dumps(asdict(record)), encoding="utf-8")

    assert load_threshold(tmp_path) == record


def test_a_threshold_written_before_the_fallback_flag_reads_as_unrecorded(
    tmp_path: Path,
) -> None:
    """The committed threshold.json predates the flag; it must still load."""
    legacy = {"realised_fpr_on_val": 0.0074, "target_fpr": 0.01, "value": 0.7431}
    (tmp_path / "threshold.json").write_text(json.dumps(legacy), encoding="utf-8")

    threshold = load_threshold(tmp_path)

    assert threshold.fallback_used is None
    assert (threshold.value, threshold.target_fpr, threshold.realised_fpr_on_val) == (
        0.7431,
        0.01,
        0.0074,
    )


def test_training_metadata_adds_the_fit_settings_and_live_features_to_the_existing_fields(
    trained_pair: tuple,
) -> None:
    _, _, artifact_dir = trained_pair
    payload = json.loads((artifact_dir / "training_metadata.json").read_text(encoding="utf-8"))

    existing = {
        "trained_at_utc",
        "dataset_size",
        "train_size",
        "val_size",
        "test_size",
        "train_fraud_rate",
        "val_fraud_rate",
        "test_fraud_rate",
        "best_hyperparameters",
        "library_versions",
    }
    fit = {
        "random_state": 42,
        "tuning_iterations": 25,
        "tuning_cv_folds": 4,
        "scale_pos_weight": 1.0,
        "early_stopping_rounds": 50,
        "best_iteration": 12,
    }
    live = {
        "live_feature_count": 4,
        "constant_features": ["f0"],
        "live_feature_count_whole_matrix": 5,
        "constant_features_whole_matrix": [],
    }
    assert set(payload) == existing | set(fit) | set(live)
    assert {key: payload[key] for key in fit} == fit
    assert {key: payload[key] for key in live} == live


def test_collect_library_versions_returns_real_strings() -> None:
    versions = collect_library_versions()
    for key in ("python", "xgboost", "scikit-learn", "numpy"):
        assert key in versions
        assert versions[key] and isinstance(versions[key], str)
