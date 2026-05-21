"""Tests for the SHAP-backed explainer wrapper.

Tests build a tiny in-memory XGBoost model with the canonical 17-feature
schema. They do NOT depend on Phase 2G artifacts existing on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import shap
import xgboost as xgb

from app.fraud.explainer import (
    FraudExplainer,
    initialize_explainer,
    reset_explainer_for_tests,
    top_contributors,
)
from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Each test starts with no cached explainer."""
    reset_explainer_for_tests()
    yield
    reset_explainer_for_tests()


def _train_tiny_model(seed: int = 42) -> xgb.Booster:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, N_FEATURES))
    # Easy signal so the model trains cleanly: positive class iff first
    # feature is above its median.
    y = (X[:, 0] > 0).astype(int)
    dmat = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "max_depth": 3,
            "eta": 0.3,
            "seed": seed,
        },
        dtrain=dmat,
        num_boost_round=20,
    )
    return booster


def _build_explainer() -> FraudExplainer:
    booster = _train_tiny_model()
    return FraudExplainer(
        booster=booster,
        explainer=shap.TreeExplainer(booster),
        threshold=0.5,
        feature_names=list(FEATURE_NAMES),
    )


def _write_artifacts(tmp_path: Path) -> Path:
    """Persist a tiny model + threshold + feature list to mimic Phase 2G."""
    booster = _train_tiny_model()
    booster.save_model(str(tmp_path / "model.json"))
    (tmp_path / "threshold.json").write_text(
        json.dumps({"value": 0.5, "target_fpr": 0.01, "realised_fpr_on_val": 0.008})
    )
    (tmp_path / "feature_list.json").write_text(
        json.dumps({"features": list(FEATURE_NAMES)})
    )
    return tmp_path


def test_explainer_loads_artifacts(tmp_path: Path) -> None:
    artifacts_dir = _write_artifacts(tmp_path)
    explainer = initialize_explainer(artifacts_dir)
    assert explainer.threshold == pytest.approx(0.5)
    assert explainer.feature_names == FEATURE_NAMES


def test_missing_artifacts_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Run .* ml\.train"):
        initialize_explainer(tmp_path)


def test_initialize_is_idempotent(tmp_path: Path) -> None:
    artifacts_dir = _write_artifacts(tmp_path)
    a = initialize_explainer(artifacts_dir)
    b = initialize_explainer(artifacts_dir)
    assert a is b


def test_shap_values_satisfy_additivity() -> None:
    """sum(shap_values) + base_value ≈ raw model output.

    This is the load-bearing correctness check for SHAP. If additivity
    fails, every explanation in the dashboard is misleading.
    """
    explainer = _build_explainer()
    rng = np.random.default_rng(0)
    x = rng.normal(size=N_FEATURES)

    local = explainer.explain_local(x)

    # The booster outputs probability (after sigmoid), but TreeExplainer for a
    # binary:logistic XGBoost returns shap_values on the *margin* (logit) by
    # default. Additivity holds on the margin, so compare in margin space.
    dmat = xgb.DMatrix(x.reshape(1, -1), feature_names=FEATURE_NAMES)
    raw_margin = float(
        explainer._booster.predict(dmat, output_margin=True)[0]  # noqa: SLF001
    )
    reconstructed = float(local.base_value + local.shap_values.sum())
    assert reconstructed == pytest.approx(raw_margin, abs=1e-4)


def test_top_contributors_are_sorted_by_abs_value() -> None:
    feature_values = np.zeros(N_FEATURES)
    shap_values = np.array([0.1, -0.9, 0.05, 0.6, -0.3] + [0.0] * (N_FEATURES - 5))
    rows = top_contributors(FEATURE_NAMES, feature_values, shap_values, k=5)

    abs_values = [abs(r["shap_value"]) for r in rows]
    assert abs_values == sorted(abs_values, reverse=True)


def test_top_contributors_respects_k() -> None:
    feature_values = np.zeros(N_FEATURES)
    shap_values = np.linspace(-1.0, 1.0, N_FEATURES)
    rows = top_contributors(FEATURE_NAMES, feature_values, shap_values, k=3)
    assert len(rows) == 3


def test_top_contributors_direction_matches_sign() -> None:
    feature_values = np.zeros(N_FEATURES)
    shap_values = np.zeros(N_FEATURES)
    shap_values[0] = 0.5  # positive → "fraud"
    shap_values[1] = -0.4  # negative → "legit"
    rows = top_contributors(FEATURE_NAMES, feature_values, shap_values, k=2)
    by_name = {r["feature"]: r for r in rows}
    assert by_name[FEATURE_NAMES[0]]["direction"] == "fraud"
    assert by_name[FEATURE_NAMES[1]]["direction"] == "legit"


def test_classify_maps_decision_ladder_correctly() -> None:
    explainer = _build_explainer()
    # threshold=0.5 → APPROVE < 0.25; REVIEW [0.25, 0.5); DECLINE ≥ 0.5
    assert explainer.classify(0.0) == "APPROVE"
    assert explainer.classify(0.24) == "APPROVE"
    assert explainer.classify(0.25) == "REVIEW"
    assert explainer.classify(0.49) == "REVIEW"
    assert explainer.classify(0.5) == "DECLINE"
    assert explainer.classify(0.99) == "DECLINE"


def test_compute_global_shap_shape() -> None:
    explainer = _build_explainer()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, N_FEATURES))
    shap_values = explainer.compute_global_shap(X)
    assert shap_values.shape == (8, N_FEATURES)


def test_compute_global_shap_rejects_wrong_shape() -> None:
    explainer = _build_explainer()
    bad = np.zeros((4, N_FEATURES - 1))
    with pytest.raises(ValueError, match="Expected shape"):
        explainer.compute_global_shap(bad)
