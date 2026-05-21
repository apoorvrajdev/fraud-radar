"""Tests for the global feature-importance ranking."""
from __future__ import annotations

import numpy as np
import pytest

from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES
from ml.analysis.global_importance import compute_feature_importance


def test_feature_importance_sorted_descending() -> None:
    """Returned list must be ordered by mean_abs_shap, largest first."""
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(50, N_FEATURES))
    result = compute_feature_importance(shap_values, list(FEATURE_NAMES))

    means = [f["mean_abs_shap"] for f in result["features"]]
    assert means == sorted(means, reverse=True)


def test_feature_importance_includes_all_features() -> None:
    """All 17 features must appear in the ranked list."""
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(50, N_FEATURES))
    result = compute_feature_importance(shap_values, list(FEATURE_NAMES))

    assert len(result["features"]) == N_FEATURES
    names = {f["feature"] for f in result["features"]}
    assert names == set(FEATURE_NAMES)


def test_feature_importance_mean_abs_shap_nonnegative() -> None:
    """Mean of absolute values is non-negative by construction."""
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(50, N_FEATURES))
    result = compute_feature_importance(shap_values, list(FEATURE_NAMES))

    for entry in result["features"]:
        assert entry["mean_abs_shap"] >= 0.0


def test_ranks_are_dense_one_indexed_unique() -> None:
    """Ranks must be 1..N_FEATURES with no gaps or duplicates."""
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(50, N_FEATURES))
    result = compute_feature_importance(shap_values, list(FEATURE_NAMES))
    ranks = [f["rank"] for f in result["features"]]
    assert ranks == list(range(1, N_FEATURES + 1))


def test_top_feature_matches_largest_mean_abs() -> None:
    """The hand-computed argmax of mean |shap| must match rank 1."""
    rng = np.random.default_rng(0)
    shap_values = rng.normal(size=(50, N_FEATURES))
    # Make feature 7 unambiguously dominant
    shap_values[:, 7] *= 10.0
    result = compute_feature_importance(shap_values, list(FEATURE_NAMES))
    assert result["features"][0]["feature"] == FEATURE_NAMES[7]
    assert result["features"][0]["rank"] == 1


def test_rejects_mismatched_feature_name_length() -> None:
    shap_values = np.zeros((10, N_FEATURES))
    with pytest.raises(ValueError, match="feature names"):
        compute_feature_importance(shap_values, ["only_one_name"])


def test_rejects_non_2d_input() -> None:
    shap_values = np.zeros((N_FEATURES,))
    with pytest.raises(ValueError, match="2-D"):
        compute_feature_importance(shap_values, list(FEATURE_NAMES))
