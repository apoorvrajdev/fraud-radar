"""Tests for calibration metrics — perfect case, hand-computed ECE, serialisable output."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ml.analysis.calibration import (
    compute_calibration_metrics,
    expected_calibration_error,
    positive_class_brier,
    positive_class_ece,
)


def test_perfect_calibration_brier_is_zero() -> None:
    """When predictions exactly match labels, Brier score is 0."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = y_true.astype(float)  # 0.0 or 1.0 exactly
    metrics = compute_calibration_metrics(y_true, y_score, n_bins=10)
    assert metrics["brier_score"] == pytest.approx(0.0)


def test_perfect_calibration_ece_is_zero() -> None:
    """Predictions exactly matching labels also drive ECE to zero."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = y_true.astype(float)
    assert expected_calibration_error(y_true, y_score, n_bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_known_input() -> None:
    """Hand-computed ECE on 5 bins.

    Inputs (10 samples, 2 per bin):
        scores: [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9]
        truth:  [0,   0,   1,   0,   1,   1,   1,   1,   1,   1]

    Per bin (uniform [0,1] split into 5):
        bin 0 (0.0–0.2)  mean_pred=0.1  mean_obs=0.0  gap=0.10  weight=0.2
        bin 1 (0.2–0.4)  mean_pred=0.3  mean_obs=0.5  gap=0.20  weight=0.2
        bin 2 (0.4–0.6)  mean_pred=0.5  mean_obs=1.0  gap=0.50  weight=0.2
        bin 3 (0.6–0.8)  mean_pred=0.7  mean_obs=1.0  gap=0.30  weight=0.2
        bin 4 (0.8–1.0)  mean_pred=0.9  mean_obs=1.0  gap=0.10  weight=0.2

    ECE = 0.2 × (0.10 + 0.20 + 0.50 + 0.30 + 0.10) = 0.24
    """
    y_score = np.array([0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])
    y_true = np.array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    ece = expected_calibration_error(y_true, y_score, n_bins=5)
    assert ece == pytest.approx(0.24, abs=1e-9)


def test_calibration_metrics_serialisable() -> None:
    """The returned dict must round-trip through json.dumps without loss."""
    rng = np.random.default_rng(42)
    y_true = (rng.random(200) < 0.1).astype(int)
    y_score = rng.random(200)
    metrics = compute_calibration_metrics(y_true, y_score, n_bins=10)

    encoded = json.dumps(metrics)
    decoded = json.loads(encoded)
    assert decoded["n_test_samples"] == 200
    assert len(decoded["bin_counts"]) == 10
    assert len(decoded["bin_mean_predicted"]) == 10
    assert len(decoded["bin_mean_observed"]) == 10
    assert decoded["brier_score"] == pytest.approx(metrics["brier_score"])
    assert decoded["expected_calibration_error"] == pytest.approx(
        metrics["expected_calibration_error"]
    )


def test_bin_counts_sum_to_n_samples() -> None:
    """Every test sample must land in exactly one bin."""
    rng = np.random.default_rng(0)
    n = 137
    y_true = (rng.random(n) < 0.05).astype(int)
    y_score = rng.random(n)
    metrics = compute_calibration_metrics(y_true, y_score, n_bins=10)
    assert sum(metrics["bin_counts"]) == n


def test_positive_class_brier_excludes_negatives() -> None:
    """Brier on positives only — negatives are dropped entirely from the mean.

    With y_true=[0,0,0,0,1,1] and y_score=[0,0,0,0,0.9,0.5]:
        positive subset scores = [0.9, 0.5]
        Brier = mean((0.9-1)^2 + (0.5-1)^2) = mean(0.01 + 0.25) = 0.13
    """
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.5])
    assert positive_class_brier(y_true, y_score) == pytest.approx(0.13, abs=1e-9)


def test_positive_class_brier_in_full_metrics_payload() -> None:
    """The full metrics dict must carry positive_class_brier alongside the aggregate."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = (rng.random(n) < 0.05).astype(int)
    y_score = rng.random(n) * 0.3 + y_true * 0.5  # positives skew higher
    metrics = compute_calibration_metrics(y_true, y_score)
    assert "positive_class_brier" in metrics
    assert "positive_class_ece" in metrics
    assert metrics["positive_class_brier"] >= 0.0


def test_positive_class_ece_uses_only_bins_with_positives() -> None:
    """ECE uniformly averaged across bins that contain at least one positive.

    Construct data with positives in only two bins:
        bin 5 (0.5–0.6): one positive at 0.55  →  |0.55 - 1.0| = 0.45
        bin 8 (0.8–0.9): one positive at 0.85  →  |0.85 - 1.0| = 0.15
        every other bin contains only negatives → excluded
    Expected positive-class ECE = (0.45 + 0.15) / 2 = 0.30
    """
    y_score = np.array([0.05, 0.05, 0.05, 0.05, 0.55, 0.85])
    y_true = np.array([0, 0, 0, 0, 1, 1])
    pos_ece = positive_class_ece(y_true, y_score, n_bins=10)
    assert pos_ece == pytest.approx(0.30, abs=1e-9)


def test_worst_positive_bin_identifies_largest_gap() -> None:
    """The worst_positive_bin entry must point at the bin with the largest gap."""
    y_score = np.array([0.05, 0.05, 0.05, 0.55, 0.85])
    y_true = np.array([0, 0, 0, 1, 1])
    metrics = compute_calibration_metrics(y_true, y_score, n_bins=10)
    worst = metrics["worst_positive_bin"]
    assert worst is not None
    # Bin 5 has the larger gap (0.45) than bin 8 (0.15)
    assert worst["bin_index"] == 5
    assert worst["mean_predicted"] == pytest.approx(0.55)
    assert worst["mean_observed"] == pytest.approx(1.0)
    assert worst["n_positives"] == 1
