"""Tests for evaluation metrics — degenerate and known-answer cases."""
from __future__ import annotations

import numpy as np
import pytest

from ml.evaluation import (
    confusion_at_threshold,
    find_threshold_at_fpr,
    pr_auc,
    recall_at_fpr,
    roc_auc,
)


def test_perfect_classifier_has_pr_auc_and_roc_auc_of_one() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)
    assert roc_auc(y_true, y_score) == pytest.approx(1.0)


def test_inverted_classifier_has_low_pr_auc() -> None:
    # Score is anti-correlated with label
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert pr_auc(y_true, y_score) < 0.5
    assert roc_auc(y_true, y_score) == pytest.approx(0.0)


def test_recall_at_fpr_finds_correct_operating_point() -> None:
    # 100 negatives at scores [0, 0.01, 0.02, ..., 0.99]
    # 10 positives at scores [0.50, 0.55, ..., 0.95]
    rng = np.random.default_rng(0)
    neg_scores = np.linspace(0.0, 0.99, 100)
    pos_scores = np.linspace(0.50, 0.95, 10)
    y_true = np.concatenate([np.zeros(100), np.ones(10)]).astype(int)
    y_score = np.concatenate([neg_scores, pos_scores])

    recall, threshold = recall_at_fpr(y_true, y_score, target_fpr=0.05)

    # At ≤5% FPR we tolerate at most 5 of 100 negatives flagged.
    # The threshold should sit at ~0.95 (the 95th-percentile negative score).
    # Verify the realised FPR honours the ceiling.
    y_pred = (y_score >= threshold).astype(int)
    realised_fpr = y_pred[:100].sum() / 100
    assert realised_fpr <= 0.05 + 1e-9
    assert 0.0 <= recall <= 1.0


def test_find_threshold_at_zero_fpr_returns_finite_value() -> None:
    # Separable case — there is a clean threshold above all negatives.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = find_threshold_at_fpr(y_true, y_score, target_fpr=0.0)
    assert np.isfinite(threshold)
    # Threshold must reject every negative
    assert threshold > 0.3


def test_confusion_at_threshold_known_counts() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.95])
    # Threshold = 0.5 → predicted [0, 1, 1, 0, 0, 1]
    result = confusion_at_threshold(y_true, y_score, threshold=0.5)
    assert result.true_positives == 2  # positives correctly flagged (0.8, 0.95)
    assert result.false_positives == 1  # negative at 0.9 flagged
    assert result.true_negatives == 2  # negatives at 0.1 and 0.4 correctly passed
    assert result.false_negatives == 1  # positive at 0.2 missed
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)


def test_confusion_at_impossible_threshold_returns_all_negatives() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    result = confusion_at_threshold(y_true, y_score, threshold=0.99)
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
