"""Phase 5D — the test-fold results training records, computed in one place.

Training writes `evaluate_test_fold`'s results into `metrics.json`, and run
verification recomputes them with the same function. These tests pin what it
computes from which labels and scores, and that training records exactly its
output.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ml import train
from ml.evaluation import confusion_at_threshold, pr_auc, recall_at_fpr, roc_auc
from ml.holdout import evaluate_test_fold
from ml.reporting import OPERATING_THRESHOLD_SOURCE, TEST_ROC_CURVE_SOURCE, result_context
from ml.tuning import TuningResult
from tests.unit.test_train_pipeline import _dataset

TRAIN = np.array([1, 0, 0, 1, 0, 0, 0, 0])
VAL = np.array([0, 1, 0, 0])
TEST = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
SCORES = np.array([0.1, 0.9, 0.4, 0.8, 0.3, 0.2, 0.7, 0.05, 0.6, 0.15])


def test_each_result_is_computed_from_the_test_labels_and_scores() -> None:
    evaluation = evaluate_test_fold(
        train_labels=TRAIN,
        val_labels=VAL,
        test_labels=TEST,
        test_scores=SCORES,
        operating_threshold=0.5,
    )

    assert evaluation.pr_auc == pr_auc(TEST, SCORES)
    assert evaluation.roc_auc == roc_auc(TEST, SCORES)
    assert evaluation.recall_at_1pct_fpr == recall_at_fpr(TEST, SCORES, 0.01)[0]
    assert evaluation.recall_at_5pct_fpr == recall_at_fpr(TEST, SCORES, 0.05)[0]
    confusion = confusion_at_threshold(TEST, SCORES, 0.5)
    assert evaluation.at_operating_threshold == confusion
    assert evaluation.context == result_context(
        train_labels=TRAIN, val_labels=VAL, test_labels=TEST, at_operating_threshold=confusion
    )


def test_the_operating_threshold_is_the_one_given_never_one_found_on_test() -> None:
    evaluation = evaluate_test_fold(
        train_labels=TRAIN,
        val_labels=VAL,
        test_labels=TEST,
        test_scores=SCORES,
        operating_threshold=0.65,
    )

    assert evaluation.at_operating_threshold.threshold == 0.65
    # Legitimate rows scored 0.8: one of seven at or above 0.65.
    assert evaluation.context.realised_fpr_on_test_at_operating_threshold == 1 / 7


def test_the_metrics_entries_label_every_thresholded_result() -> None:
    metrics = evaluate_test_fold(
        train_labels=TRAIN,
        val_labels=VAL,
        test_labels=TEST,
        test_scores=SCORES,
        operating_threshold=0.5,
    ).metrics()

    assert set(metrics) == {
        "test_pr_auc",
        "test_roc_auc",
        "recall_at_1pct_fpr",
        "recall_at_5pct_fpr",
        "at_operating_threshold",
        "threshold_source",
        "context",
    }
    assert metrics["threshold_source"] == {
        "at_operating_threshold": OPERATING_THRESHOLD_SOURCE,
        "recall_at_1pct_fpr": TEST_ROC_CURVE_SOURCE,
        "recall_at_5pct_fpr": TEST_ROC_CURVE_SOURCE,
    }


def test_training_records_exactly_the_holdout_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    def stub_tune(*_: Any, **__: Any) -> TuningResult:
        return TuningResult(
            best_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3},
            best_score=0.42,
            cv_results_summary={"mean_test_score": [0.42], "std_test_score": [0.0]},
        )

    monkeypatch.setattr(train, "tune_hyperparameters", stub_tune)
    ds = _dataset()
    outcome = train.train_and_evaluate(ds, n_iter=1, cv_splits=2, target_fpr=0.05)
    splits = outcome.splits

    recomputed = evaluate_test_fold(
        train_labels=ds.y[splits.train],
        val_labels=ds.y[splits.val],
        test_labels=ds.y[splits.test],
        test_scores=outcome.test_scores,
        operating_threshold=outcome.threshold.value,
    ).metrics()

    assert outcome.metrics == {**recomputed, "best_cv_pr_auc": 0.42}
