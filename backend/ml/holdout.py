"""The results recorded for a run's test fold, computed in one place.

Training scores the test fold once and records these results in
`metrics.json`. Analysis later checks a run by computing them again from the
saved model's scores. Both call `evaluate_test_fold`, so the check compares
the recorded numbers with training's own definitions rather than with a copy
of them.

Nothing here selects anything. The operating threshold comes in already fixed
on the val fold, and every result is computed from test-fold labels and
scores, apart from the train and val fraud counts reported beside them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.evaluation import (
    ConfusionAtThreshold,
    confusion_at_threshold,
    pr_auc,
    recall_at_fpr,
    roc_auc,
)
from ml.reporting import (
    OPERATING_THRESHOLD_SOURCE,
    TEST_ROC_CURVE_SOURCE,
    ResultContext,
    result_context,
)


@dataclass(frozen=True)
class HoldoutEvaluation:
    """A model's results on the test fold, and the conditions they were measured under.

    `recall_at_1pct_fpr` and `recall_at_5pct_fpr` are points on the test ROC
    curve. `at_operating_threshold` is measured at the threshold chosen on the
    val fold.
    """

    pr_auc: float
    roc_auc: float
    recall_at_1pct_fpr: float
    recall_at_5pct_fpr: float
    at_operating_threshold: ConfusionAtThreshold
    context: ResultContext

    def metrics(self) -> dict[str, object]:
        """The test-fold entries of `metrics.json`."""
        return {
            "test_pr_auc": self.pr_auc,
            "test_roc_auc": self.roc_auc,
            "recall_at_1pct_fpr": self.recall_at_1pct_fpr,
            "recall_at_5pct_fpr": self.recall_at_5pct_fpr,
            "at_operating_threshold": self.at_operating_threshold.as_dict(),
            "threshold_source": {
                "at_operating_threshold": OPERATING_THRESHOLD_SOURCE,
                "recall_at_1pct_fpr": TEST_ROC_CURVE_SOURCE,
                "recall_at_5pct_fpr": TEST_ROC_CURVE_SOURCE,
            },
            # The conditions the numbers above were measured under, not further metrics.
            "context": self.context.to_dict(),
        }


def evaluate_test_fold(
    *,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    operating_threshold: float,
) -> HoldoutEvaluation:
    """Evaluate `test_scores` against `test_labels` at a threshold fixed beforehand."""
    recall_at_1pct, _ = recall_at_fpr(test_labels, test_scores, target_fpr=0.01)
    recall_at_5pct, _ = recall_at_fpr(test_labels, test_scores, target_fpr=0.05)
    confusion = confusion_at_threshold(test_labels, test_scores, operating_threshold)
    return HoldoutEvaluation(
        pr_auc=pr_auc(test_labels, test_scores),
        roc_auc=roc_auc(test_labels, test_scores),
        recall_at_1pct_fpr=recall_at_1pct,
        recall_at_5pct_fpr=recall_at_5pct,
        at_operating_threshold=confusion,
        context=result_context(
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            at_operating_threshold=confusion,
        ),
    )
