"""What a benchmark result is reported with, besides the result itself.

A PR-AUC cannot be read without the positive rate it was measured at, and a
test fold with three frauds supports a different claim than one with three
thousand. A recall read at a threshold chosen on the test fold is not what a
model reaches at a threshold fixed in advance. The Phase 5D methodology
therefore reports every result with the conditions it was measured under, and
with where each threshold came from.

Nothing here is a performance metric, and nothing here selects, tunes or fits
anything. Every function reads labels or a confusion matrix that has already
been computed, after every choice about the model has been made.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml.evaluation import ConfusionAtThreshold

# The methodology's own wording (Phase 5D decision 13). A test pins it to the
# decision record, so the two cannot drift apart.
LABEL_DELAY_NOTE = (
    "The chronological split treats every training label as known at the val boundary, "
    "whereas real fraud labels arrive days to months later through chargebacks and "
    "investigations, so the results are optimistic in a way this benchmark does not measure."
)

# Where the threshold behind a thresholded result came from.
#
# OPERATING_THRESHOLD_SOURCE: the operating threshold recorded in threshold.json,
# selected on the val fold (or the fallback, when threshold.json records one)
# before the test fold was scored.
#
# TEST_ROC_CURVE_SOURCE: the highest threshold whose FPR on the test fold stays
# within a ceiling. It is read off the test fold itself, so the result is a
# point on the test ROC curve, not a result at a threshold fixed in advance.
OPERATING_THRESHOLD_SOURCE = "threshold.json"
TEST_ROC_CURVE_SOURCE = "test_roc_curve"


@dataclass(frozen=True)
class ResultContext:
    """The conditions a test result was measured under.

    `realised_fpr_on_test_at_operating_threshold` is the share of the test
    fold's legitimate rows scored at or above the operating threshold, which
    was fixed before the test fold was scored. It is None when the test fold
    has no legitimate rows, where a false-positive rate is undefined.
    """

    test_prevalence: float
    train_fraud_count: int
    val_fraud_count: int
    test_fraud_count: int
    realised_fpr_on_test_at_operating_threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_prevalence": self.test_prevalence,
            "fraud_counts": {
                "train": self.train_fraud_count,
                "val": self.val_fraud_count,
                "test": self.test_fraud_count,
            },
            "realised_fpr_on_test_at_operating_threshold": (
                self.realised_fpr_on_test_at_operating_threshold
            ),
            "label_delay_note": LABEL_DELAY_NOTE,
        }


def result_context(
    *,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    at_operating_threshold: ConfusionAtThreshold,
) -> ResultContext:
    """The context for a result scored on `test_labels` at the operating threshold.

    The realised FPR is read from `at_operating_threshold`, the confusion
    matrix the result reports, so the two always describe the same threshold
    and the same rows. A confusion matrix whose row or fraud count differs from
    `test_labels` was computed on some other rows, and is refused.
    """
    test_rows = len(test_labels)
    if test_rows == 0:
        raise ValueError("The test fold is empty, so no result was measured on it.")

    test_fraud_count = int(np.sum(test_labels))
    confusion = at_operating_threshold
    confusion_rows = (
        confusion.true_positives
        + confusion.false_positives
        + confusion.true_negatives
        + confusion.false_negatives
    )
    confusion_frauds = confusion.true_positives + confusion.false_negatives
    if (confusion_rows, confusion_frauds) != (test_rows, test_fraud_count):
        raise ValueError(
            f"The confusion matrix covers {confusion_rows} rows with {confusion_frauds} "
            f"frauds, but the test fold holds {test_rows} rows with {test_fraud_count}; "
            "it was not computed on this test fold."
        )

    negatives = confusion.false_positives + confusion.true_negatives
    return ResultContext(
        test_prevalence=test_fraud_count / test_rows,
        train_fraud_count=int(np.sum(train_labels)),
        val_fraud_count=int(np.sum(val_labels)),
        test_fraud_count=test_fraud_count,
        realised_fpr_on_test_at_operating_threshold=(
            confusion.false_positives / negatives if negatives else None
        ),
    )
