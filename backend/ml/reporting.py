"""What a benchmark result is reported with, besides the result itself.

A PR-AUC cannot be read without the positive rate it was measured at, and a
test fold with three frauds supports a different claim than one with three
thousand. A recall read at a threshold chosen on the test fold is not what a
model reaches at a threshold fixed in advance, and a feature that never varies
in training cannot contribute at all. The Phase 5D methodology therefore
reports every result with the conditions it was measured under, with where
each threshold came from, and with the features that could contribute.

Nothing here is a performance metric, and nothing here selects, tunes or fits
anything. Every function reads a feature matrix, labels or a confusion matrix
that already exists, and none of it feeds back into a choice about the model.
"""
from __future__ import annotations

from collections.abc import Sequence
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

# SEGMENT_TEST_ROC_CURVE_SOURCE: like TEST_ROC_CURVE_SOURCE, but read off the ROC
# curve of one segment's own test rows, so each segment's recall is a point on
# that segment's curve.
SEGMENT_TEST_ROC_CURVE_SOURCE = "segment_test_roc_curve"


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


@dataclass(frozen=True)
class LiveFeatures:
    """The features that could contribute to a run's model.

    A feature is live when it takes more than one distinct value in the run's
    training fold: a column that never varies in training gives the model
    nothing to learn from. `constant` names the others, in feature order. The
    `_whole_matrix` pair applies the same test to every row of the run's
    matrix, where a column constant in training may still vary.
    """

    live_count: int
    constant: tuple[str, ...]
    live_count_whole_matrix: int
    constant_whole_matrix: tuple[str, ...]


def live_features(
    *,
    training_fold: np.ndarray,
    whole_matrix: np.ndarray,
    feature_names: Sequence[str],
) -> LiveFeatures:
    """Count the live features of a run from its own feature matrix.

    Counted, never assumed: which features are constant depends on the
    dataset and on where its folds fall.
    """
    for label, matrix in (("training fold", training_fold), ("whole matrix", whole_matrix)):
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"The {label} has shape {matrix.shape}, but {len(feature_names)} "
                "feature names were given."
            )

    constant = _constant_columns(training_fold, feature_names)
    constant_whole_matrix = _constant_columns(whole_matrix, feature_names)
    return LiveFeatures(
        live_count=len(feature_names) - len(constant),
        constant=constant,
        live_count_whole_matrix=len(feature_names) - len(constant_whole_matrix),
        constant_whole_matrix=constant_whole_matrix,
    )


def constant_features(matrix: np.ndarray, feature_names: Sequence[str]) -> tuple[str, ...]:
    """Names of the features that never vary over `matrix`'s rows, in feature order.

    The test `live_features` applies to a training fold, for any set of rows —
    for instance the test fold another run's model is measured on.
    """
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(
            f"The matrix has shape {matrix.shape}, but {len(feature_names)} feature names "
            "were given."
        )
    return _constant_columns(matrix, feature_names)


def _constant_columns(matrix: np.ndarray, feature_names: Sequence[str]) -> tuple[str, ...]:
    """Names of the columns holding at most one distinct value.

    Values that compare equal are one value, and so are all NaNs.
    """
    return tuple(
        name
        for index, name in enumerate(feature_names)
        if np.unique(matrix[:, index]).size <= 1
    )
