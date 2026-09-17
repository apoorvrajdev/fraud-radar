"""Phase 5D — what every result is reported with, besides the result itself.

The methodology reports each result with the conditions it was measured under
(decision 13) and counts the features that could contribute to it (decision 6).
These tests pin how each condition is computed from the folds and the
confusion matrix it describes, that the label-delay note is the decision
record's own wording, and that liveness is decided on the training fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation import ConfusionAtThreshold, confusion_at_threshold
from ml.reporting import LABEL_DELAY_NOTE, constant_features, live_features, result_context

METHODOLOGY = (
    Path(__file__).resolve().parents[3] / "docs" / "adr" / "PHASE_5D_BENCHMARK_METHODOLOGY.md"
)

# Fraud counts differ in every fold, so a fold read in place of another shows.
TRAIN_LABELS = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0])
VAL_LABELS = np.array([0, 0, 0, 0, 1, 0])
TEST_LABELS = np.array([0, 1, 0, 0, 1, 0, 0, 0])
TEST_SCORES = np.array([0.9, 0.8, 0.2, 0.6, 0.3, 0.1, 0.4, 0.05])
OPERATING_THRESHOLD = 0.5


def _context(
    test_labels: np.ndarray = TEST_LABELS, test_scores: np.ndarray = TEST_SCORES
) -> dict[str, object]:
    return result_context(
        train_labels=TRAIN_LABELS,
        val_labels=VAL_LABELS,
        test_labels=test_labels,
        at_operating_threshold=confusion_at_threshold(
            test_labels, test_scores, OPERATING_THRESHOLD
        ),
    ).to_dict()


def test_each_fold_contributes_its_own_fraud_count() -> None:
    assert _context()["fraud_counts"] == {"train": 4, "val": 1, "test": 2}


def test_test_prevalence_is_the_test_fold_fraud_rate() -> None:
    assert _context()["test_prevalence"] == 2 / 8


def test_the_realised_test_fpr_counts_legitimate_rows_at_or_above_the_threshold() -> None:
    """Six legitimate test rows; those scored 0.9 and 0.6 clear the 0.5 threshold."""
    context = _context()

    assert context["realised_fpr_on_test_at_operating_threshold"] == 2 / 6


def test_a_legitimate_row_scored_exactly_at_the_threshold_counts_as_flagged() -> None:
    scores = TEST_SCORES.copy()
    scores[2] = OPERATING_THRESHOLD

    assert _context(test_scores=scores)["realised_fpr_on_test_at_operating_threshold"] == 3 / 6


def test_without_legitimate_test_rows_the_realised_fpr_is_undefined() -> None:
    labels = np.ones(3, dtype=np.int64)
    context = _context(test_labels=labels, test_scores=np.array([0.9, 0.1, 0.6]))

    assert context["realised_fpr_on_test_at_operating_threshold"] is None
    assert context["test_prevalence"] == 1.0


def test_a_confusion_matrix_from_other_rows_is_refused() -> None:
    val_confusion = confusion_at_threshold(VAL_LABELS, np.linspace(0, 1, 6), OPERATING_THRESHOLD)

    with pytest.raises(ValueError, match="not computed on this test fold"):
        result_context(
            train_labels=TRAIN_LABELS,
            val_labels=VAL_LABELS,
            test_labels=TEST_LABELS,
            at_operating_threshold=val_confusion,
        )


def test_a_confusion_matrix_of_the_same_size_but_other_labels_is_refused() -> None:
    other_labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    with pytest.raises(ValueError, match="not computed on this test fold"):
        result_context(
            train_labels=TRAIN_LABELS,
            val_labels=VAL_LABELS,
            test_labels=TEST_LABELS,
            at_operating_threshold=confusion_at_threshold(
                other_labels, TEST_SCORES, OPERATING_THRESHOLD
            ),
        )


def test_an_empty_test_fold_is_refused() -> None:
    nothing_scored = ConfusionAtThreshold(
        threshold=OPERATING_THRESHOLD,
        true_positives=0,
        false_positives=0,
        true_negatives=0,
        false_negatives=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
    )

    with pytest.raises(ValueError, match="test fold is empty"):
        result_context(
            train_labels=TRAIN_LABELS,
            val_labels=VAL_LABELS,
            test_labels=np.array([], dtype=np.int64),
            at_operating_threshold=nothing_scored,
        )


def test_the_context_serialises_to_strict_json() -> None:
    context = _context()

    assert set(context) == {
        "test_prevalence",
        "fraud_counts",
        "realised_fpr_on_test_at_operating_threshold",
        "label_delay_note",
    }
    assert json.loads(json.dumps(context, allow_nan=False)) == context


def test_the_label_delay_note_is_the_methodology_wording() -> None:
    assert _context()["label_delay_note"] == LABEL_DELAY_NOTE

    decision_record = METHODOLOGY.read_text(encoding="utf-8")
    wording = LABEL_DELAY_NOTE[0].lower() + LABEL_DELAY_NOTE[1:].removesuffix(".")
    assert f"a label-delay note: {wording}." in decision_record


# ---------------------------------------------------------------------------
# Live features
# ---------------------------------------------------------------------------

NAMES = ["varies", "constant_everywhere", "constant_in_training", "two_values"]


def _matrix() -> tuple[np.ndarray, np.ndarray]:
    """Four training rows, then two later rows; returns (training fold, whole matrix)."""
    whole = np.array(
        [
            [0.1, 5.0, 1.0, 0.0],
            [0.7, 5.0, 1.0, 1.0],
            [0.3, 5.0, 1.0, 0.0],
            [0.9, 5.0, 1.0, 0.0],
            [0.2, 5.0, 4.0, 1.0],
            [0.4, 5.0, 1.0, 0.0],
        ]
    )
    return whole[:4], whole


def test_a_feature_is_live_when_it_varies_in_the_training_fold() -> None:
    training, whole = _matrix()

    live = live_features(training_fold=training, whole_matrix=whole, feature_names=NAMES)

    assert live.live_count == 2
    assert live.constant == ("constant_everywhere", "constant_in_training")


def test_the_whole_matrix_count_is_kept_beside_the_training_fold_count() -> None:
    """A column constant in training that varies later is live over the whole matrix only."""
    training, whole = _matrix()

    live = live_features(training_fold=training, whole_matrix=whole, feature_names=NAMES)

    assert live.live_count_whole_matrix == 3
    assert live.constant_whole_matrix == ("constant_everywhere",)


def test_constant_features_are_named_in_feature_order() -> None:
    training, whole = _matrix()
    reordered = [3, 2, 1, 0]
    names = [NAMES[index] for index in reordered]

    live = live_features(
        training_fold=training[:, reordered],
        whole_matrix=whole[:, reordered],
        feature_names=names,
    )

    assert live.constant == ("constant_in_training", "constant_everywhere")


@pytest.mark.parametrize(
    ("column", "is_live"),
    [
        pytest.param([0.0, -0.0, 0.0], False, id="signed-zeros-are-one-value"),
        pytest.param([np.nan, np.nan, np.nan], False, id="all-nan-is-one-value"),
        pytest.param([np.nan, 1.0, 1.0], True, id="nan-and-a-number-are-two-values"),
        pytest.param([999.0, 999.0, 998.0], True, id="one-differing-row-is-enough"),
    ],
)
def test_distinct_values_decide_liveness(column: list[float], is_live: bool) -> None:
    matrix = np.array(column).reshape(-1, 1)

    live = live_features(training_fold=matrix, whole_matrix=matrix, feature_names=["f"])

    assert live.live_count == int(is_live)
    assert live.constant == (() if is_live else ("f",))


@pytest.mark.parametrize("fold", ["training_fold", "whole_matrix"])
def test_a_matrix_that_does_not_match_the_feature_names_is_refused(fold: str) -> None:
    training, whole = _matrix()
    matrices = {"training_fold": training, "whole_matrix": whole}
    matrices[fold] = matrices[fold][:, :3]

    with pytest.raises(ValueError, match="feature names were given"):
        live_features(**matrices, feature_names=NAMES)


def test_constant_features_apply_the_liveness_test_to_any_rows() -> None:
    """The same test as a training fold's, over rows that are not one — a later fold, say."""
    _, whole = _matrix()
    later_rows = whole[4:]

    assert constant_features(later_rows, NAMES) == ("constant_everywhere",)
    assert constant_features(whole[:4], NAMES) == live_features(
        training_fold=whole[:4], whole_matrix=whole, feature_names=NAMES
    ).constant


def test_constant_features_refuse_a_matrix_that_does_not_match_the_names() -> None:
    _, whole = _matrix()

    with pytest.raises(ValueError, match="feature names were given"):
        constant_features(whole[:, :3], NAMES)
