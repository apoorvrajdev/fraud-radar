"""Phase 5D — what every result is reported with, besides the result itself.

The methodology reports each result with the conditions it was measured under
(decision 13). These tests pin how each condition is computed from the folds
and the confusion matrix it describes, and that the label-delay note is the
decision record's own wording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.evaluation import ConfusionAtThreshold, confusion_at_threshold
from ml.reporting import LABEL_DELAY_NOTE, result_context

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
