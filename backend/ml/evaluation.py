"""Evaluation metrics for the fraud classifier.

Pure functions over (y_true, y_score) arrays — no model dependency, no DB.
Operating-point selection uses FPR rather than score percentile because the
business cost of fraud platforms is "how many real customers do we annoy"
(FPR), not "what fraction of scores do we flag".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ConfusionAtThreshold:
    """Confusion-matrix derived metrics at a fixed score threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (sklearn: average_precision_score)."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def find_threshold_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> float:
    """Return the highest score threshold whose FPR does not exceed target.

    Walk the ROC curve from strictest threshold downward; pick the last
    operating point still at-or-under target_fpr. Returns +inf if no
    operating point qualifies (extremely rare; means every prediction
    triggers a false positive).
    """
    fpr, _tpr, thresholds = roc_curve(y_true, y_score)
    # roc_curve returns FPR ascending; find the largest index with FPR <= target
    qualifying = np.where(fpr <= target_fpr)[0]
    if len(qualifying) == 0:
        return float("inf")
    # Take the last qualifying point — that's the most lenient threshold
    # we can use while still respecting the FPR ceiling.
    return float(thresholds[qualifying[-1]])


def recall_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> tuple[float, float]:
    """Return (recall, threshold) at the operating point whose FPR ≤ target."""
    threshold = find_threshold_at_fpr(y_true, y_score, target_fpr)
    if not np.isfinite(threshold):
        return (0.0, threshold)
    y_pred = (y_score >= threshold).astype(int)
    return (float(recall_score(y_true, y_pred, zero_division=0)), threshold)


def confusion_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> ConfusionAtThreshold:
    """Return TP/FP/TN/FN and derived precision/recall/F1 at a fixed threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ConfusionAtThreshold(
        threshold=float(threshold),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def save_pr_curve_png(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    *,
    title: str = "Precision-Recall curve",
) -> None:
    """Render and save a precision-recall curve PNG.

    Matplotlib is imported lazily so unit tests that don't need plotting
    avoid the import cost.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = pr_auc(y_true, y_score)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title}  (PR-AUC = {ap:.4f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
