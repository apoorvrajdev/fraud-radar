"""Calibration analysis: Brier score, ECE, and the reliability diagram.

XGBoost trained with `scale_pos_weight` to handle extreme class imbalance is
systematically over-confident on the positive class. These functions quantify
the miscalibration on the held-out test set; the model card calls out that a
production deployment would apply Platt or isotonic post-hoc calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss

_DEFAULT_N_BINS = 10


def _bin_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int,
) -> tuple[list[int], list[int], list[float], list[float]]:
    """Bin predictions into n_bins uniform intervals over [0, 1].

    Returns four parallel lists of length n_bins:
        bin_counts, bin_positive_counts, bin_mean_predicted, bin_mean_observed.
    Empty bins emit count=0 and the bin midpoint / 0.0 as placeholders.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # `np.digitize` with the interior edges puts scores into 0..n_bins-1.
    # A score exactly at an edge falls into the lower bin (right=False).
    bin_idx = np.digitize(y_score, edges[1:-1], right=False)

    counts: list[int] = []
    pos_counts: list[int] = []
    mean_pred: list[float] = []
    mean_obs: list[float] = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_in_bin = int(mask.sum())
        n_pos_in_bin = int(y_true[mask].sum()) if n_in_bin else 0
        counts.append(n_in_bin)
        pos_counts.append(n_pos_in_bin)
        if n_in_bin == 0:
            mean_pred.append(float((edges[b] + edges[b + 1]) / 2))
            mean_obs.append(0.0)
        else:
            mean_pred.append(float(y_score[mask].mean()))
            mean_obs.append(float(y_true[mask].mean()))
    return counts, pos_counts, mean_pred, mean_obs


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> float:
    """Weighted-mean gap between predicted and observed rates per bin.

    ECE = Σ_b (|bin_b| / N) · |mean_pred_b - mean_observed_b|

    Empty bins contribute zero by construction.
    """
    counts, _pos_counts, mean_pred, mean_obs = _bin_predictions(y_true, y_score, n_bins)
    total = int(sum(counts))
    if total == 0:
        return 0.0
    ece = 0.0
    for n_in_bin, mp, mo in zip(counts, mean_pred, mean_obs, strict=True):
        if n_in_bin == 0:
            continue
        ece += (n_in_bin / total) * abs(mp - mo)
    return float(ece)


def positive_class_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score restricted to rows where y_true == 1.

    Strips out the well-calibrated negative class so the over-prediction on
    actual fraud cases is visible. The aggregate Brier on a 1.5%-positive
    dataset is dominated by the negative-class fit and can look excellent
    even when the model is systematically over-confident on harder cases.

    Returns 0.0 when the input contains no positives — caller is responsible
    for treating that case as "undefined" if it matters.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    mask = y_true_arr == 1
    if mask.sum() == 0:
        return 0.0
    return float(np.mean((y_score_arr[mask] - 1.0) ** 2))


def positive_class_ece(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> float:
    """ECE averaged uniformly across bins that contain at least one positive.

    The aggregate ECE is bin-size-weighted, which means the dominant
    negative-class bin (predicting near zero, observing near zero) absorbs
    most of the calibration story. This variant restricts to bins that
    contain real fraud and weights them uniformly so the calibration gap on
    high-prediction bins surfaces.

    Returns 0.0 when no bin contains a positive.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.clip(np.asarray(y_score, dtype=np.float64), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_score_arr, edges[1:-1], right=False)

    gaps: list[float] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        if int(y_true_arr[mask].sum()) == 0:
            continue
        gap = abs(float(y_score_arr[mask].mean()) - float(y_true_arr[mask].mean()))
        gaps.append(gap)
    if not gaps:
        return 0.0
    return float(np.mean(gaps))


def _worst_positive_bin(
    bin_counts: list[int],
    bin_positive_counts: list[int],
    bin_mean_predicted: list[float],
    bin_mean_observed: list[float],
) -> dict[str, Any] | None:
    """Return the bin with the largest |mean_pred - mean_obs| gap among bins
    containing at least one positive. Returns None when no such bin exists.
    """
    best_idx: int | None = None
    best_gap = -1.0
    for b, (n, n_pos, mp, mo) in enumerate(
        zip(bin_counts, bin_positive_counts, bin_mean_predicted, bin_mean_observed, strict=True)
    ):
        if n == 0 or n_pos == 0:
            continue
        gap = abs(mp - mo)
        if gap > best_gap:
            best_gap = gap
            best_idx = b
    if best_idx is None:
        return None
    return {
        "bin_index": int(best_idx),
        "mean_predicted": float(bin_mean_predicted[best_idx]),
        "mean_observed": float(bin_mean_observed[best_idx]),
        "n_samples": int(bin_counts[best_idx]),
        "n_positives": int(bin_positive_counts[best_idx]),
        "gap": float(best_gap),
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = _DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Return the calibration-metrics JSON payload.

    Includes aggregate Brier and ECE alongside positive-class variants so the
    model card can show both. The aggregate numbers look excellent on a
    fraud dataset because the dominant negative class is well-calibrated;
    the positive-class variants surface the systematic over-prediction that
    a Platt or isotonic post-hoc step would correct.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    counts, pos_counts, mean_pred, mean_obs = _bin_predictions(
        y_true_arr, y_score_arr, n_bins
    )
    brier = float(brier_score_loss(y_true_arr, y_score_arr))
    ece = expected_calibration_error(y_true_arr, y_score_arr, n_bins=n_bins)
    pos_brier = positive_class_brier(y_true_arr, y_score_arr)
    pos_ece = positive_class_ece(y_true_arr, y_score_arr, n_bins=n_bins)
    worst = _worst_positive_bin(counts, pos_counts, mean_pred, mean_obs)

    return {
        "brier_score": brier,
        "positive_class_brier": pos_brier,
        "expected_calibration_error": ece,
        "positive_class_ece": pos_ece,
        "bin_counts": counts,
        "bin_positive_counts": pos_counts,
        "bin_mean_predicted": mean_pred,
        "bin_mean_observed": mean_obs,
        "worst_positive_bin": worst,
        "n_test_samples": int(len(y_true_arr)),
        "note": (
            "Aggregate Brier and ECE are dominated by the well-calibrated "
            "negative-class bin. The positive-class variants surface the "
            "systematic over-prediction on harder cases that a Platt or "
            "isotonic post-hoc calibration step would correct."
        ),
    }


def render_calibration_plot(metrics: dict[str, Any], path: Path) -> None:
    """Save a reliability diagram PNG to `path`.

    Imports matplotlib lazily so unit-test loaders don't pay the import cost.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = metrics["bin_counts"]
    mean_pred = metrics["bin_mean_predicted"]
    mean_obs = metrics["bin_mean_observed"]
    brier = metrics["brier_score"]
    ece = metrics["expected_calibration_error"]

    # Only plot non-empty bins; empty bins clutter the line.
    pred = [p for p, c in zip(mean_pred, counts, strict=True) if c > 0]
    obs = [o for o, c in zip(mean_obs, counts, strict=True) if c > 0]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    try:
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
        ax.plot(pred, obs, marker="o", linewidth=2, label="observed")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraud rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Model calibration on held-out test set")
        ax.text(
            0.02,
            0.95,
            f"Brier = {brier:.4f}\nECE   = {ece:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)
