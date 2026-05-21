"""Country-bucketed segment analysis for the fraud model.

Segments are mutually exclusive and collectively exhaustive — every
country routes to exactly one bucket. The buckets measure performance
stability across geographic distributions, not demographic fairness;
the synthetic dataset carries no protected attributes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

# Bucket membership tables. The order matters for the validation pass:
# `_DEVELOPED` and `_HIGH_FRAUD` are checked before falling through to "Other".
_US_BUCKET = "US"
_DEVELOPED_BUCKET = "Developed"
_OTHER_BUCKET = "Other"
_HIGH_FRAUD_BUCKET = "High-fraud"

BUCKET_ORDER: list[str] = [
    _US_BUCKET,
    _DEVELOPED_BUCKET,
    _OTHER_BUCKET,
    _HIGH_FRAUD_BUCKET,
]

_DEVELOPED_COUNTRIES: frozenset[str] = frozenset({
    "GB", "CA", "AU", "DE", "FR", "JP", "NL", "SE", "CH",
})

_HIGH_FRAUD_COUNTRIES: frozenset[str] = frozenset({
    "RU", "CN", "NG", "RO", "VE", "ID",
})

_MIN_FRAUDS_FOR_METRIC = 5

_SKIP_REASON_FEW_POSITIVES = "too few positives (<5)"
_SKIP_REASON_NO_NEGATIVES = (
    "no negative samples — PR-AUC and Recall@FPR are undefined"
)


def bucket_for_country(country: str) -> str:
    """Route a single country code to its bucket name.

    Buckets are checked in priority order — `High-fraud` wins over `Other`
    so an unknown country that happens to also be in the high-fraud list
    is never silently demoted.
    """
    if country == "US":
        return _US_BUCKET
    if country in _HIGH_FRAUD_COUNTRIES:
        return _HIGH_FRAUD_BUCKET
    if country in _DEVELOPED_COUNTRIES:
        return _DEVELOPED_BUCKET
    return _OTHER_BUCKET


def _validate_bucketing(countries: Sequence[str]) -> None:
    """Assert every row landed in exactly one bucket — no leaks, no doubles."""
    for c in countries:
        bucket = bucket_for_country(c)
        if bucket not in BUCKET_ORDER:
            raise ValueError(f"Country {c!r} routed to unknown bucket {bucket!r}")
        # Double-membership check: a country code may not be in both the
        # developed and high-fraud lists. Asserted by construction at module
        # load — if someone later edits the constants and creates overlap,
        # this catches it.
        if c in _DEVELOPED_COUNTRIES and c in _HIGH_FRAUD_COUNTRIES:
            raise ValueError(
                f"Country {c!r} appears in both developed and high-fraud sets"
            )


def _recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Return recall at the operating point whose FPR ≤ target.

    Pulled in here rather than reused from ml.evaluation so the analysis
    layer doesn't reach back across the training-stage boundary.
    """
    fpr, _tpr, thresholds = roc_curve(y_true, y_score)
    qualifying = np.where(fpr <= target_fpr)[0]
    if len(qualifying) == 0:
        return 0.0
    threshold = thresholds[qualifying[-1]]
    y_pred = (y_score >= threshold).astype(int)
    return float((y_pred[y_true == 1].sum()) / max((y_true == 1).sum(), 1))


def _segment_block(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Compute one segment's metrics block.

    PR-AUC and Recall@FPR require BOTH positives and negatives to be defined
    — a segment composed entirely of one class returns garbage values from
    sklearn (PR-AUC=1.0 on no-negative inputs, Recall@FPR ill-defined). We
    skip explicitly in two cases and surface the reason in `skipped_reason`:

      - n_frauds < 5     → "too few positives (<5)"
      - n_negatives == 0 → "no negative samples — PR-AUC and Recall@FPR are undefined"

    When skipped, `pr_auc` and `recall_at_1pct_fpr` are explicitly `null` so
    downstream consumers can't misread garbage as a measurement.
    """
    n = int(len(y_true))
    n_frauds = int(y_true.sum())
    n_negatives = n - n_frauds
    fraud_rate = float(n_frauds / n) if n > 0 else 0.0

    base: dict[str, Any] = {
        "n_transactions": n,
        "n_frauds": n_frauds,
        "n_negatives": n_negatives,
        "fraud_rate": fraud_rate,
    }

    if n_frauds < _MIN_FRAUDS_FOR_METRIC:
        return {
            **base,
            "pr_auc": None,
            "recall_at_1pct_fpr": None,
            "skipped_reason": _SKIP_REASON_FEW_POSITIVES,
        }
    if n_negatives == 0:
        return {
            **base,
            "pr_auc": None,
            "recall_at_1pct_fpr": None,
            "skipped_reason": _SKIP_REASON_NO_NEGATIVES,
        }

    return {
        **base,
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_1pct_fpr": _recall_at_fpr(y_true, y_score, target_fpr=0.01),
    }


def compute_segment_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    countries: Sequence[str],
) -> dict[str, Any]:
    """Return the segment-metrics JSON payload.

    Includes a `global` block (the full test set) so the model card can show
    one apples-to-apples comparison line beneath the per-segment table.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    countries = list(countries)
    if len(countries) != len(y_true):
        raise ValueError(
            f"countries length {len(countries)} != y_true length {len(y_true)}"
        )

    _validate_bucketing(countries)

    bucket_of = np.array([bucket_for_country(c) for c in countries])

    segments: dict[str, Any] = {}
    for name in BUCKET_ORDER:
        mask = bucket_of == name
        segments[name] = _segment_block(y_true[mask], y_score[mask])

    return {
        "global": _segment_block(y_true, y_score),
        "segments": segments,
        "notes": (
            "Segment analysis measures performance stability across "
            "geographic distributions, not demographic fairness. The "
            "synthetic dataset has no protected attributes."
        ),
    }
