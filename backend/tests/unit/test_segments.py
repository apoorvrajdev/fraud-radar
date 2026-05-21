"""Tests for the country-bucketed segment analysis."""
from __future__ import annotations

import numpy as np
import pytest

from ml.analysis.segments import (
    BUCKET_ORDER,
    bucket_for_country,
    compute_segment_metrics,
)

# All countries we expect to encounter in our synthetic / test traffic.
ALL_TEST_COUNTRIES: list[str] = [
    # US bucket
    "US",
    # Developed bucket
    "GB", "CA", "AU", "DE", "FR", "JP", "NL", "SE", "CH",
    # High-fraud bucket
    "RU", "CN", "NG", "RO", "VE", "ID",
    # Other bucket — countries from the synthetic generator that fit none of the above
    "BR", "IN", "MX", "AR", "ZA", "TH", "VN",
]


def test_bucket_us_falls_in_us_segment() -> None:
    assert bucket_for_country("US") == "US"


def test_bucket_unknown_country_falls_in_other() -> None:
    assert bucket_for_country("BR") == "Other"
    assert bucket_for_country("AR") == "Other"
    assert bucket_for_country("XX") == "Other"  # unknown ISO code


def test_high_fraud_country_routes_to_high_fraud_bucket() -> None:
    for c in ("RU", "CN", "NG", "RO", "VE", "ID"):
        assert bucket_for_country(c) == "High-fraud", c


def test_developed_country_routes_to_developed_bucket() -> None:
    for c in ("GB", "CA", "AU", "DE", "FR", "JP", "NL", "SE", "CH"):
        assert bucket_for_country(c) == "Developed", c


@pytest.mark.parametrize("country", ALL_TEST_COUNTRIES)
def test_buckets_are_mutually_exclusive(country: str) -> None:
    """Every country lands in exactly one bucket — no leaks, no doubles."""
    bucket = bucket_for_country(country)
    assert bucket in BUCKET_ORDER, f"{country} → unknown bucket {bucket!r}"
    # Mutual exclusivity is implicit because bucket_for_country returns one
    # string; we verify the union by routing every country once and checking
    # the result space is exactly BUCKET_ORDER.
    assert BUCKET_ORDER == ["US", "Developed", "Other", "High-fraud"]


def test_collectively_exhaustive_over_test_countries() -> None:
    buckets = {bucket_for_country(c) for c in ALL_TEST_COUNTRIES}
    assert buckets == set(BUCKET_ORDER)


def test_segment_metrics_skips_segments_with_too_few_frauds() -> None:
    """A segment with < 5 positive examples emits null metrics + a skip note."""
    rng = np.random.default_rng(0)
    n_per_segment = 100

    y_true_us = (rng.random(n_per_segment) < 0.10).astype(int)  # ~10 frauds — kept
    y_true_dev = np.zeros(n_per_segment, dtype=int)             # 0 frauds — skipped
    y_true_other = (rng.random(n_per_segment) < 0.03).astype(int)
    y_true_hf = (rng.random(n_per_segment) < 0.10).astype(int)

    y_true = np.concatenate([y_true_us, y_true_dev, y_true_other, y_true_hf])
    y_score = rng.random(len(y_true))
    countries = (
        ["US"] * n_per_segment
        + ["GB"] * n_per_segment
        + ["BR"] * n_per_segment
        + ["RU"] * n_per_segment
    )

    result = compute_segment_metrics(y_true, y_score, countries)
    assert "skipped_reason" in result["segments"]["Developed"]
    assert "too few positives" in result["segments"]["Developed"]["skipped_reason"]
    assert result["segments"]["Developed"]["pr_auc"] is None
    assert result["segments"]["Developed"]["recall_at_1pct_fpr"] is None
    # US has ~10 frauds → metrics computed
    assert result["segments"]["US"]["pr_auc"] is not None


def test_segment_with_zero_negatives_marks_metrics_undefined() -> None:
    """A bucket containing only positives has no defined PR-AUC or Recall@FPR.

    Mirrors the synthetic-dataset scenario where the High-fraud bucket sees
    only fraud-injected transactions and no organic traffic — sklearn would
    silently return 1.0 / 0.0, which we must not pass off as a measurement.
    """
    rng = np.random.default_rng(0)
    # High-fraud bucket: 16 rows, all positives, no negatives
    # US bucket: balanced, 50 negatives + 5 positives — enough to be measured
    y_true = np.concatenate([np.ones(16), np.zeros(50), np.ones(5)]).astype(int)
    y_score = rng.random(len(y_true))
    countries = ["RU"] * 16 + ["US"] * 55

    result = compute_segment_metrics(y_true, y_score, countries)
    hf = result["segments"]["High-fraud"]
    assert hf["n_transactions"] == 16
    assert hf["n_frauds"] == 16
    assert hf["n_negatives"] == 0
    assert hf["pr_auc"] is None
    assert hf["recall_at_1pct_fpr"] is None
    assert "skipped_reason" in hf
    assert "no negative" in hf["skipped_reason"].lower()

    # The measurable US bucket still gets metrics
    us = result["segments"]["US"]
    assert us["n_negatives"] == 50
    assert us["pr_auc"] is not None


def test_segment_metrics_global_block_matches_full_set() -> None:
    """The `global` block must aggregate over every transaction."""
    rng = np.random.default_rng(1)
    n = 200
    y_true = (rng.random(n) < 0.10).astype(int)
    y_score = rng.random(n)
    countries = ["US"] * 50 + ["GB"] * 50 + ["BR"] * 50 + ["RU"] * 50
    result = compute_segment_metrics(y_true, y_score, countries)
    assert result["global"]["n_transactions"] == n
    assert result["global"]["n_frauds"] == int(y_true.sum())


def test_segment_metrics_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_segment_metrics(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            ["US", "GB", "BR"],  # one extra country
        )
