"""Tests for chronological splitting — no leakage, correct ratios."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from ml.splits import SplitIndices, assert_no_temporal_leakage, chronological_split


def _make_timestamps(n: int) -> list[datetime]:
    """Strictly increasing timestamps, one per minute."""
    base = datetime(2026, 1, 1, 0, 0, 0)
    return [base + timedelta(minutes=i) for i in range(n)]


def test_split_ratios_70_15_15_on_round_number() -> None:
    ts = _make_timestamps(1000)
    splits = chronological_split(ts)
    assert splits.sizes == (700, 150, 150)


def test_split_absorbs_remainder_into_test() -> None:
    # 103 → train=72, val=15, test=16 (test gets the leftover)
    ts = _make_timestamps(103)
    splits = chronological_split(ts)
    assert splits.sizes[0] == 72
    assert splits.sizes[1] == 15
    assert splits.sizes[2] == 16
    assert sum(splits.sizes) == 103


def test_train_indices_are_chronologically_earliest() -> None:
    ts = _make_timestamps(100)
    splits = chronological_split(ts)
    arr = np.asarray(ts)

    train_ts = arr[splits.train]
    val_ts = arr[splits.val]
    test_ts = arr[splits.test]

    assert train_ts.max() < val_ts.min()
    assert val_ts.max() < test_ts.min()


def test_no_index_appears_in_two_splits() -> None:
    ts = _make_timestamps(500)
    splits = chronological_split(ts)
    all_indices = np.concatenate([splits.train, splits.val, splits.test])
    assert len(np.unique(all_indices)) == len(all_indices)
    assert set(all_indices.tolist()) == set(range(500))


def test_unsorted_input_is_sorted_internally() -> None:
    # Feed in reverse order — splitter must still put earliest into train
    ts = list(reversed(_make_timestamps(100)))
    splits = chronological_split(ts)
    arr = np.asarray(ts)

    assert arr[splits.train].max() < arr[splits.val].min()
    assert arr[splits.val].max() < arr[splits.test].min()


def test_fractions_must_sum_to_one() -> None:
    ts = _make_timestamps(100)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        chronological_split(ts, train_frac=0.5, val_frac=0.2, test_frac=0.2)


def test_too_few_rows_raises() -> None:
    with pytest.raises(ValueError, match="at least 3 rows"):
        chronological_split(_make_timestamps(2))


def test_assert_no_temporal_leakage_passes_on_clean_split() -> None:
    ts = _make_timestamps(1000)
    splits = chronological_split(ts)
    assert_no_temporal_leakage(np.asarray(ts), splits)


def test_assert_no_temporal_leakage_catches_swapped_indices() -> None:
    ts = np.asarray(_make_timestamps(100))
    # Deliberately corrupt: train index pointing into the future
    bad = SplitIndices(
        train=np.array([99]),
        val=np.array([50]),
        test=np.array([60]),
    )
    with pytest.raises(AssertionError, match="Temporal leakage"):
        assert_no_temporal_leakage(ts, bad)
