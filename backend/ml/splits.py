"""Chronological train/val/test splitter for time-ordered fraud data.

Fraud distributions drift. Random splits leak future information into the
training fold and inflate metrics. Splitting by timestamp models the real
deployment shape: train on the past, predict the future.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays into the underlying sorted-by-time array."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def sizes(self) -> tuple[int, int, int]:
        return (len(self.train), len(self.val), len(self.test))


def chronological_split(
    timestamps: list[datetime] | np.ndarray,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitIndices:
    """Return chronological 70/15/15 index arrays over `timestamps`.

    The returned arrays are indices into the *time-sorted* order of the input.
    Caller must reorder X/y by the same sort before indexing.

    No shuffling occurs. Earlier timestamps → train, middle → val, later → test.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0; got "
            f"{train_frac} + {val_frac} + {test_frac} = "
            f"{train_frac + val_frac + test_frac}"
        )

    n = len(timestamps)
    if n < 3:
        raise ValueError(f"Need at least 3 rows to split; got {n}")

    ts_array = np.asarray(timestamps)
    sort_order = np.argsort(ts_array, kind="stable")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # Test absorbs any rounding remainder so all rows are accounted for
    train_idx = sort_order[:n_train]
    val_idx = sort_order[n_train : n_train + n_val]
    test_idx = sort_order[n_train + n_val :]

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def assert_no_temporal_leakage(
    timestamps: np.ndarray,
    splits: SplitIndices,
) -> None:
    """Raise AssertionError if any train ts > any val ts, etc.

    Defensive check for the orchestrator — cheap, catches sort bugs early.
    """
    train_max = timestamps[splits.train].max() if len(splits.train) else None
    val_min = timestamps[splits.val].min() if len(splits.val) else None
    val_max = timestamps[splits.val].max() if len(splits.val) else None
    test_min = timestamps[splits.test].min() if len(splits.test) else None

    if train_max is not None and val_min is not None:
        assert train_max <= val_min, (
            f"Temporal leakage: train_max={train_max} > val_min={val_min}"
        )
    if val_max is not None and test_min is not None:
        assert val_max <= test_min, (
            f"Temporal leakage: val_max={val_max} > test_min={test_min}"
        )
