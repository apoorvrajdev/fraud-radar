"""Batch feature extraction for benchmark datasets.

The invariant this module exists to protect: **there is one implementation of
the 17 features, and it is the production one.** `build_feature_matrix` does
not compute anything. It arranges each transaction's history the way the live
scoring service arranges it, then calls the same
`FeatureExtractor.extract()` the API calls, and collects what comes back.

The live path (`app/services/scoring._load_context`) loads a customer's
transactions in `[tx.created_at - 180d, tx.created_at)` and hands them to
`extract(..., recent_transactions=...)`, which then aggregates in memory and
never touches the session. This module reproduces exactly that: a per-customer
window of prior transactions, evicted at the same boundary, passed through the
same keyword.

Two consequences worth stating plainly:

* The `db` argument is a sentinel that raises on any attribute access. If a
  future change made `extract()` fall back to SQL, this path would fail loudly
  rather than silently produce different numbers.
* Matching the *serving* path means this differs from the current synthetic
  training path, which calls `extract()` without `recent_transactions` and so
  queries unbounded history. The only feature affected is `days_since_last_tx`,
  and only for gaps longer than the 180-day window. That difference is proven
  and pinned by the parity tests rather than left as a surprise.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import timedelta
from typing import Any, cast

import numpy as np
from sqlalchemy.orm import Session

from app.fraud.feature_spec import DEFAULT_FEATURESET, feature_names
from app.fraud.features import FeatureExtractor
from app.models.transaction import Transaction
from ml.data import LabelledDataset
from ml.datasets.base import CanonicalDataset, DatasetContractError

log = logging.getLogger("ml.features.batch")

# Must equal `app.services.scoring._RECENT_HISTORY_WINDOW`. Deliberately not
# imported: the offline path does not depend on the serving package. A test
# asserts the two are identical, so drift fails CI instead of quietly changing
# what "recent" means on one side only.
HISTORY_WINDOW = timedelta(days=180)

_PROGRESS_EVERY = 50_000


class BatchExtractionError(DatasetContractError):
    """Batch extraction could not proceed, or tried to reach a database."""


class _NoDatabase:
    """Stands in for the Session that batch extraction must never use.

    `extract()` takes a Session positionally and only uses it when the caller
    withholds customer, merchant or history. Since this path supplies all
    three, any access here means that contract broke.
    """

    def __getattr__(self, name: str) -> Any:
        raise BatchExtractionError(
            f"Batch feature extraction attempted a database access ({name!r}). "
            "Every lookup must be satisfied from the canonical dataset; a SQL "
            "fallback would compute features under different semantics."
        )


def build_feature_matrix(
    dataset: CanonicalDataset,
    *,
    featureset: str = DEFAULT_FEATURESET,
    history_window: timedelta = HISTORY_WINDOW,
    progress_every: int = _PROGRESS_EVERY,
) -> LabelledDataset:
    """Extract features for every transaction, in the dataset's own order.

    Rows come out in the canonical chronological order the dataset guarantees,
    so a row index maps to a transaction id and back. Labels stay in their own
    array; they are never attached to a transaction object.
    """
    names = feature_names(featureset)
    dataset.validate()

    extractor = FeatureExtractor()
    no_db = cast(Session, _NoDatabase())

    # One bounded window per customer. Entries are references into the dataset,
    # so the windows cost pointers rather than copies of the transactions.
    windows: dict[str, deque[Transaction]] = {}

    rows: list[list[float]] = []
    labels: list[int] = []
    timestamps: list[Any] = []
    ids: list[str] = []

    for index, tx in enumerate(dataset.transactions, start=1):
        customer = dataset.customers.get(tx.customer_id)
        merchant = dataset.merchants.get(tx.merchant_id)
        if customer is None or merchant is None:  # pragma: no cover - validate() covers it
            raise BatchExtractionError(
                f"Transaction {tx.id!r} has no customer or merchant in the dataset."
            )

        window = windows.setdefault(tx.customer_id, deque())
        cutoff = tx.created_at - history_window
        while window and window[0].created_at < cutoff:
            window.popleft()

        features = extractor.extract(
            no_db,
            tx,
            customer=customer,
            merchant=merchant,
            recent_transactions=list(window),
        )
        if len(features.values) != len(names):
            raise BatchExtractionError(
                f"Extractor returned {len(features.values)} values but featureset "
                f"{featureset!r} declares {len(names)}."
            )

        rows.append(features.values)
        labels.append(dataset.labels[tx.id])
        timestamps.append(tx.created_at)
        ids.append(tx.id)

        # Append after extracting: a transaction is never part of its own history.
        window.append(tx)

        if progress_every and index % progress_every == 0:
            log.info("    %d/%d rows extracted...", index, dataset.n_rows)

    log.info("Extracted %d rows x %d features (%s)", len(rows), len(names), featureset)

    return LabelledDataset(
        X=np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)),
        y=np.asarray(labels, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=object),
        transaction_ids=ids,
        feature_names=names,
    )
