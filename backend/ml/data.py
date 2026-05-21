"""Load training data from the DB and run the production feature extractor.

This module pays a deliberate performance cost: extracting features for the
full 50k-row training set takes several minutes because each call hits SQLite
multiple times for velocity windows. The benefit is zero train/inference skew
— the same code path that scores live transactions is the one that produces
the training matrix.

Decimal → float boundary lives inside `FeatureExtractor.extract()`. Everything
in this file works in numpy float64 land; Decimal never crosses the line.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.fraud import FEATURE_NAMES, FeatureExtractor
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelledDataset:
    """The full extracted training matrix with parallel labels and timestamps."""

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    transaction_ids: list[str]
    feature_names: list[str]

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def fraud_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0


def load_labelled_dataset(
    db: Session,
    *,
    fraud_labels_by_pattern: bool = False,
    limit: int | None = None,
) -> LabelledDataset:
    """Materialise (X, y, timestamps, ids) for every transaction in the DB.

    The label is derived from the synthetic dataset's `fraud_pattern` column
    on the Transaction table. Because the schema does not carry that column
    directly (it's stored in the CSV companion file), we treat any transaction
    whose `fraud_decision` is missing AND whose `is_card_present` shape matches
    the synthetic injection patterns as legitimate. To keep things simple and
    correct for the portfolio scope, the label is recovered from the synthetic
    CSV alongside the DB — see `load_labels_from_csv` below.
    """
    raise NotImplementedError(
        "Direct DB label recovery is not used — call load_dataset_with_csv_labels."
    )


def load_dataset_with_csv_labels(
    db: Session,
    csv_path: str | None = None,
    *,
    limit: int | None = None,
) -> LabelledDataset:
    """Build (X, y, timestamps, ids) by joining DB-extracted features with CSV labels.

    The synthetic generator writes ground-truth `is_fraud` labels to a CSV at
    `backend/ml/data/synthetic_transactions.csv`. The DB holds the operational
    rows without label leakage. We join them on transaction `id` so features
    come from the production feature extractor (DB-backed) and labels come
    from the synthetic CSV.
    """
    import csv
    from pathlib import Path

    csv_file = Path(csv_path) if csv_path else Path("ml/data/synthetic_transactions.csv")
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Labels CSV not found at {csv_file}. "
            f"Run `uv run python -m ml.generate_dataset` first."
        )

    log.info("Loading labels from %s", csv_file)
    labels_by_id: dict[str, int] = {}
    with csv_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_by_id[row["id"]] = 1 if row["is_fraud"].lower() == "true" else 0
    log.info("Loaded %d labels", len(labels_by_id))

    # Pre-fetch customers and merchants into dicts so the feature extractor
    # doesn't pay a DB hit per transaction for those lookups.
    log.info("Pre-fetching customers and merchants...")
    customers = {c.id: c for c in db.execute(select(Customer)).scalars()}
    merchants = {m.id: m for m in db.execute(select(Merchant)).scalars()}
    log.info("Loaded %d customers, %d merchants", len(customers), len(merchants))

    # Order matters less than determinism here; we sort by created_at later
    # during the chronological split, but loading in id order is reproducible.
    stmt = select(Transaction).order_by(Transaction.created_at)
    if limit is not None:
        stmt = stmt.limit(limit)
    log.info("Streaming transactions and extracting features (this takes a while)...")

    extractor = FeatureExtractor()
    rows_X: list[list[float]] = []
    rows_y: list[int] = []
    rows_ts: list[datetime] = []
    rows_ids: list[str] = []
    skipped = 0

    for i, tx in enumerate(db.execute(stmt).scalars()):
        if tx.id not in labels_by_id:
            skipped += 1
            continue
        customer = customers.get(tx.customer_id)
        merchant = merchants.get(tx.merchant_id)
        if customer is None or merchant is None:
            skipped += 1
            continue
        features = extractor.extract(db, tx, customer=customer, merchant=merchant)
        rows_X.append(features.values)
        rows_y.append(labels_by_id[tx.id])
        rows_ts.append(tx.created_at)
        rows_ids.append(tx.id)
        if (i + 1) % 5000 == 0:
            log.info("    %d rows processed...", i + 1)

    log.info("Extraction complete: %d rows kept, %d skipped", len(rows_X), skipped)

    return LabelledDataset(
        X=np.asarray(rows_X, dtype=np.float64),
        y=np.asarray(rows_y, dtype=np.int64),
        timestamps=np.asarray(rows_ts),
        transaction_ids=rows_ids,
        feature_names=list(FEATURE_NAMES),
    )
