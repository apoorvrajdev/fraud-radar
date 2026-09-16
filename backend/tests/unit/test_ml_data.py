"""The synthetic training loader, and the provenance record for its data.

Runs against an in-memory SQLite database and a label CSV in a temporary
directory — the same two sources the real synthetic path joins. SQLite is the
point: it returns naive datetimes even for timezone-aware columns, which is
exactly the case the loader has to normalise.
"""
from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.data import (
    SYNTHETIC_DATASET_NAME,
    LabelledDataset,
    _as_utc,
    load_dataset_with_csv_labels,
    synthetic_provenance,
)
from ml.datasets.base import DataOrigin, DatasetContractError, DatasetProvenance

# Naive, as the generator writes it.
START = datetime(2026, 4, 20, 4, 28, 46, 812521)
LABELS = {"tx-0": True, "tx-1": False, "tx-2": False, "tx-3": True}


def _transaction(tx_id: str, at: datetime) -> Transaction:
    return Transaction(
        id=tx_id,
        idempotency_key=tx_id,
        customer_id="c-1",
        merchant_id="m-1",
        amount=Decimal("42.50"),
        currency="USD",
        status="APPROVED",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        created_at=at,
    )


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    session.add(
        Customer(
            id="c-1",
            email="c-1@example.invalid",
            full_name="Customer One",
            country="US",
            risk_tier="LOW",
            account_age_days=400,
        )
    )
    session.add(
        Merchant(
            id="m-1", name="Merchant One", category="GROCERY", mcc="5411",
            country="US", risk_rating="LOW",
        )
    )
    # Inserted out of time order; the loader reads them back by created_at.
    for tx_id, offset_hours in (("tx-2", 5), ("tx-0", 0), ("tx-3", 9), ("tx-1", 2)):
        session.add(_transaction(tx_id, START + timedelta(hours=offset_hours)))
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_transactions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "is_fraud"])
        writer.writeheader()
        for tx_id, is_fraud in LABELS.items():
            writer.writerow({"id": tx_id, "is_fraud": str(is_fraud)})
    return path


def test_the_database_really_returns_naive_timestamps(db_session: Session) -> None:
    """The premise of the normalisation, checked rather than assumed."""
    stored = db_session.get(Transaction, "tx-0")
    assert stored is not None
    db_session.refresh(stored)
    assert stored.created_at.tzinfo is None


def test_loaded_timestamps_are_timezone_aware_utc(
    db_session: Session, labels_csv: Path
) -> None:
    ds = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))

    assert all(ts.tzinfo is UTC for ts in ds.timestamps)


def test_normalisation_keeps_the_instant_and_the_order(
    db_session: Session, labels_csv: Path
) -> None:
    ds = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))

    assert ds.transaction_ids == ["tx-0", "tx-1", "tx-2", "tx-3"]
    assert list(ds.timestamps) == [
        (START + timedelta(hours=offset)).replace(tzinfo=UTC) for offset in (0, 2, 5, 9)
    ]
    assert list(ds.y) == [1, 0, 0, 1]


def test_a_naive_timestamp_is_read_as_utc() -> None:
    assert _as_utc(START) == START.replace(tzinfo=UTC)
    assert _as_utc(START).tzinfo is UTC


def test_an_aware_timestamp_is_converted_to_utc_not_relabelled() -> None:
    india = timezone(timedelta(hours=5, minutes=30))
    local = datetime(2026, 4, 20, 10, 0, tzinfo=india)

    converted = _as_utc(local)

    assert converted.tzinfo is UTC
    assert converted == local
    assert converted.hour == 4
    assert converted.minute == 30


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_records_the_label_csv_digest_and_nothing_else(
    db_session: Session, labels_csv: Path
) -> None:
    ds = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))

    provenance = synthetic_provenance(ds, csv_path=labels_csv)

    assert dict(provenance.files) == {
        "synthetic_transactions.csv": hashlib.sha256(labels_csv.read_bytes()).hexdigest()
    }


def test_provenance_says_the_database_has_no_digest(
    db_session: Session, labels_csv: Path
) -> None:
    ds = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))

    provenance = synthetic_provenance(ds, csv_path=labels_csv)

    assert "label CSV only" in provenance.notes
    assert "no digest" in provenance.notes
    assert any("operational database" in step for step in provenance.preprocessing)
    assert "naive database timestamps interpreted as UTC" in provenance.preprocessing


def test_provenance_describes_the_rows_that_were_used(
    db_session: Session, labels_csv: Path
) -> None:
    ds = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))

    provenance = synthetic_provenance(ds, csv_path=labels_csv)

    assert provenance.name == SYNTHETIC_DATASET_NAME
    assert provenance.origin is DataOrigin.SYNTHETIC
    assert provenance.label_field == "is_fraud"
    assert (provenance.row_count, provenance.fraud_count) == (4, 2)
    assert provenance.period_start == START.replace(tzinfo=UTC)
    assert provenance.period_end == (START + timedelta(hours=9)).replace(tzinfo=UTC)
    assert provenance.subsample is None
    assert provenance.retrieved_at is None


def test_a_row_limit_is_recorded_as_preprocessing(
    db_session: Session, labels_csv: Path
) -> None:
    limited = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv), limit=2)
    full = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))

    limited_steps = synthetic_provenance(limited, csv_path=labels_csv, limit=2).preprocessing
    full_steps = synthetic_provenance(full, csv_path=labels_csv).preprocessing

    assert "limited to the first 2 database transactions by created_at" in limited_steps
    assert not any(step.startswith("limited to") for step in full_steps)
    assert synthetic_provenance(limited, csv_path=labels_csv, limit=2).row_count == 2


def test_provenance_survives_a_round_trip(db_session: Session, labels_csv: Path) -> None:
    ds = load_dataset_with_csv_labels(db_session, csv_path=str(labels_csv))
    provenance = synthetic_provenance(ds, csv_path=labels_csv)

    assert DatasetProvenance.from_dict(provenance.to_dict()) == provenance


def test_naive_timestamps_cannot_reach_the_provenance(labels_csv: Path) -> None:
    """What the loader's normalisation prevents, refused if it ever regresses."""
    ds = LabelledDataset(
        X=np.zeros((2, 17)),
        y=np.array([0, 1]),
        timestamps=np.asarray([START, START + timedelta(hours=1)], dtype=object),
        transaction_ids=["a", "b"],
        feature_names=[f"f{index}" for index in range(17)],
    )

    with pytest.raises(DatasetContractError, match="timezone-aware"):
        synthetic_provenance(ds, csv_path=labels_csv)
