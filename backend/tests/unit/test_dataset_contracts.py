"""Phase 5A — the canonical dataset contract.

These tests exist because every failure they describe is silent in production:
an unsorted stream produces plausible-looking velocity features computed from
the future, a missing label shifts the label column against the feature matrix,
and a session-attached object turns the offline path into a database client.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.datasets.base import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalDataset,
    DataOrigin,
    DatasetAdapter,
    DatasetContractError,
    DatasetProvenance,
    Subsample,
)

ANCHOR = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
DIGEST = "a" * 64


def _customer(customer_id: str = "cust-1") -> Customer:
    return Customer(
        id=customer_id,
        email=f"{customer_id}@example.test",
        full_name="Test Customer",
        country="US",
        risk_tier="LOW",
        account_age_days=365,
    )


def _merchant(merchant_id: str = "merch-1") -> Merchant:
    return Merchant(
        id=merchant_id,
        name="Test Merchant",
        category="RETAIL",
        mcc="5999",
        country="US",
        risk_rating="LOW",
    )


def _transaction(
    tx_id: str,
    *,
    created_at: datetime,
    customer_id: str = "cust-1",
    merchant_id: str = "merch-1",
) -> Transaction:
    return Transaction(
        id=tx_id,
        idempotency_key=tx_id,
        customer_id=customer_id,
        merchant_id=merchant_id,
        amount=Decimal("100.00"),
        currency="USD",
        status="APPROVED",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        created_at=created_at,
    )


def _provenance(**overrides: object) -> DatasetProvenance:
    payload: dict[str, object] = {
        "name": "fixture",
        "version": "v1",
        "origin": DataOrigin.SYNTHETIC,
        "source_url": "https://example.test/fixture",
        "citation": "Fixture dataset, generated in-test",
        "license": "CC0",
        "label_field": "is_fraud",
        "label_definition": "1 = confirmed fraudulent authorisation",
    }
    payload.update(overrides)
    return DatasetProvenance(**payload)  # type: ignore[arg-type]


def _dataset(transactions: list[Transaction], labels: dict[str, int]) -> CanonicalDataset:
    return CanonicalDataset(
        customers={"cust-1": _customer()},
        merchants={"merch-1": _merchant()},
        transactions=transactions,
        labels=labels,
        provenance=_provenance(),
    )


# ---------------------------------------------------------------------------
# DatasetProvenance
# ---------------------------------------------------------------------------


def test_provenance_defaults_to_the_current_schema_version() -> None:
    assert _provenance().schema_version == CANONICAL_SCHEMA_VERSION


@pytest.mark.parametrize(
    "field_name", ["name", "version", "license", "label_field", "label_definition"]
)
def test_provenance_rejects_empty_required_fields(field_name: str) -> None:
    with pytest.raises(DatasetContractError, match=field_name):
        _provenance(**{field_name: "   "})


def test_provenance_rejects_more_frauds_than_rows() -> None:
    with pytest.raises(DatasetContractError, match="exceeds row_count"):
        _provenance(row_count=10, fraud_count=11)


def test_provenance_rejects_negative_counts() -> None:
    with pytest.raises(DatasetContractError, match="row_count"):
        _provenance(row_count=-1)


def test_provenance_rejects_inverted_period() -> None:
    with pytest.raises(DatasetContractError, match="is after"):
        _provenance(period_start=ANCHOR, period_end=ANCHOR - timedelta(days=1))


def test_provenance_rejects_naive_timestamps() -> None:
    with pytest.raises(DatasetContractError, match="timezone-aware"):
        _provenance(period_start=datetime(2026, 3, 1, 12, 0))


def test_provenance_rejects_a_non_sha256_file_digest() -> None:
    with pytest.raises(DatasetContractError, match="SHA-256"):
        _provenance(files={"data.csv": "not-a-digest"})


def test_provenance_accepts_a_valid_digest() -> None:
    assert _provenance(files={"data.csv": DIGEST}).files["data.csv"] == DIGEST


def test_provenance_file_map_cannot_be_mutated_after_construction() -> None:
    files = {"data.csv": DIGEST}
    provenance = _provenance(files=files)
    files["injected.csv"] = DIGEST  # mutating the caller's dict must not leak in
    assert dict(provenance.files) == {"data.csv": DIGEST}
    with pytest.raises(TypeError):
        provenance.files["another.csv"] = DIGEST  # type: ignore[index]


def test_fraud_rate_is_derived_from_the_counts() -> None:
    assert _provenance(row_count=1000, fraud_count=15).fraud_rate == pytest.approx(0.015)


def test_fraud_rate_is_none_when_counts_are_unknown() -> None:
    assert _provenance().fraud_rate is None
    assert _provenance(row_count=0, fraud_count=0).fraud_rate is None


def test_provenance_round_trips_through_a_dict() -> None:
    original = _provenance(
        origin=DataOrigin.REAL,
        retrieved_at=ANCHOR,
        files={"data.csv": DIGEST},
        row_count=1000,
        fraud_count=15,
        period_start=ANCHOR - timedelta(days=30),
        period_end=ANCHOR,
        subsample=Subsample(strategy="cards", seed=42, max_entities=200, selected_entities=200),
        preprocessing=("amount cast to Decimal(2dp)", "timestamps normalised to UTC"),
        notes="fixture",
    )
    assert DatasetProvenance.from_dict(original.to_dict()) == original


def test_provenance_dict_is_json_ready() -> None:
    payload = _provenance(retrieved_at=ANCHOR, row_count=10, fraud_count=1).to_dict()
    assert payload["retrieved_at"] == ANCHOR.isoformat()
    assert payload["origin"] == "synthetic"
    assert payload["fraud_rate"] == pytest.approx(0.1)


def test_subsample_rejects_a_nonpositive_entity_cap() -> None:
    with pytest.raises(DatasetContractError, match="max_entities"):
        Subsample(strategy="cards", seed=42, max_entities=0)


# ---------------------------------------------------------------------------
# CanonicalDataset invariants
# ---------------------------------------------------------------------------


def test_valid_dataset_passes_and_reports_its_shape() -> None:
    txs = [
        _transaction("tx-1", created_at=ANCHOR),
        _transaction("tx-2", created_at=ANCHOR + timedelta(hours=1)),
        _transaction("tx-3", created_at=ANCHOR + timedelta(hours=2)),
    ]
    ds = _dataset(txs, {"tx-1": 0, "tx-2": 1, "tx-3": 0}).validate()

    assert ds.n_rows == 3
    assert ds.fraud_count == 1
    assert ds.fraud_rate == pytest.approx(1 / 3)
    assert ds.period == (ANCHOR, ANCHOR + timedelta(hours=2))


def test_empty_dataset_has_no_period_and_a_zero_rate() -> None:
    ds = _dataset([], {}).validate()
    assert ds.period is None
    assert ds.fraud_rate == 0.0


def test_unsorted_transactions_are_rejected() -> None:
    txs = [
        _transaction("tx-2", created_at=ANCHOR + timedelta(hours=1)),
        _transaction("tx-1", created_at=ANCHOR),
    ]
    with pytest.raises(DatasetContractError, match="sorted by created_at"):
        _dataset(txs, {"tx-1": 0, "tx-2": 0}).validate()


def test_equal_timestamps_are_allowed() -> None:
    """Ties are ordinary in transaction streams; only going backwards is not."""
    txs = [_transaction("tx-1", created_at=ANCHOR), _transaction("tx-2", created_at=ANCHOR)]
    assert _dataset(txs, {"tx-1": 0, "tx-2": 0}).validate().n_rows == 2


def test_naive_transaction_timestamps_are_rejected() -> None:
    txs = [_transaction("tx-1", created_at=datetime(2026, 3, 1, 12, 0))]
    with pytest.raises(DatasetContractError, match="timezone-aware"):
        _dataset(txs, {"tx-1": 0}).validate()


def test_duplicate_transaction_ids_are_rejected() -> None:
    txs = [
        _transaction("tx-1", created_at=ANCHOR),
        _transaction("tx-1", created_at=ANCHOR + timedelta(hours=1)),
    ]
    with pytest.raises(DatasetContractError, match="Duplicate transaction id"):
        _dataset(txs, {"tx-1": 0}).validate()


def test_missing_label_is_rejected() -> None:
    txs = [
        _transaction("tx-1", created_at=ANCHOR),
        _transaction("tx-2", created_at=ANCHOR + timedelta(hours=1)),
    ]
    with pytest.raises(DatasetContractError, match="no label"):
        _dataset(txs, {"tx-1": 0}).validate()


def test_orphan_label_is_rejected() -> None:
    txs = [_transaction("tx-1", created_at=ANCHOR)]
    with pytest.raises(DatasetContractError, match="unknown transactions"):
        _dataset(txs, {"tx-1": 0, "tx-ghost": 1}).validate()


def test_non_binary_label_is_rejected() -> None:
    txs = [_transaction("tx-1", created_at=ANCHOR)]
    with pytest.raises(DatasetContractError, match="labels must be 0 or 1"):
        _dataset(txs, {"tx-1": 2}).validate()


def test_unknown_customer_reference_is_rejected() -> None:
    txs = [_transaction("tx-1", created_at=ANCHOR, customer_id="cust-ghost")]
    with pytest.raises(DatasetContractError, match="unknown customer"):
        _dataset(txs, {"tx-1": 0}).validate()


def test_unknown_merchant_reference_is_rejected() -> None:
    txs = [_transaction("tx-1", created_at=ANCHOR, merchant_id="merch-ghost")]
    with pytest.raises(DatasetContractError, match="unknown merchant"):
        _dataset(txs, {"tx-1": 0}).validate()


def test_session_attached_transactions_are_rejected() -> None:
    """Canonical objects are transient — the offline path issues no queries."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    SessionTesting = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session: Session = SessionTesting()
    try:
        customer, merchant = _customer(), _merchant()
        tx = _transaction("tx-1", created_at=ANCHOR)
        session.add_all([customer, merchant, tx])
        session.flush()

        dataset = CanonicalDataset(
            customers={"cust-1": customer},
            merchants={"merch-1": merchant},
            transactions=[tx],
            labels={"tx-1": 0},
            provenance=_provenance(),
        )
        with pytest.raises(DatasetContractError, match="attached to a Session"):
            dataset.validate()
    finally:
        session.close()


def test_labels_are_not_reachable_from_the_transaction_objects() -> None:
    """The target lives outside the ORM object, so no feature can read it."""
    tx = _transaction("tx-1", created_at=ANCHOR)
    ds = _dataset([tx], {"tx-1": 1}).validate()

    assert ds.labels["tx-1"] == 1
    for attribute in ("is_fraud", "label", "target", "fraud_label"):
        assert not hasattr(tx, attribute)


def test_validate_returns_the_dataset_for_chaining() -> None:
    ds = _dataset([_transaction("tx-1", created_at=ANCHOR)], {"tx-1": 0})
    assert ds.validate() is ds


# ---------------------------------------------------------------------------
# DatasetAdapter protocol
# ---------------------------------------------------------------------------


def test_a_conforming_object_satisfies_the_adapter_protocol() -> None:
    class FixtureAdapter:
        name = "fixture"

        def load(
            self,
            root: object,
            *,
            max_entities: int | None = None,
            seed: int = 42,
        ) -> CanonicalDataset:
            return _dataset([], {})

    assert isinstance(FixtureAdapter(), DatasetAdapter)


def test_an_object_without_load_does_not_satisfy_the_protocol() -> None:
    class NotAnAdapter:
        name = "nope"

    assert not isinstance(NotAnAdapter(), DatasetAdapter)
