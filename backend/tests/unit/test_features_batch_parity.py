"""Phase 5C — the batch builder must agree with the live scoring path exactly.

The fixture is built so that every one of the 17 features varies across it:
amounts spanning three orders of magnitude, weekend and off-hours timestamps,
country mismatches on both the customer and merchant side, bursts inside the
1h and 24h windows, a dormant card, a gap longer than the 180-day history
window, card-present and card-not-present rows, and merchants across the risk
tiers including a high-risk category.

Parity is asserted element by element with exact float equality. Approximate
comparison would let a genuine formula change hide inside a tolerance, which
is precisely the failure this test exists to catch.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import and_, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.fraud.feature_spec import FEATURESETS, feature_names
from app.fraud.features import FeatureExtractor
from app.models.base import Base
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.datasets.base import (
    CanonicalDataset,
    DataOrigin,
    DatasetProvenance,
)
from ml.features.batch import (
    HISTORY_WINDOW,
    BatchExtractionError,
    _NoDatabase,
    build_feature_matrix,
)

ANCHOR = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)  # a Saturday


def _customer(customer_id: str, *, country: str, risk_tier: str, age_days: int) -> Customer:
    return Customer(
        id=customer_id,
        email=f"{customer_id}@example.invalid",
        full_name=f"Customer {customer_id}",
        country=country,
        risk_tier=risk_tier,
        account_age_days=age_days,
    )


def _merchant(merchant_id: str, *, country: str, category: str, risk: str) -> Merchant:
    return Merchant(
        id=merchant_id,
        name=f"Merchant {merchant_id}",
        category=category,
        mcc="5411",
        country=country,
        risk_rating=risk,
    )


def _tx(
    tx_id: str,
    *,
    customer_id: str,
    merchant_id: str,
    amount: str,
    at: datetime,
    country: str = "US",
    card_present: bool = True,
) -> Transaction:
    return Transaction(
        id=tx_id,
        idempotency_key=tx_id,
        customer_id=customer_id,
        merchant_id=merchant_id,
        amount=Decimal(amount),
        currency="USD",
        status="APPROVED",
        payment_method="CARD",
        country=country,
        is_card_present=card_present,
        created_at=at,
    )


def _fixture_entities() -> tuple[dict[str, Customer], dict[str, Merchant]]:
    customers = {
        # Dense history, low risk, domestic.
        "c-dense": _customer("c-dense", country="US", risk_tier="LOW", age_days=900),
        # Sparse history, medium risk, foreign — drives country_mismatch_customer.
        "c-sparse": _customer("c-sparse", country="GB", risk_tier="MEDIUM", age_days=45),
        # Dormant then active, high risk, brand-new account.
        "c-dormant": _customer("c-dormant", country="US", risk_tier="HIGH", age_days=3),
    }
    merchants = {
        "m-grocery": _merchant("m-grocery", country="US", category="GROCERY", risk="LOW"),
        "m-travel": _merchant("m-travel", country="FR", category="TRAVEL", risk="MEDIUM"),
        "m-crypto": _merchant("m-crypto", country="US", category="CRYPTO", risk="HIGH"),
    }
    return customers, merchants


def _fixture_transactions() -> list[Transaction]:
    """~40 rows across three cards, ordered chronologically."""
    rows: list[Transaction] = []

    # c-dense: a long steady history, then a burst inside the 1h window,
    # then an amount far above its 30-day mean (drives the z-score).
    for day in range(20):
        rows.append(
            _tx(
                f"dense-{day:02d}",
                customer_id="c-dense",
                merchant_id="m-grocery" if day % 3 else "m-travel",
                amount=f"{40 + day * 3}.50",
                at=ANCHOR - timedelta(days=40 - day, hours=day % 7),
                card_present=day % 4 != 0,
            )
        )
    burst_start = ANCHOR - timedelta(hours=3)
    for minute in (0, 12, 25, 47):
        rows.append(
            _tx(
                f"dense-burst-{minute:02d}",
                customer_id="c-dense",
                merchant_id="m-crypto",
                amount="820.00",
                at=burst_start + timedelta(minutes=minute),
                card_present=False,
            )
        )
    rows.append(
        _tx(
            "dense-spike",
            customer_id="c-dense",
            merchant_id="m-crypto",
            amount="4900.00",
            at=ANCHOR + timedelta(minutes=5),
            card_present=False,
        )
    )
    # Weekend and off-hours rows for the same card.
    rows.append(
        _tx(
            "dense-weekend",
            customer_id="c-dense",
            merchant_id="m-grocery",
            amount="66.00",
            at=ANCHOR + timedelta(hours=6),  # still Saturday
        )
    )
    rows.append(
        _tx(
            "dense-offhours",
            customer_id="c-dense",
            merchant_id="m-grocery",
            amount="18.00",
            at=ANCHOR + timedelta(days=2, hours=-9),  # 03:00 on a Monday
        )
    )

    # c-sparse: three widely spaced transactions, one abroad.
    rows.append(
        _tx("sparse-0", customer_id="c-sparse", merchant_id="m-travel", amount="210.00",
            at=ANCHOR - timedelta(days=120), country="FR")
    )
    rows.append(
        _tx("sparse-1", customer_id="c-sparse", merchant_id="m-travel", amount="95.25",
            at=ANCHOR - timedelta(days=35), country="GB", card_present=False)
    )
    rows.append(
        _tx("sparse-2", customer_id="c-sparse", merchant_id="m-grocery", amount="12.99",
            at=ANCHOR - timedelta(days=1), country="US")
    )

    # c-dormant: one very old transaction, then activity after a gap far
    # longer than the 180-day history window.
    rows.append(
        _tx("dormant-old", customer_id="c-dormant", merchant_id="m-grocery", amount="30.00",
            at=ANCHOR - timedelta(days=400))
    )
    rows.append(
        _tx("dormant-return", customer_id="c-dormant", merchant_id="m-crypto", amount="2500.00",
            at=ANCHOR - timedelta(days=2), card_present=False)
    )
    rows.append(
        _tx("dormant-followup", customer_id="c-dormant", merchant_id="m-crypto", amount="2600.00",
            at=ANCHOR - timedelta(days=1, hours=20), card_present=False)
    )

    rows.sort(key=lambda tx: (tx.created_at, tx.id))
    return rows


def _provenance() -> DatasetProvenance:
    return DatasetProvenance(
        name="parity-fixture",
        version="v1",
        origin=DataOrigin.SYNTHETIC,
        source_url="https://example.test/fixture",
        citation="parity fixture",
        license="CC0",
        label_field="is_fraud",
        label_definition="1 = fraud",
    )


def _canonical(transactions: list[Transaction] | None = None) -> CanonicalDataset:
    customers, merchants = _fixture_entities()
    rows = transactions if transactions is not None else _fixture_transactions()
    labels = {tx.id: int(tx.id.endswith(("spike", "return"))) for tx in rows}
    return CanonicalDataset(
        customers=customers,
        merchants=merchants,
        transactions=rows,
        labels=labels,
        provenance=_provenance(),
    ).validate()


@pytest.fixture
def db_session() -> Iterator[Session]:
    """A session holding the same fixture, for the SQL-backed comparison."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    SessionTesting = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = SessionTesting()

    customers, merchants = _fixture_entities()
    session.add_all(list(customers.values()))
    session.add_all(list(merchants.values()))
    session.add_all(_fixture_transactions())
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _serving_history(db: Session, tx: Transaction) -> list[Transaction]:
    """Exactly what `scoring._load_context` loads for this transaction."""
    cutoff = tx.created_at - HISTORY_WINDOW
    stmt = (
        select(Transaction)
        .where(
            and_(
                Transaction.customer_id == tx.customer_id,
                Transaction.created_at < tx.created_at,
                Transaction.created_at >= cutoff,
            )
        )
        .order_by(Transaction.created_at.desc())
    )
    return list(db.execute(stmt).scalars().all())


def _serving_vector(db: Session, tx_id: str) -> list[float]:
    """Feature vector as the live scoring service would compute it."""
    tx = db.get(Transaction, tx_id)
    assert tx is not None
    customer = db.get(Customer, tx.customer_id)
    merchant = db.get(Merchant, tx.merchant_id)
    assert customer is not None and merchant is not None

    return FeatureExtractor().extract(
        db,
        tx,
        customer=customer,
        merchant=merchant,
        recent_transactions=_serving_history(db, tx),
    ).values


# ---------------------------------------------------------------------------
# Golden parity
# ---------------------------------------------------------------------------


def test_batch_matches_the_serving_path_for_every_row(db_session: Session) -> None:
    """The whole point of 5C: one implementation, two callers, same numbers."""
    matrix = build_feature_matrix(_canonical())
    names = feature_names("v1")

    for row_index, tx_id in enumerate(matrix.transaction_ids):
        expected = _serving_vector(db_session, tx_id)
        actual = matrix.X[row_index]

        assert len(expected) == len(names)
        for feature_index, name in enumerate(names):
            assert actual[feature_index] == expected[feature_index], (
                f"{name} diverged on {tx_id}: "
                f"batch={actual[feature_index]!r} serving={expected[feature_index]!r}"
            )


def test_parity_holds_for_every_feature_being_exercised(db_session: Session) -> None:
    """A fixture where a feature never varies would make its parity vacuous."""
    matrix = build_feature_matrix(_canonical())
    names = feature_names("v1")

    constant = [
        name
        for index, name in enumerate(names)
        if len(np.unique(matrix.X[:, index])) == 1
    ]
    assert constant == [], f"features constant across the fixture: {constant}"


def test_feature_order_matches_the_frozen_registry() -> None:
    matrix = build_feature_matrix(_canonical())
    assert matrix.feature_names == FEATURESETS["v1"]
    assert matrix.X.shape == (matrix.y.shape[0], len(FEATURESETS["v1"]))


def test_reordering_the_registry_would_break_parity(db_session: Session) -> None:
    """Guards the assertion above: position, not just membership, is checked."""
    matrix = build_feature_matrix(_canonical())
    serving = _serving_vector(db_session, matrix.transaction_ids[-1])
    rotated = serving[1:] + serving[:1]

    assert list(matrix.X[-1]) == serving
    assert list(matrix.X[-1]) != rotated


# ---------------------------------------------------------------------------
# History semantics
# ---------------------------------------------------------------------------


def test_velocity_features_see_prior_rows_only(db_session: Session) -> None:
    """The burst rows must show a rising 1h count, not a constant one."""
    matrix = build_feature_matrix(_canonical())
    index_of = {tx_id: i for i, tx_id in enumerate(matrix.transaction_ids)}
    column = feature_names("v1").index("tx_count_1h")

    counts = [
        matrix.X[index_of[f"dense-burst-{minute:02d}"], column] for minute in (0, 12, 25, 47)
    ]
    assert counts == [0.0, 1.0, 2.0, 3.0]


def test_a_transaction_is_never_part_of_its_own_history() -> None:
    """Off-by-one here would leak the current row into its own velocity count."""
    single = [
        _tx("solo", customer_id="c-dense", merchant_id="m-grocery", amount="50.00", at=ANCHOR)
    ]
    matrix = build_feature_matrix(_canonical(single))
    names = feature_names("v1")

    assert matrix.X[0, names.index("tx_count_1h")] == 0.0
    assert matrix.X[0, names.index("tx_count_24h")] == 0.0
    assert matrix.X[0, names.index("days_since_last_tx")] == 999.0


def test_future_transactions_cannot_affect_an_earlier_row() -> None:
    """Truncating the dataset after row i must not change row i's features."""
    full = build_feature_matrix(_canonical())
    transactions = _fixture_transactions()

    for cut in (1, 5, len(transactions) // 2, len(transactions)):
        truncated = build_feature_matrix(_canonical(transactions[:cut]))
        assert truncated.X.shape[0] == cut
        np.testing.assert_array_equal(truncated.X, full.X[:cut])


def test_history_window_eviction_matches_the_serving_window(db_session: Session) -> None:
    """The dormant card's return is the row where the 180-day boundary bites."""
    matrix = build_feature_matrix(_canonical())
    index_of = {tx_id: i for i, tx_id in enumerate(matrix.transaction_ids)}
    column = feature_names("v1").index("days_since_last_tx")

    batch_value = matrix.X[index_of["dormant-return"], column]
    serving_value = _serving_vector(db_session, "dormant-return")[column]

    assert batch_value == serving_value == 999.0


def test_unbounded_sql_path_differs_only_in_days_since_last_tx(db_session: Session) -> None:
    """Documents the divergence rather than leaving it to be rediscovered.

    The synthetic training path calls `extract()` without `recent_transactions`,
    which queries unbounded history. For a gap longer than 180 days that yields
    a real day count where the serving path yields the 999 sentinel. Every
    other feature is unaffected.
    """
    tx = db_session.get(Transaction, "dormant-return")
    assert tx is not None
    customer = db_session.get(Customer, tx.customer_id)
    merchant = db_session.get(Merchant, tx.merchant_id)

    unbounded = FeatureExtractor().extract(
        db_session, tx, customer=customer, merchant=merchant
    ).values
    serving = _serving_vector(db_session, "dormant-return")
    column = feature_names("v1").index("days_since_last_tx")

    assert unbounded[column] == 398.0
    assert serving[column] == 999.0
    assert unbounded[:column] == serving[:column]
    assert unbounded[column + 1 :] == serving[column + 1 :]


# ---------------------------------------------------------------------------
# Determinism, traceability, labels
# ---------------------------------------------------------------------------


def test_output_is_byte_identical_across_runs() -> None:
    first = build_feature_matrix(_canonical())
    second = build_feature_matrix(_canonical())

    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.transaction_ids == second.transaction_ids


def test_rows_follow_the_canonical_chronological_order() -> None:
    matrix = build_feature_matrix(_canonical())
    expected = [tx.id for tx in _fixture_transactions()]

    assert matrix.transaction_ids == expected
    timestamps = list(matrix.timestamps)
    assert timestamps == sorted(timestamps)


def test_every_row_traces_back_to_its_transaction() -> None:
    dataset = _canonical()
    matrix = build_feature_matrix(dataset)
    by_id = {tx.id: tx for tx in dataset.transactions}

    assert len(set(matrix.transaction_ids)) == len(matrix.transaction_ids)
    for row_index, tx_id in enumerate(matrix.transaction_ids):
        assert matrix.timestamps[row_index] == by_id[tx_id].created_at


def test_labels_come_from_the_dataset_map_not_the_rows() -> None:
    dataset = _canonical()
    matrix = build_feature_matrix(dataset)

    for row_index, tx_id in enumerate(matrix.transaction_ids):
        assert matrix.y[row_index] == dataset.labels[tx_id]
    for tx in dataset.transactions:
        assert not hasattr(tx, "is_fraud")
        assert tx.fraud_score is None


def test_empty_dataset_produces_an_empty_matrix_with_the_right_width() -> None:
    matrix = build_feature_matrix(_canonical([]))
    assert matrix.X.shape == (0, len(FEATURESETS["v1"]))
    assert matrix.transaction_ids == []


# ---------------------------------------------------------------------------
# No database, ever
# ---------------------------------------------------------------------------


def test_the_sentinel_refuses_any_attribute_access() -> None:
    with pytest.raises(BatchExtractionError, match="database access"):
        _NoDatabase().execute  # noqa: B018


def test_extraction_never_touches_a_session() -> None:
    """If extract() ever fell back to SQL here, this would raise, not drift."""
    matrix = build_feature_matrix(_canonical())
    assert matrix.X.shape[0] == len(_fixture_transactions())


def test_unknown_featureset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown featureset version"):
        build_feature_matrix(_canonical(), featureset="v99")


def test_batch_window_equals_the_serving_window() -> None:
    """Pins the two constants together without importing serving code."""
    from app.services.scoring import _RECENT_HISTORY_WINDOW

    assert HISTORY_WINDOW == _RECENT_HISTORY_WINDOW
