"""Phase 5B — the quality report.

The report exists so nobody has to take a benchmark number on trust. Its most
valuable line is the least glamorous: which canonical fields are constant on
this dataset, because a constant field is an input that cannot contribute and
a metric quoted without that caveat overstates what the model is doing.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.datasets.base import CanonicalDataset, DataOrigin, DatasetProvenance, Subsample
from ml.datasets.quality import (
    QUALITY_REPORT_FILENAME,
    ExclusionRecord,
    FieldNote,
    FieldStatus,
    QualityReport,
    build_quality_report,
)

ANCHOR = datetime(2019, 6, 1, 9, 0, tzinfo=UTC)


def _customer(customer_id: str, *, country: str = "US", risk_tier: str = "LOW") -> Customer:
    return Customer(
        id=customer_id,
        email=f"{customer_id}@example.invalid",
        full_name=f"Customer {customer_id}",
        country=country,
        risk_tier=risk_tier,
        account_age_days=0,
    )


def _merchant(merchant_id: str, *, category: str = "GROCERY", risk: str = "LOW") -> Merchant:
    return Merchant(
        id=merchant_id,
        name=f"Merchant {merchant_id}",
        category=category,
        mcc="5411",
        country="US",
        risk_rating=risk,
    )


def _tx(
    tx_id: str,
    *,
    customer_id: str,
    merchant_id: str,
    amount: str,
    minutes: int,
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
        country="US",
        is_card_present=card_present,
        created_at=ANCHOR + timedelta(minutes=minutes),
    )


def _dataset(*, subsample: Subsample | None = None) -> CanonicalDataset:
    customers = {"c1": _customer("c1"), "c2": _customer("c2")}
    merchants = {
        "m1": _merchant("m1"),
        "m2": _merchant("m2", category="TRAVEL", risk="MEDIUM"),
    }
    transactions = [
        _tx("t1", customer_id="c1", merchant_id="m1", amount="10.00", minutes=0),
        _tx("t2", customer_id="c1", merchant_id="m1", amount="20.00", minutes=10),
        _tx("t3", customer_id="c1", merchant_id="m2", amount="900.00", minutes=20,
            card_present=False),
        _tx("t4", customer_id="c2", merchant_id="m2", amount="30.00", minutes=30),
    ]
    provenance = DatasetProvenance(
        name="fixture",
        version="v1",
        origin=DataOrigin.SYNTHETIC,
        source_url="https://example.test",
        citation="fixture",
        license="CC0",
        label_field="is_fraud",
        label_definition="1 = fraud",
        subsample=subsample,
    )
    return CanonicalDataset(
        customers=customers,
        merchants=merchants,
        transactions=transactions,
        labels={"t1": 0, "t2": 0, "t3": 1, "t4": 0},
        provenance=provenance,
    ).validate()


def test_report_counts_rows_kept_and_excluded() -> None:
    report = build_quality_report(
        _dataset(),
        raw_row_count=6,
        exclusions=(ExclusionRecord("duplicate_trans_num", 2, ("a", "b")),),
    )

    assert report.raw_row_count == 6
    assert report.kept_row_count == 4
    assert report.excluded_row_count == 2


def test_report_states_the_label_distribution() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)
    assert report.fraud_count == 1
    assert report.fraud_rate == pytest.approx(0.25)


def test_report_records_the_observed_period() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)
    assert report.period_start == ANCHOR.isoformat()
    assert report.period_end == (ANCHOR + timedelta(minutes=30)).isoformat()


def test_report_counts_entities_and_their_activity() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)

    assert report.customer_count == 2
    assert report.merchant_count == 2
    assert report.transactions_per_customer["max"] == pytest.approx(3.0)


def test_report_summarises_amounts() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)
    assert report.amount_percentiles["max"] == pytest.approx(900.0)
    assert report.amount_percentiles["p50"] == pytest.approx(25.0)


def test_report_breaks_fraud_down_by_category() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)

    assert report.rows_by_category == {"GROCERY": 2, "TRAVEL": 2}
    assert report.fraud_rate_by_category["TRAVEL"] == pytest.approx(0.5)
    assert report.fraud_rate_by_category["GROCERY"] == pytest.approx(0.0)


def test_report_measures_the_card_present_mix() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)
    assert report.card_present_share == pytest.approx(0.75)


def test_report_names_constant_fields() -> None:
    """A constant field is a dead feature; the report must say so outright."""
    report = build_quality_report(_dataset(), raw_row_count=4)

    assert report.constant_fields["customer.country"] == "US"
    assert report.constant_fields["customer.risk_tier"] == "LOW"
    assert report.constant_fields["customer.account_age_days"] == "0"
    assert "merchant.category" not in report.constant_fields
    assert "transaction.is_card_present" not in report.constant_fields


def test_constant_fields_are_surfaced_as_a_warning() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)
    assert any("Constant canonical fields" in warning for warning in report.warnings)


def test_heavy_exclusion_raises_a_warning() -> None:
    report = build_quality_report(
        _dataset(),
        raw_row_count=100,
        exclusions=(ExclusionRecord("non_positive_amount", 40),),
    )
    assert any("40.00% of source rows were excluded" in w for w in report.warnings)


def test_small_exclusion_does_not_raise_that_warning() -> None:
    report = build_quality_report(
        _dataset(), raw_row_count=1000, exclusions=(ExclusionRecord("dupes", 1),)
    )
    assert not any("source rows were excluded" in w for w in report.warnings)


def test_empty_dataset_is_flagged() -> None:
    empty = CanonicalDataset(
        customers={},
        merchants={},
        transactions=[],
        labels={},
        provenance=_dataset().provenance,
    ).validate()
    report = build_quality_report(empty, raw_row_count=0)

    assert "Dataset is empty after exclusions." in report.warnings
    assert report.amount_percentiles["max"] == 0.0


def test_dataset_without_positives_is_flagged() -> None:
    dataset = _dataset()
    unlabelled = CanonicalDataset(
        customers=dataset.customers,
        merchants=dataset.merchants,
        transactions=dataset.transactions,
        labels=dict.fromkeys(dataset.labels, 0),
        provenance=dataset.provenance,
    ).validate()

    report = build_quality_report(unlabelled, raw_row_count=4)
    assert any("No positive labels" in warning for warning in report.warnings)


def test_report_carries_the_subsample_record() -> None:
    subsample = Subsample(strategy="cards", seed=42, max_entities=2, selected_entities=2)
    report = build_quality_report(_dataset(subsample=subsample), raw_row_count=4)

    assert report.subsample is not None
    assert report.subsample["strategy"] == "cards"
    assert report.subsample["seed"] == 42


def test_report_carries_the_field_inventory_and_exclusions() -> None:
    report = build_quality_report(
        _dataset(),
        raw_row_count=5,
        exclusions=(ExclusionRecord("duplicate_trans_num", 1, ("dup",)),),
        field_notes=(FieldNote("cc_num", FieldStatus.MAPPED, "uuid5"),),
    )
    payload = report.to_dict()

    assert payload["exclusions"] == [
        {"reason": "duplicate_trans_num", "count": 1, "examples": ["dup"]}
    ]
    assert payload["field_notes"] == [
        {"field": "cc_num", "status": "mapped", "detail": "uuid5"}
    ]


def test_zero_count_exclusions_stay_in_the_report() -> None:
    """Showing that a check ran and found nothing is itself information."""
    report = build_quality_report(
        _dataset(), raw_row_count=4, exclusions=(ExclusionRecord("duplicate_trans_num", 0),)
    )
    assert report.to_dict()["exclusions"][0]["count"] == 0


def test_extra_fields_are_passed_through() -> None:
    report = build_quality_report(_dataset(), raw_row_count=4, extra={"unix_time_mismatches": 3})
    assert report.to_dict()["extra"]["unix_time_mismatches"] == 3


def test_report_writes_json_to_a_run_directory(tmp_path: Path) -> None:
    report = build_quality_report(_dataset(), raw_row_count=4)
    written = report.write(tmp_path / "runs" / "sparkov-fixture")

    assert written.name == QUALITY_REPORT_FILENAME
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["dataset"] == "fixture"
    assert payload["origin"] == "synthetic"
    assert payload["rows"]["kept"] == 4


def test_written_report_is_stable_json(tmp_path: Path) -> None:
    """Sorted keys keep a committed report diffable between runs."""
    report = build_quality_report(_dataset(), raw_row_count=4)
    text = report.write(tmp_path).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text)["period"]["start"] == ANCHOR.isoformat()


# ---------------------------------------------------------------------------
# The Sparkov report, end to end on fixtures
# ---------------------------------------------------------------------------


def test_sparkov_report_carries_the_adapter_specific_checks(tmp_path: Path) -> None:
    from tests.unit.test_dataset_sparkov import build_fixture_corpus, fixture_adapter

    root = build_fixture_corpus(tmp_path)
    result = fixture_adapter(tmp_path).load_detailed(root)
    report = sparkov_report(result)

    assert report.dataset == "sparkov"
    assert report.origin == "synthetic"
    assert report.kept_row_count == result.dataset.n_rows
    assert report.extra["unix_time_mismatches"] == 0
    assert report.extra["multi_category_merchant_names"] == 0


def test_sparkov_report_names_the_fields_that_are_inert(tmp_path: Path) -> None:
    """On Sparkov both country features and the risk tier cannot contribute."""
    from tests.unit.test_dataset_sparkov import build_fixture_corpus, fixture_adapter

    root = build_fixture_corpus(tmp_path)
    report = sparkov_report(fixture_adapter(tmp_path).load_detailed(root))

    assert report.constant_fields["customer.country"] == "US"
    assert report.constant_fields["merchant.country"] == "US"
    assert report.constant_fields["transaction.country"] == "US"
    assert report.constant_fields["customer.risk_tier"] == "LOW"
    assert report.constant_fields["customer.account_age_days"] == "0"


def sparkov_report(result: object) -> QualityReport:
    from ml.datasets.sparkov import SparkovLoadResult, build_report

    assert isinstance(result, SparkovLoadResult)
    return build_report(result)
