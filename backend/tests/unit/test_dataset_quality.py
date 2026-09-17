"""Phase 5B — the quality report.

The report exists so nobody has to take a benchmark number on trust. Its most
valuable line is the least glamorous: which canonical fields are constant on
this dataset, because a constant field is an input that cannot contribute and
a metric quoted without that caveat overstates what the model is doing.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.datasets.base import CanonicalDataset, DataOrigin, DatasetProvenance, Subsample
from ml.datasets.quality import (
    FRAUD_DISTRIBUTION_DEFINITIONS,
    QUALITY_REPORT_FILENAME,
    ExclusionRecord,
    FieldNote,
    FieldStatus,
    QualityReport,
    build_quality_report,
    describe_fraud_distribution,
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
# Fraud across cards, over time, and across the chronological folds
# ---------------------------------------------------------------------------

# Twenty rows: the 70/15/15 split holds rows 0-13, 14-16 and 17-19.
# Card A's frauds are all in train; card B has a fraud in train and one in
# test; card C's frauds are in val and test; card D has none. Rows 13 and 14,
# the last train row and the first val row, share one instant. February 2019
# holds no rows.
_SPREAD_ROWS: tuple[tuple[str, datetime, bool], ...] = (
    ("A", datetime(2019, 1, 2, 9, tzinfo=UTC), True),  # 0
    ("B", datetime(2019, 1, 3, 9, tzinfo=UTC), True),  # 1
    ("C", datetime(2019, 1, 4, 9, tzinfo=UTC), False),  # 2
    ("D", datetime(2019, 1, 5, 9, tzinfo=UTC), False),  # 3
    ("A", datetime(2019, 1, 6, 9, tzinfo=UTC), True),  # 4
    ("B", datetime(2019, 1, 7, 9, tzinfo=UTC), False),  # 5
    ("C", datetime(2019, 1, 8, 9, tzinfo=UTC), False),  # 6
    ("D", datetime(2019, 3, 1, 9, tzinfo=UTC), False),  # 7
    ("A", datetime(2019, 3, 2, 9, tzinfo=UTC), False),  # 8
    ("B", datetime(2019, 3, 3, 9, tzinfo=UTC), False),  # 9
    ("D", datetime(2019, 3, 4, 9, tzinfo=UTC), False),  # 10
    ("D", datetime(2019, 3, 5, 9, tzinfo=UTC), False),  # 11
    ("D", datetime(2019, 3, 6, 9, tzinfo=UTC), False),  # 12
    ("D", datetime(2019, 3, 7, 10, tzinfo=UTC), False),  # 13: last train row
    ("C", datetime(2019, 3, 7, 10, tzinfo=UTC), True),  # 14: first val row, same instant
    ("A", datetime(2019, 4, 1, 9, tzinfo=UTC), False),  # 15
    ("D", datetime(2019, 4, 2, 9, tzinfo=UTC), False),  # 16: last val row
    ("B", datetime(2019, 4, 10, 9, tzinfo=UTC), True),  # 17: first test row
    ("C", datetime(2019, 4, 11, 9, tzinfo=UTC), True),  # 18
    ("D", datetime(2019, 4, 12, 9, tzinfo=UTC), False),  # 19
)


def _spread_dataset(
    rows: tuple[tuple[str, datetime, bool], ...] = _SPREAD_ROWS,
) -> CanonicalDataset:
    customers = {card: _customer(card) for card in ("A", "B", "C", "D", "E")}  # E: no rows
    transactions = [
        Transaction(
            id=f"t{index:02d}",
            idempotency_key=f"t{index:02d}",
            customer_id=card,
            merchant_id="m1",
            amount=Decimal("10.00"),
            currency="USD",
            status="APPROVED",
            payment_method="CARD",
            country="US",
            is_card_present=True,
            created_at=moment,
        )
        for index, (card, moment, _) in enumerate(rows)
    ]
    return CanonicalDataset(
        customers=customers,
        merchants={"m1": _merchant("m1")},
        transactions=transactions,
        labels={f"t{index:02d}": int(fraud) for index, (_, _, fraud) in enumerate(rows)},
        provenance=_dataset().provenance,
    ).validate()


def _split(distribution: dict[str, Any]) -> dict[str, Any]:
    split: dict[str, Any] = distribution["chronological_split"]
    assert split["available"] is True
    return split


def test_fraud_is_counted_across_the_cards_that_have_transactions() -> None:
    cards = describe_fraud_distribution(_spread_dataset())["cards"]

    assert cards["with_transactions"] == 4
    assert cards["with_fraud"] == 3
    assert cards["frauds_per_card_with_fraud"]["p50"] == pytest.approx(2.0)
    assert cards["frauds_per_card_with_fraud"]["max"] == pytest.approx(2.0)


def test_fraud_is_counted_for_every_calendar_month_including_empty_ones() -> None:
    by_month = describe_fraud_distribution(_spread_dataset())["by_month"]

    assert by_month == [
        {"month": "2019-01", "rows": 7, "frauds": 3},
        {"month": "2019-02", "rows": 0, "frauds": 0},
        {"month": "2019-03", "rows": 8, "frauds": 1},
        {"month": "2019-04", "rows": 5, "frauds": 2},
    ]


def test_months_are_calendar_months_in_utc() -> None:
    """00:30 on 1 February at UTC+1 is still 31 January in UTC."""
    plus_one = timezone(timedelta(hours=1))
    rows = (
        ("A", datetime(2019, 1, 31, 12, tzinfo=UTC), True),
        ("B", datetime(2019, 2, 1, 0, 30, tzinfo=plus_one), False),
        ("C", datetime(2019, 2, 1, 1, 30, tzinfo=plus_one), False),
    )

    by_month = describe_fraud_distribution(_spread_dataset(rows))["by_month"]

    assert by_month == [
        {"month": "2019-01", "rows": 2, "frauds": 1},
        {"month": "2019-02", "rows": 1, "frauds": 0},
    ]


def test_each_fold_reports_its_rows_frauds_fraud_cards_and_period() -> None:
    folds = _split(describe_fraud_distribution(_spread_dataset()))["folds"]

    assert [
        (fold["name"], fold["rows"], fold["frauds"], fold["cards_with_fraud"]) for fold in folds
    ] == [("train", 14, 3, 2), ("val", 3, 1, 1), ("test", 3, 2, 2)]
    assert folds[0]["first_timestamp"] == _SPREAD_ROWS[0][1].isoformat()
    assert folds[0]["last_timestamp"] == _SPREAD_ROWS[13][1].isoformat()
    assert folds[1]["first_timestamp"] == _SPREAD_ROWS[14][1].isoformat()
    assert folds[2]["last_timestamp"] == _SPREAD_ROWS[19][1].isoformat()


def test_cards_with_fraud_on_both_sides_of_each_boundary_are_counted() -> None:
    """Train|val: only B has fraud before and after. Val|test: B and C do."""
    boundaries = _split(describe_fraud_distribution(_spread_dataset()))["boundaries"]

    first, second = boundaries
    assert first["between"] == ["train", "val"]
    assert first["cards_with_fraud_on_both_sides"] == 1
    assert (first["frauds_before_on_those_cards"], first["frauds_after_on_those_cards"]) == (1, 1)
    assert second["between"] == ["val", "test"]
    assert second["cards_with_fraud_on_both_sides"] == 2
    assert (second["frauds_before_on_those_cards"], second["frauds_after_on_those_cards"]) == (
        2,
        2,
    )


def test_a_boundary_instant_shared_by_both_folds_is_recorded() -> None:
    boundaries = _split(describe_fraud_distribution(_spread_dataset()))["boundaries"]

    first, second = boundaries
    assert first["instant_shared"] is True
    assert first["earlier_fold_last_timestamp"] == first["later_fold_first_timestamp"]
    assert first["rows_at_shared_instant"] == {"earlier_fold": 1, "later_fold": 1}
    assert second["instant_shared"] is False
    assert second["rows_at_shared_instant"] == {"earlier_fold": 0, "later_fold": 0}


def test_the_report_states_the_definitions_behind_the_counts() -> None:
    distribution = describe_fraud_distribution(_spread_dataset())

    assert distribution["definitions"] == dict(FRAUD_DISTRIBUTION_DEFINITIONS)
    assert set(distribution["definitions"]) == {
        "card",
        "month",
        "folds",
        "boundary",
        "card_with_fraud_on_both_sides",
        "shared_instant",
    }
    assert "chronological_split" in distribution["definitions"]["folds"]
    assert "recorded, not corrected" in distribution["definitions"]["shared_instant"]


def test_the_folds_are_the_existing_chronological_split_of_the_rows() -> None:
    from ml.splits import chronological_split

    dataset = _spread_dataset()
    timestamps = [tx.created_at for tx in dataset.transactions]
    splits = chronological_split(timestamps)

    folds = _split(describe_fraud_distribution(dataset))["folds"]

    assert [fold["rows"] for fold in folds] == list(splits.sizes)


def test_too_few_rows_for_three_folds_report_no_split() -> None:
    split = describe_fraud_distribution(_dataset())["chronological_split"]

    assert split == {
        "available": False,
        "reason": "Too few rows for every fold of the 70/15/15 split to hold one.",
    }


def test_an_empty_dataset_reports_no_fraud_distribution_to_speak_of() -> None:
    empty = CanonicalDataset(
        customers={},
        merchants={},
        transactions=[],
        labels={},
        provenance=_dataset().provenance,
    ).validate()

    distribution = describe_fraud_distribution(empty)

    assert distribution["cards"] == {
        "with_transactions": 0,
        "with_fraud": 0,
        "frauds_per_card_with_fraud": None,
    }
    assert distribution["by_month"] == []
    assert distribution["chronological_split"]["available"] is False


def test_without_frauds_no_card_spans_a_boundary() -> None:
    rows = tuple((card, moment, False) for card, moment, _ in _SPREAD_ROWS)

    distribution = describe_fraud_distribution(_spread_dataset(rows))

    assert distribution["cards"]["frauds_per_card_with_fraud"] is None
    for boundary in _split(distribution)["boundaries"]:
        assert boundary["cards_with_fraud_on_both_sides"] == 0
        assert boundary["frauds_after_on_those_cards"] == 0


def test_the_written_report_carries_the_fraud_distribution(tmp_path: Path) -> None:
    report = build_quality_report(_spread_dataset(), raw_row_count=20)

    payload = json.loads(report.write(tmp_path).read_text(encoding="utf-8"))

    assert payload["fraud_distribution"] == json.loads(
        json.dumps(describe_fraud_distribution(_spread_dataset()))
    )
    assert payload["fraud_distribution"]["chronological_split"]["available"] is True


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


def test_sparkov_report_writes_the_complete_unix_time_offset_distribution(
    tmp_path: Path,
) -> None:
    """The distribution sits beside the modal-offset fields, which stay as they were."""
    from tests.unit.test_dataset_sparkov import build_fixture_corpus, fixture_adapter

    root = build_fixture_corpus(tmp_path)
    report = sparkov_report(fixture_adapter(tmp_path).load_detailed(root))
    extra = json.loads(report.write(tmp_path / "run").read_text(encoding="utf-8"))["extra"]

    assert extra["unix_time_modal_offset_seconds"] == 0
    assert extra["unix_time_rows_at_modal_offset"] == report.kept_row_count
    assert extra["unix_time_offset_is_uniform"] is True

    distribution = extra["unix_time_offset_distribution"]
    assert distribution["offsets"] == [{"offset_seconds": 0, "rows": report.kept_row_count}]
    assert type(distribution["offsets"][0]["offset_seconds"]) is int
    assert distribution["rows_compared"] + distribution["rows_without_unix_time"] == (
        report.kept_row_count
    )
    assert distribution["truncated"] is False


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



def test_the_report_folds_are_the_folds_a_run_trained_on_the_same_rows_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report built before training, run trained after: the folds agree exactly."""
    from ml import loading, train
    from ml.datasets.sparkov import build_report
    from ml.tuning import TuningResult
    from tests.unit.test_dataset_sparkov import fixture_adapter
    from tests.unit.test_train_runs import _benchmark_corpus

    corpus = _benchmark_corpus(tmp_path)
    adapter = fixture_adapter(tmp_path)
    report = build_report(adapter.load_detailed(corpus, max_entities=3, seed=11))

    def stub_tune(*_: object, **__: object) -> TuningResult:
        return TuningResult(
            best_params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.3},
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    monkeypatch.setattr(loading, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(train, "tune_hyperparameters", stub_tune)
    monkeypatch.setattr(train, "RUNS_ROOT", tmp_path / "runs")
    train.main(
        [
            "--dataset", "sparkov",
            "--root", str(corpus),
            "--cache-root", str(tmp_path / "cache"),
            "--run-name", "sparkov-fixture",
            "--max-cards", "3",
            "--seed", "11",
            "--n-iter", "1",
            "--cv-splits", "2",
            "--target-fpr", "0.05",
        ]
    )
    run = tmp_path / "runs" / "sparkov-fixture"
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    metadata = json.loads((run / "training_metadata.json").read_text(encoding="utf-8"))
    fraud_counts = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["context"][
        "fraud_counts"
    ]

    folds = report.fraud_distribution["chronological_split"]["folds"]

    assert [(fold["first_timestamp"], fold["last_timestamp"]) for fold in folds] == [
        (split["start"], split["end"]) for split in record["splits"]
    ]
    assert [fold["rows"] for fold in folds] == [
        metadata["train_size"],
        metadata["val_size"],
        metadata["test_size"],
    ]
    assert {fold["name"]: fold["frauds"] for fold in folds} == fraud_counts
