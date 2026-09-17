"""Phase 5B — the Sparkov adapter, exercised on fixtures.

CI never downloads the real 500 MB corpus. Everything here runs against a
handful of hand-written rows that reproduce the properties that matter: the
`fraud_` merchant prefix, the `_pos` / `_net` category suffixes, duplicate
transaction numbers, junk amounts, and cards with histories worth keeping
whole.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from ml.datasets import registry, sparkov
from ml.datasets.base import CanonicalDataset, DataOrigin, DatasetContractError
from ml.datasets.manifest import ManifestError
from ml.datasets.quality import FieldStatus
from ml.datasets.sparkov import (
    CATEGORY_MAP,
    SPARKOV_COLUMNS,
    UNIX_TIME_OFFSET_DEFINITION,
    UNIX_TIME_OFFSET_LISTING_LIMIT,
    SparkovAdapter,
    SparkovSchemaError,
    clean_merchant_name,
    customer_id_for,
    is_card_present_for,
    merchant_id_for,
    transaction_id_for,
)

HEADER = ["", *SPARKOV_COLUMNS]


def _row(
    index: int,
    *,
    trans_num: str,
    cc_num: str = "4111111111111111",
    merchant: str = "fraud_Rippin, Kub and Mann",
    category: str = "grocery_pos",
    amt: str = "120.50",
    timestamp: str = "2019-01-02 10:15:00",
    is_fraud: str = "0",
    unix_time: str | None = None,
) -> list[str]:
    # Deliberately malformed timestamps are a fixture case, so the derived
    # epoch falls back to 0 instead of the helper refusing to build the row.
    try:
        epoch = str(int(datetime.fromisoformat(timestamp).replace(tzinfo=UTC).timestamp()))
    except ValueError:
        epoch = "0"
    return [
        str(index),
        timestamp,
        cc_num,
        merchant,
        category,
        amt,
        "Jane",
        "Doe",
        "F",
        "1 Example Street",
        "Springfield",
        "IL",
        "62701",
        "39.7",
        "-89.6",
        "120000",
        "Engineer",
        "1985-04-12",
        trans_num,
        unix_time if unix_time is not None else epoch,
        "39.8",
        "-89.7",
        is_fraud,
    ]


def _write_csv(path: Path, rows: Sequence[Sequence[str]], *, header: Sequence[str] = HEADER) -> None:
    lines = [",".join(f'"{value}"' for value in header)]
    lines += [",".join(f'"{value}"' for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path, filenames: Sequence[str] = ("fraudTrain.csv",)) -> Path:
    payload = {
        "manifest_version": "1",
        "datasets": {
            "sparkov": {
                "display_name": "Fixture Sparkov",
                "origin": "synthetic",
                "source": "kaggle",
                "reference": "kartik2112/fraud-detection",
                "source_url": "https://www.kaggle.com/datasets/kartik2112/fraud-detection",
                "license": "CC0: Public Domain (fixture)",
                "citation": "Fixture citation",
                "label_field": "is_fraud",
                "label_definition": "1 = fraud",
                "notes": "External synthetic benchmark. Not real card-transaction data.",
                "files": {name: {"bytes": None, "sha256": None} for name in filenames},
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_fixture_corpus(tmp_path: Path) -> Path:
    """Three cards, mixed channels and categories, one fraud.

    Exposed as a plain function so the quality-report suite can reuse the same
    corpus without duplicating the row definitions.
    """
    root = tmp_path / "raw" / "sparkov"
    root.mkdir(parents=True)
    rows = [
        _row(0, trans_num="t1", cc_num="4111", timestamp="2019-01-01 08:00:00"),
        _row(1, trans_num="t2", cc_num="4111", timestamp="2019-01-03 09:30:00",
             category="shopping_net", amt="45.00", merchant="fraud_Schiller Ltd"),
        _row(2, trans_num="t3", cc_num="4222", timestamp="2019-01-02 11:00:00",
             category="gas_transport", amt="60.25", merchant="fraud_Stokes Inc"),
        _row(3, trans_num="t4", cc_num="4222", timestamp="2019-01-04 23:45:00",
             category="misc_net", amt="900.00", is_fraud="1", merchant="fraud_Schiller Ltd"),
        _row(4, trans_num="t5", cc_num="4333", timestamp="2019-01-05 12:00:00",
             category="travel", amt="1500.00", merchant="fraud_Kuhn LLC"),
    ]
    _write_csv(root / "fraudTrain.csv", rows)
    return root


def fixture_adapter(tmp_path: Path) -> SparkovAdapter:
    """An adapter pointed at a fixture manifest, with hash pinning off."""
    return SparkovAdapter(manifest_path=_manifest(tmp_path), verify_hashes=False)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build_fixture_corpus(tmp_path)


@pytest.fixture
def adapter(tmp_path: Path) -> SparkovAdapter:
    return fixture_adapter(tmp_path)


def _load(adapter: SparkovAdapter, root: Path, **kwargs: object) -> CanonicalDataset:
    return adapter.load(root, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_adapter_registers_itself_under_its_dataset_name() -> None:
    assert registry.get_adapter("sparkov").name == "sparkov"


def test_registered_adapter_satisfies_the_protocol() -> None:
    from ml.datasets.base import DatasetAdapter

    assert isinstance(registry.get_adapter("sparkov"), DatasetAdapter)


# ---------------------------------------------------------------------------
# Identity mapping
# ---------------------------------------------------------------------------


def test_ids_are_deterministic_across_calls() -> None:
    assert customer_id_for("4111") == customer_id_for("4111")
    assert transaction_id_for("t1") == transaction_id_for("t1")
    assert merchant_id_for("Kuhn LLC", "TRAVEL") == merchant_id_for("Kuhn LLC", "TRAVEL")


def test_ids_differ_across_entity_kinds_for_the_same_source_value() -> None:
    """A card number and a merchant name that collide must not share an id."""
    assert customer_id_for("x") != merchant_id_for("x", "RETAIL") != transaction_id_for("x")


def test_merchant_identity_includes_the_canonical_category() -> None:
    """A Merchant carries one category, so the id must not outlive it."""
    assert merchant_id_for("Kuhn LLC", "TRAVEL") != merchant_id_for("Kuhn LLC", "RETAIL")


def test_merchant_name_under_two_categories_is_split_and_counted(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    """Otherwise one category wins arbitrarily and rides along on the id."""
    root = tmp_path / "raw" / "spanning"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [
            _row(0, trans_num="a", merchant="fraud_Both Ltd", category="grocery_pos"),
            _row(1, trans_num="b", merchant="fraud_Both Ltd", category="travel"),
            _row(2, trans_num="c", merchant="fraud_Both Ltd", category="grocery_net"),
        ],
    )

    result = adapter.load_detailed(root)
    categories = {m.category for m in result.dataset.merchants.values()}

    assert result.multi_category_merchants == 1
    assert categories == {"GROCERY", "TRAVEL"}
    # Both grocery channels map to one canonical category, so they share a merchant.
    assert len(result.dataset.merchants) == 2


def test_ids_are_uuid_shaped_for_the_36_char_columns() -> None:
    assert len(customer_id_for("4111111111111111")) == 36


def test_raw_card_number_never_appears_in_the_canonical_objects(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    blob = json.dumps(
        [
            {
                "id": customer.id,
                "email": customer.email,
                "full_name": customer.full_name,
                "country": customer.country,
            }
            for customer in dataset.customers.values()
        ]
    )
    for card in ("4111", "4222", "4333"):
        assert card not in blob


def test_merchant_prefix_is_stripped() -> None:
    """`fraud_` is on every merchant name, fraudulent or not — a naming artifact."""
    assert clean_merchant_name("fraud_Kuhn LLC") == "Kuhn LLC"
    assert clean_merchant_name("Kuhn LLC") == "Kuhn LLC"


def test_stripping_the_prefix_does_not_merge_distinct_merchants(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    names = {merchant.name for merchant in dataset.merchants.values()}
    assert names == {"Rippin, Kub and Mann", "Schiller Ltd", "Stokes Inc", "Kuhn LLC"}


# ---------------------------------------------------------------------------
# Schema mapping
# ---------------------------------------------------------------------------


def test_every_sparkov_category_is_mapped() -> None:
    """The source publishes 14 categories; an unmapped one must fail loudly."""
    assert len(CATEGORY_MAP) == 14
    assert set(CATEGORY_MAP) == {
        "grocery_pos",
        "grocery_net",
        "food_dining",
        "shopping_pos",
        "shopping_net",
        "misc_pos",
        "misc_net",
        "home",
        "kids_pets",
        "personal_care",
        "gas_transport",
        "entertainment",
        "travel",
        "health_fitness",
    }


def test_mapping_targets_are_exactly_these_canonical_categories() -> None:
    """Pinned as an equality: a retargeted mapping must be a deliberate edit."""
    assert set(CATEGORY_MAP.values()) == {
        "GROCERY",
        "RESTAURANT",
        "RETAIL",
        "GAS_STATION",
        "ENTERTAINMENT",
        "TRAVEL",
    }


def test_health_fitness_maps_to_entertainment_not_an_online_bucket() -> None:
    """Category carries merchant type; is_card_present carries channel.

    ONLINE_SERVICE (MCC 5968, direct marketing/subscription) asserts a
    card-not-present channel, but health_fitness has no _pos/_net suffix and
    is treated as card-present — so routing it there would contradict the
    adapter's own channel handling.
    """
    assert CATEGORY_MAP["health_fitness"] == "ENTERTAINMENT"
    assert is_card_present_for("health_fitness") is True
    assert "ONLINE_SERVICE" not in set(CATEGORY_MAP.values())


def test_health_fitness_keeps_the_medium_risk_level() -> None:
    from ml.synthesis.merchants import CATEGORIES

    assert CATEGORIES[CATEGORY_MAP["health_fitness"]]["risk"] == "MEDIUM"


def test_category_suffix_decides_card_present() -> None:
    assert is_card_present_for("grocery_pos") is True
    assert is_card_present_for("shopping_net") is False
    assert is_card_present_for("travel") is True  # no suffix: assumed in person


def test_categories_map_onto_the_canonical_taxonomy(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    by_name = {merchant.name: merchant for merchant in dataset.merchants.values()}

    grocery = by_name["Rippin, Kub and Mann"]
    assert grocery.category == "GROCERY"
    assert grocery.mcc == "5411"
    assert grocery.risk_rating == "LOW"


def test_merchant_risk_comes_from_the_taxonomy_not_from_fraud(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    """Risk derived from observed fraud rate would be target leakage."""
    dataset = _load(adapter, corpus)
    fraud_merchant_ids = {
        tx.merchant_id for tx in dataset.transactions if dataset.labels[tx.id] == 1
    }
    for merchant_id in fraud_merchant_ids:
        merchant = dataset.merchants[merchant_id]
        assert merchant.risk_rating in {"LOW", "MEDIUM", "HIGH"}
        assert merchant.risk_rating == ("LOW" if merchant.category == "RETAIL" else merchant.risk_rating)


def test_amounts_become_two_decimal_place_decimals(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    amounts = {tx.idempotency_key: tx.amount for tx in dataset.transactions}

    assert amounts["t1"] == Decimal("120.50")
    assert all(isinstance(amount, Decimal) for amount in amounts.values())
    assert all(amount.as_tuple().exponent == -2 for amount in amounts.values())


def test_constant_fields_are_filled_inertly(adapter: SparkovAdapter, corpus: Path) -> None:
    dataset = _load(adapter, corpus)
    customer = next(iter(dataset.customers.values()))

    assert customer.country == "US"
    assert customer.risk_tier == "LOW"
    assert customer.account_age_days == 0
    assert all(tx.currency == "USD" for tx in dataset.transactions)
    assert all(tx.payment_method == "CARD" for tx in dataset.transactions)


def test_idempotency_key_keeps_the_source_transaction_number(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    assert {tx.idempotency_key for tx in dataset.transactions} == {"t1", "t2", "t3", "t4", "t5"}


# ---------------------------------------------------------------------------
# Timestamps and ordering
# ---------------------------------------------------------------------------


def test_timestamps_are_utc_aware_and_sorted(adapter: SparkovAdapter, corpus: Path) -> None:
    dataset = _load(adapter, corpus)
    timestamps = [tx.created_at for tx in dataset.transactions]

    assert all(ts.tzinfo is not None for ts in timestamps)
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == datetime(2019, 1, 1, 8, 0, tzinfo=UTC)


def test_period_reflects_the_loaded_rows(adapter: SparkovAdapter, corpus: Path) -> None:
    dataset = _load(adapter, corpus)
    assert dataset.period == (
        datetime(2019, 1, 1, 8, 0, tzinfo=UTC),
        datetime(2019, 1, 5, 12, 0, tzinfo=UTC),
    )


def test_unix_time_disagreement_is_reported_not_enforced(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = tmp_path / "raw" / "skew"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [_row(0, trans_num="t1", unix_time="1"), _row(1, trans_num="t2")],
    )

    result = adapter.load_detailed(root)
    assert result.unix_time.mismatches == 1
    assert result.dataset.n_rows == 2


def test_a_uniform_unix_time_offset_is_recognised(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    """Every row off by the same amount is reported as one uniform offset.

    Recognising the shape says nothing about what the offset means. On the
    published files `unix_time` is not a timezone offset but a whole-day shift,
    and it stays a diagnostic either way (Phase 5D decision 17).
    """
    root = tmp_path / "raw" / "offset"
    root.mkdir(parents=True)
    offset = 5 * 3600
    rows = []
    for index in range(4):
        timestamp = f"2019-01-0{index + 1} 10:15:00"
        epoch = int(datetime.fromisoformat(timestamp).replace(tzinfo=UTC).timestamp())
        rows.append(
            _row(index, trans_num=f"t{index}", timestamp=timestamp, unix_time=str(epoch - offset))
        )
    _write_csv(root / "fraudTrain.csv", rows)

    check = adapter.load_detailed(root).unix_time

    assert check.mismatches == 4
    assert check.modal_offset_seconds == offset
    assert check.rows_at_modal_offset == 4
    assert check.is_uniform_offset is True


def test_scattered_unix_time_errors_are_not_a_uniform_offset(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = tmp_path / "raw" / "scatter"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [
            _row(0, trans_num="t0", timestamp="2019-01-01 10:15:00", unix_time="1"),
            _row(1, trans_num="t1", timestamp="2019-01-02 10:15:00", unix_time="2"),
            _row(2, trans_num="t2", timestamp="2019-01-03 10:15:00"),
        ],
    )

    check = adapter.load_detailed(root).unix_time
    assert check.mismatches == 2
    assert check.is_uniform_offset is False


def _epoch(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).replace(tzinfo=UTC).timestamp())


def _corpus_with_offsets(tmp_path: Path, name: str, offsets: Sequence[int | None]) -> Path:
    """One row per day, each `offset` seconds behind its wall clock.

    `None` writes a non-numeric `unix_time`, which cannot yield an offset.
    """
    root = tmp_path / "raw" / name
    root.mkdir(parents=True)
    rows = []
    for index, offset in enumerate(offsets):
        timestamp = f"2019-01-{index + 1:02d} 10:15:00"
        unix_time = "not-a-number" if offset is None else str(_epoch(timestamp) - offset)
        rows.append(_row(index, trans_num=f"t{index}", timestamp=timestamp, unix_time=unix_time))
    _write_csv(root / "fraudTrain.csv", rows)
    return root


def test_a_uniform_offset_is_one_listed_entry_covering_every_row(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = _corpus_with_offsets(tmp_path, "uniform", [3600, 3600, 3600, 3600])

    distribution = adapter.load_detailed(root).unix_time.offset_distribution

    assert distribution.to_dict() == {
        "offset_definition": UNIX_TIME_OFFSET_DEFINITION,
        "rows_compared": 4,
        "rows_without_unix_time": 0,
        "distinct_offsets": 1,
        "min_offset_seconds": 3600,
        "max_offset_seconds": 3600,
        "listing_limit": UNIX_TIME_OFFSET_LISTING_LIMIT,
        "offsets": [{"offset_seconds": 3600, "rows": 4}],
        "truncated": False,
        "unlisted_offsets": 0,
        "unlisted_rows": 0,
    }


def test_every_observed_offset_is_counted_and_rows_without_one_are_counted_apart(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = _corpus_with_offsets(
        tmp_path, "several", [18000, 14400, 18000, 0, 18000, 14400, None]
    )

    distribution = adapter.load_detailed(root).unix_time.offset_distribution

    assert [(entry.offset_seconds, entry.rows) for entry in distribution.offsets] == [
        (18000, 3),
        (14400, 2),
        (0, 1),
    ]
    assert distribution.rows_compared == 6
    assert distribution.rows_without_unix_time == 1
    assert distribution.distinct_offsets == 3
    assert (distribution.min_offset_seconds, distribution.max_offset_seconds) == (0, 18000)
    assert distribution.truncated is False
    # The non-numeric row forces a float column; offsets still come out exact ints.
    assert all(type(entry.offset_seconds) is int for entry in distribution.offsets)


def test_the_modal_offset_fields_are_unchanged_and_agree_with_the_distribution(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    """A tie for the mode resolves to the smaller offset, as it always has."""
    root = _corpus_with_offsets(tmp_path, "tied", [3600, -3600, 3600, -3600, 0])

    check = adapter.load_detailed(root).unix_time

    assert check.mismatches == 4
    assert check.modal_offset_seconds == -3600
    assert check.rows_at_modal_offset == 2
    assert check.is_uniform_offset is False
    first = check.offset_distribution.offsets[0]
    assert (first.offset_seconds, first.rows) == (-3600, 2)


def test_the_distribution_order_is_deterministic_and_independent_of_row_order(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    """Most frequent first; equal counts ordered by the smaller offset."""
    offsets = [7200, -60, 7200, 30, -60, 30, 30]
    forward = adapter.load_detailed(_corpus_with_offsets(tmp_path, "forward", offsets))
    backward = adapter.load_detailed(
        _corpus_with_offsets(tmp_path, "backward", list(reversed(offsets)))
    )

    listed = forward.unix_time.offset_distribution.to_dict()["offsets"]
    assert listed == [
        {"offset_seconds": 30, "rows": 3},
        {"offset_seconds": -60, "rows": 2},
        {"offset_seconds": 7200, "rows": 2},
    ]
    assert (
        forward.unix_time.offset_distribution.to_dict()
        == backward.unix_time.offset_distribution.to_dict()
    )


def test_the_listing_is_bounded_and_states_what_it_left_out() -> None:
    offsets = [100, 100, 100, 200, 200, 300, 400]
    timestamps = [f"2019-01-{index + 1:02d} 00:00:00" for index in range(len(offsets))]
    frame = sparkov.parse_timestamps(
        pd.DataFrame(
            {
                "trans_date_trans_time": timestamps,
                "unix_time": [
                    _epoch(timestamp) - offset
                    for timestamp, offset in zip(timestamps, offsets, strict=True)
                ],
            }
        )
    )

    distribution = sparkov.check_unix_time(frame, listing_limit=2).offset_distribution

    assert [(entry.offset_seconds, entry.rows) for entry in distribution.offsets] == [
        (100, 3),
        (200, 2),
    ]
    assert distribution.distinct_offsets == 4
    assert distribution.truncated is True
    assert distribution.unlisted_offsets == 2
    assert distribution.unlisted_rows == 2
    assert distribution.rows_compared == 7
    assert (distribution.min_offset_seconds, distribution.max_offset_seconds) == (100, 400)


def test_a_listing_limit_below_one_is_refused() -> None:
    frame = sparkov.parse_timestamps(
        pd.DataFrame({"trans_date_trans_time": ["2019-01-01 00:00:00"], "unix_time": [0]})
    )
    with pytest.raises(ValueError, match="listing_limit"):
        sparkov.check_unix_time(frame, listing_limit=0)


@pytest.mark.parametrize(
    "offsets",
    [
        pytest.param([18000, 18000, 18000, 18000], id="nonzero-offset"),
        pytest.param([0, 0, 0], id="zero-offset"),
        pytest.param([3600, 3600, None], id="rows-without-unix-time-do-not-count"),
    ],
)
def test_one_distinct_offset_is_uniform(
    tmp_path: Path, adapter: SparkovAdapter, offsets: list[int | None]
) -> None:
    check = adapter.load_detailed(_corpus_with_offsets(tmp_path, "one", offsets)).unix_time

    assert check.offset_distribution.distinct_offsets == 1
    assert check.is_uniform_offset is True


@pytest.mark.parametrize(
    "offsets",
    [
        pytest.param([18000, 18000, 18000, 0], id="modal-offset-beside-exact-rows"),
        pytest.param([0, 0, 1], id="one-second-apart"),
        pytest.param([3600, -3600], id="equal-counts"),
    ],
)
def test_more_than_one_distinct_offset_is_never_uniform(
    tmp_path: Path, adapter: SparkovAdapter, offsets: list[int | None]
) -> None:
    check = adapter.load_detailed(_corpus_with_offsets(tmp_path, "several", offsets)).unix_time

    assert check.offset_distribution.distinct_offsets > 1
    assert check.is_uniform_offset is False


@pytest.mark.parametrize(
    "offsets",
    [
        pytest.param([18000, 18000, 18000, 0], id="exact-row-last"),
        pytest.param([0, 18000, 18000, 18000], id="exact-row-first"),
        pytest.param([18000, 0, 18000, 18000], id="exact-row-between"),
    ],
)
def test_uniformity_and_modal_fields_do_not_depend_on_row_order(
    tmp_path: Path, adapter: SparkovAdapter, offsets: list[int | None]
) -> None:
    """Three rows at +18000 and one matching the wall clock, in any order."""
    result = adapter.load_detailed(_corpus_with_offsets(tmp_path, "ordered", offsets))
    check = result.unix_time

    assert check.mismatches == 3
    assert check.modal_offset_seconds == 18000
    assert check.rows_at_modal_offset == 3
    assert check.is_uniform_offset is False
    assert sparkov.build_report(result).to_dict()["extra"]["unix_time_offset_is_uniform"] is False


def test_no_comparable_rows_is_not_uniform(tmp_path: Path, adapter: SparkovAdapter) -> None:
    """With no numeric unix_time there is no offset to share."""
    check = adapter.load_detailed(_corpus_with_offsets(tmp_path, "none", [None, None])).unix_time

    assert check.offset_distribution.rows_compared == 0
    assert check.mismatches == 0
    assert check.modal_offset_seconds is None
    assert check.rows_at_modal_offset == 0
    assert check.is_uniform_offset is False


def test_an_empty_frame_is_not_uniform() -> None:
    frame = sparkov.parse_timestamps(
        pd.DataFrame(
            {
                "trans_date_trans_time": pd.Series([], dtype="string"),
                "unix_time": pd.Series([], dtype="float64"),
            }
        )
    )

    check = sparkov.check_unix_time(frame)

    assert check.offset_distribution.rows_compared == 0
    assert check.modal_offset_seconds is None
    assert check.is_uniform_offset is False


def test_the_wall_clock_drives_the_transaction_timestamp(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    """A wrong epoch must not move a transaction: the wall clock is authoritative."""
    root = tmp_path / "raw" / "authority"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [_row(0, trans_num="t1", timestamp="2019-04-05 23:45:00", unix_time="0")],
    )

    dataset = adapter.load(root)
    assert dataset.transactions[0].created_at == datetime(2019, 4, 5, 23, 45, tzinfo=UTC)


def test_unix_time_order_never_orders_transactions(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    """The published fraudTrain.csv is stored in `unix_time` order.

    Its whole-day offset changes on 2019-02-28, so the wall clock steps
    backwards once in file order. Canonical order and every timestamp follow
    the wall clock, and the shift is only reported (Phase 5D decision 17).
    """
    day = 86_400
    in_file_order = [  # (trans_num, wall clock, whole days unix_time lags it by)
        ("t0", "2019-02-27 23:00:00", 2_557),
        ("t1", "2019-02-28 23:30:00", 2_557),
        ("t2", "2019-02-28 00:30:00", 2_556),
        ("t3", "2019-03-01 00:10:00", 2_556),
    ]
    unix_times = [_epoch(wall) - lag * day for _, wall, lag in in_file_order]
    assert unix_times == sorted(unix_times), "fixture must be in unix_time order"
    root = tmp_path / "raw" / "unix-order"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [
            _row(index, trans_num=trans_num, timestamp=wall, unix_time=str(unix_time))
            for index, ((trans_num, wall, _), unix_time) in enumerate(
                zip(in_file_order, unix_times, strict=True)
            )
        ],
    )

    result = adapter.load_detailed(root)

    transactions = result.dataset.transactions
    assert [tx.idempotency_key for tx in transactions] == ["t0", "t2", "t1", "t3"]
    walls = {trans_num: wall for trans_num, wall, _ in in_file_order}
    for tx in transactions:
        expected = datetime.fromisoformat(walls[tx.idempotency_key]).replace(tzinfo=UTC)
        assert tx.created_at == expected
    distribution = result.unix_time.offset_distribution
    assert distribution.distinct_offsets == 2
    assert {entry.offset_seconds for entry in distribution.offsets} == {2_556 * day, 2_557 * day}


# ---------------------------------------------------------------------------
# Label separation
# ---------------------------------------------------------------------------


def test_labels_live_outside_the_transaction_objects(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    fraud_ids = [tx_id for tx_id, label in dataset.labels.items() if label == 1]

    assert len(fraud_ids) == 1
    for tx in dataset.transactions:
        for attribute in ("is_fraud", "label", "target", "fraud_label"):
            assert not hasattr(tx, attribute)
        assert tx.fraud_score is None
        assert tx.fraud_decision is None


def test_label_values_are_binary_and_complete(adapter: SparkovAdapter, corpus: Path) -> None:
    dataset = _load(adapter, corpus)
    assert set(dataset.labels.values()) <= {0, 1}
    assert set(dataset.labels) == {tx.id for tx in dataset.transactions}


def test_fraud_rate_is_computed_from_the_labels(adapter: SparkovAdapter, corpus: Path) -> None:
    dataset = _load(adapter, corpus)
    assert dataset.fraud_count == 1
    assert dataset.fraud_rate == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Exclusions — never silent
# ---------------------------------------------------------------------------


def test_bad_rows_are_excluded_explicitly_and_counted(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = tmp_path / "raw" / "messy"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [
            _row(0, trans_num="good"),
            _row(1, trans_num="zero-amount", amt="0"),
            _row(2, trans_num="negative", amt="-5.00"),
            _row(3, trans_num="good"),  # duplicate transaction number
            _row(4, trans_num="bad-label", is_fraud="7"),
        ],
    )

    result = adapter.load_detailed(root)
    reasons = {record.reason: record.count for record in result.exclusions}

    assert result.raw_row_count == 5
    assert result.dataset.n_rows == 1
    assert reasons["non_positive_or_unparsable_amount"] == 2
    assert reasons["duplicate_trans_num"] == 1
    assert reasons["label_not_zero_or_one"] == 1
    assert result.excluded_row_count == 4


def test_zero_count_exclusion_reasons_are_still_reported(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    """A reader should see that the duplicate check ran, not infer it."""
    result = adapter.load_detailed(corpus)
    reasons = {record.reason: record.count for record in result.exclusions}

    assert reasons["duplicate_trans_num"] == 0
    assert reasons["missing_required_field"] == 0


def test_exclusions_are_recorded_in_provenance_preprocessing(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = tmp_path / "raw" / "messy2"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [_row(0, trans_num="good"), _row(1, trans_num="bad", amt="-1.00")],
    )

    dataset = _load(adapter, root)
    joined = " | ".join(dataset.provenance.preprocessing)
    assert "excluded 1 row(s): non_positive_or_unparsable_amount" in joined


def test_duplicate_exclusion_keeps_the_first_occurrence(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = tmp_path / "raw" / "dupes"
    root.mkdir(parents=True)
    _write_csv(
        root / "fraudTrain.csv",
        [
            _row(0, trans_num="dup", amt="10.00"),
            _row(1, trans_num="dup", amt="99.00"),
        ],
    )

    dataset = _load(adapter, root)
    assert dataset.n_rows == 1
    assert dataset.transactions[0].amount == Decimal("10.00")


def test_unmapped_category_fails_loudly(tmp_path: Path, adapter: SparkovAdapter) -> None:
    """A new source category is schema drift, not a row to quietly discard."""
    root = tmp_path / "raw" / "newcat"
    root.mkdir(parents=True)
    _write_csv(root / "fraudTrain.csv", [_row(0, trans_num="t1", category="crypto_net")])

    with pytest.raises(SparkovSchemaError, match="Unmapped Sparkov categories: crypto_net"):
        _load(adapter, root)


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def test_subsampling_keeps_whole_card_histories(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    """Row-level sampling would delete history and corrupt velocity features."""
    dataset = _load(adapter, corpus, max_entities=1)

    assert len(dataset.customers) == 1
    only_customer = next(iter(dataset.customers))
    counts = {tx.customer_id for tx in dataset.transactions}
    assert counts == {only_customer}

    full = _load(adapter, corpus)
    kept_card_rows = [tx for tx in full.transactions if tx.customer_id == only_customer]
    assert dataset.n_rows == len(kept_card_rows)


def test_subsampling_is_deterministic_for_a_seed(adapter: SparkovAdapter, corpus: Path) -> None:
    first = _load(adapter, corpus, max_entities=2, seed=7)
    second = _load(adapter, corpus, max_entities=2, seed=7)
    assert [tx.id for tx in first.transactions] == [tx.id for tx in second.transactions]


def test_different_seeds_can_select_different_cards(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    selections = {
        frozenset(_load(adapter, corpus, max_entities=1, seed=seed).customers)
        for seed in range(6)
    }
    assert len(selections) > 1


def test_subsample_record_is_written_into_provenance(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus, max_entities=2, seed=11)
    subsample = dataset.provenance.subsample

    assert subsample is not None
    assert subsample.strategy == "cards"
    assert subsample.seed == 11
    assert subsample.max_entities == 2
    assert subsample.selected_entities == 2


def test_no_subsampling_is_recorded_as_such(adapter: SparkovAdapter, corpus: Path) -> None:
    subsample = _load(adapter, corpus).provenance.subsample
    assert subsample is not None
    assert subsample.strategy == "none"
    assert subsample.selected_entities == 3


def test_cap_above_the_entity_count_keeps_everything(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus, max_entities=999)
    assert len(dataset.customers) == 3
    assert dataset.provenance.subsample is not None
    assert dataset.provenance.subsample.strategy == "none"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_marks_the_dataset_synthetic(adapter: SparkovAdapter, corpus: Path) -> None:
    provenance = _load(adapter, corpus).provenance

    assert provenance.origin is DataOrigin.SYNTHETIC
    assert provenance.name == "sparkov"
    assert provenance.license.startswith("CC0")
    assert provenance.label_field == "is_fraud"
    assert "not real card-transaction data" in provenance.notes.lower()


def test_provenance_records_counts_period_and_digests(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    provenance = _load(adapter, corpus).provenance

    assert provenance.row_count == 5
    assert provenance.fraud_count == 1
    assert provenance.fraud_rate == pytest.approx(0.2)
    assert provenance.period_start == datetime(2019, 1, 1, 8, 0, tzinfo=UTC)
    assert set(provenance.files) == {"fraudTrain.csv"}
    assert len(provenance.files["fraudTrain.csv"]) == 64


def test_provenance_lists_the_preprocessing_applied(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    steps = " | ".join(_load(adapter, corpus).provenance.preprocessing)
    assert "uuid5" in steps
    assert "fraud_" in steps
    assert "UTC" in steps


# ---------------------------------------------------------------------------
# Verification, schema drift, and multi-file loads
# ---------------------------------------------------------------------------


def test_hash_verification_refuses_unexpected_bytes(tmp_path: Path, corpus: Path) -> None:
    payload = json.loads(_manifest(tmp_path).read_text(encoding="utf-8"))
    payload["datasets"]["sparkov"]["files"]["fraudTrain.csv"]["sha256"] = "f" * 64
    manifest_path = tmp_path / "pinned.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    verifying = SparkovAdapter(manifest_path=manifest_path, verify_hashes=True)
    with pytest.raises(ManifestError, match="hash_mismatch"):
        verifying.load(corpus)


def test_missing_source_file_is_reported_clearly(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    with pytest.raises(SparkovSchemaError, match="Expected source file not found"):
        _load(adapter, tmp_path / "nowhere")


def test_missing_column_is_reported_with_its_name(tmp_path: Path, adapter: SparkovAdapter) -> None:
    root = tmp_path / "raw" / "truncated"
    root.mkdir(parents=True)
    header = [column for column in HEADER if column != "unix_time"]
    rows = [[value for index, value in enumerate(_row(0, trans_num="t1")) if HEADER[index] != "unix_time"]]
    _write_csv(root / "fraudTrain.csv", rows, header=header)

    with pytest.raises(SparkovSchemaError, match="missing expected Sparkov columns: unix_time"):
        _load(adapter, root)


def test_both_published_files_are_concatenated(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "both"
    root.mkdir(parents=True)
    _write_csv(root / "fraudTrain.csv", [_row(0, trans_num="a", timestamp="2019-01-01 08:00:00")])
    _write_csv(root / "fraudTest.csv", [_row(0, trans_num="b", timestamp="2020-07-01 08:00:00")])

    adapter = SparkovAdapter(
        manifest_path=_manifest(tmp_path, ("fraudTrain.csv", "fraudTest.csv")),
        verify_hashes=False,
    )
    dataset = adapter.load(root)

    assert {tx.idempotency_key for tx in dataset.transactions} == {"a", "b"}
    assert dataset.transactions[0].created_at < dataset.transactions[1].created_at


def test_loaded_dataset_satisfies_the_canonical_contract(
    adapter: SparkovAdapter, corpus: Path
) -> None:
    dataset = _load(adapter, corpus)
    assert dataset.validate() is dataset


def test_empty_corpus_is_rejected_by_provenance_rather_than_produced_silently(
    tmp_path: Path, adapter: SparkovAdapter
) -> None:
    root = tmp_path / "raw" / "empty"
    root.mkdir(parents=True)
    _write_csv(root / "fraudTrain.csv", [])

    dataset = _load(adapter, root)
    assert dataset.n_rows == 0
    assert dataset.provenance.row_count == 0


# ---------------------------------------------------------------------------
# Field inventory
# ---------------------------------------------------------------------------


def test_field_inventory_covers_every_source_column() -> None:
    notes = sparkov.field_inventory()
    covered = " ".join(note.field for note in notes)
    for column in SPARKOV_COLUMNS:
        base = column.split("_")[0] if column in {"lat", "long"} else column
        assert base in covered, f"{column} missing from the field inventory"


def test_field_inventory_flags_the_unavailable_fields() -> None:
    notes = {note.field: note for note in sparkov.field_inventory()}
    assert notes["Customer.risk_tier"].status is FieldStatus.UNAVAILABLE
    assert notes["Customer.account_age_days"].status is FieldStatus.UNAVAILABLE
    assert "invent signal" in notes["Customer.account_age_days"].detail


def test_field_inventory_marks_dropped_cardholder_detail() -> None:
    notes = {note.field: note for note in sparkov.field_inventory()}
    for column in ("first", "last", "dob", "street"):
        assert notes[column].status is FieldStatus.DROPPED


def test_adapter_does_not_import_scoring_or_session_code() -> None:
    """The adapter builds transient objects; it must not reach into serving."""
    source = Path(sparkov.__file__).read_text(encoding="utf-8")
    assert "app.services" not in source
    assert "Session" not in source


def test_canonical_objects_are_transient(adapter: SparkovAdapter, corpus: Path) -> None:
    from sqlalchemy.orm import object_session

    dataset = _load(adapter, corpus)
    assert all(object_session(tx) is None for tx in dataset.transactions)


def test_contract_violation_surfaces_as_a_contract_error() -> None:
    """SparkovSchemaError is a DatasetContractError, so callers catch one type."""
    assert issubclass(SparkovSchemaError, DatasetContractError)
