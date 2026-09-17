"""Sparkov adapter — an external *synthetic* benchmark, into canonical form.

The source is Kaggle `kartik2112/fraud-detection`: simulated card transactions
generated with Brandon Harris's Sparkov tool, covering 1 Jan 2019 – 31 Dec 2020
for 1,000 cards and 800 merchants, published CC0. It is **not** real card data,
and nothing here should describe it as such. Its value is that it was generated
by someone else's simulator, so it tests whether this project's features and
rules transfer beyond the generator they were designed against.

What this adapter does *not* do is as important as what it does. It does not
touch scoring, it does not invent fields the source lacks, and it does not
quietly drop rows: every exclusion is counted, reasoned, and reported.

**Which clock is authoritative.** `trans_date_trans_time`, and only it (Phase
5D decision 17). Every transaction timestamp — and so every ordering, split,
history window and temporal feature — reads the wall clock. `unix_time` is a
diagnostic: its offset from the wall clock is measured and reported, never used
as time and never corrected. The generator's source suggested that offset
would be a timezone; on the published files it is a whole 2,556 or 2,557 days,
so `unix_time` cannot stand in as a clock.
"""
from __future__ import annotations

import argparse
import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from ml.datasets.base import (
    CanonicalDataset,
    DataOrigin,
    DatasetContractError,
    DatasetProvenance,
    Subsample,
)
from ml.datasets.manifest import (
    DEFAULT_MANIFEST_PATH,
    DatasetManifestEntry,
    load_manifest_entry,
    require_verified,
    sha256_file,
)
from ml.datasets.quality import (
    ExclusionRecord,
    FieldNote,
    FieldStatus,
    QualityReport,
    build_quality_report,
)
from ml.datasets.registry import register_adapter
from ml.paths import RAW_DATA_DIR, run_dir
from ml.synthesis.merchants import CATEGORIES

log = logging.getLogger("ml.datasets.sparkov")

DATASET_NAME = "sparkov"
DATASET_VERSION = "kaggle-2020-08-05"

# Ids are derived from source keys with uuid5, so two runs over the same file
# produce identical ids and nothing card-like is ever stored or logged.
SPARKOV_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://www.kaggle.com/datasets/kartik2112/fraud-detection"
)

# The 22 named columns the published CSVs carry, verified against the Kaggle
# data card and independent descriptions of the files. The files also carry an
# unnamed leading index column, which is ignored.
SPARKOV_COLUMNS: tuple[str, ...] = (
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
)

# Columns this adapter actually reads. The rest are cardholder PII or
# coordinates with no canonical home; they are reported as dropped.
_USED_COLUMNS: tuple[str, ...] = (
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "trans_num",
    "unix_time",
    "is_fraud",
)

# Sparkov's 14 categories onto this project's 12-category taxonomy.
#
# Category carries merchant *type* here; channel is carried separately by
# is_card_present, which is why grocery_pos and grocery_net both land in
# GROCERY. `health_fitness` has no exact equivalent, so it goes to
# ENTERTAINMENT: a discretionary recreation service, MEDIUM risk, in the same
# 7xxx MCC family as health clubs. ONLINE_SERVICE would contradict the axis
# split above — its MCC 5968 (direct marketing/subscription) asserts a
# card-not-present channel, while health_fitness carries no _pos/_net suffix
# and is treated as card-present. This is still a judgment call, and it is
# reported as one in the field inventory.
CATEGORY_MAP: Mapping[str, str] = {
    "grocery_pos": "GROCERY",
    "grocery_net": "GROCERY",
    "food_dining": "RESTAURANT",
    "shopping_pos": "RETAIL",
    "shopping_net": "RETAIL",
    "misc_pos": "RETAIL",
    "misc_net": "RETAIL",
    "home": "RETAIL",
    "kids_pets": "RETAIL",
    "personal_care": "RETAIL",
    "gas_transport": "GAS_STATION",
    "entertainment": "ENTERTAINMENT",
    "travel": "TRAVEL",
    "health_fitness": "ENTERTAINMENT",
}

# Every merchant name in the source is prefixed "fraud_" — on legitimate and
# fraudulent rows alike, so it carries no signal. Stripped for readability;
# keeping it would invite a future text-based model to latch onto the token.
_MERCHANT_PREFIX = "fraud_"

_CARD_PRESENT_SUFFIX: Mapping[str, bool] = {"_pos": True, "_net": False}

# The generator writes a date and a time of day, which the publisher joined
# into one column. One explicit format, parsed once: per-element inference
# would be slow over 1.85M rows and could resolve ambiguous strings
# inconsistently, which is the worst possible failure for a timestamp that
# every velocity feature is computed from.
SPARKOV_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Where the single parse result lives while the frame is being processed.
# No leading underscore: pandas' itertuples renames such columns positionally.
_PARSED_AT_COLUMN = "parsed_at"

# Constants for fields the source does not carry. Chosen to be inert rather
# than plausible: a fabricated account age would feed a live feature with
# invented signal.
_DEFAULT_COUNTRY = "US"
_DEFAULT_RISK_TIER = "LOW"
_DEFAULT_ACCOUNT_AGE_DAYS = 0
_DEFAULT_CURRENCY = "USD"
_DEFAULT_PAYMENT_METHOD = "CARD"
_DEFAULT_STATUS = "APPROVED"

_UNIX_TIME_TOLERANCE_SECONDS = 1.0

# The offset distribution lists at most this many distinct offsets, so the
# report stays a few kilobytes however scattered `unix_time` turns out to be.
# Totals and the observed range always cover every row; a truncated listing
# says exactly how many offsets and rows it left out.
UNIX_TIME_OFFSET_LISTING_LIMIT = 50

UNIX_TIME_OFFSET_DEFINITION = (
    "trans_date_trans_time parsed as UTC, in epoch seconds, minus unix_time"
)

# Measured, not guessed: building canonical Transaction objects plus their
# label entries costs ~1,983 bytes per row (tracemalloc over 20k rows), so the
# published 1.85M-row corpus needs roughly 3.7 GB for those objects alone,
# before the source frame and the feature matrix.
ESTIMATED_BYTES_PER_ROW = 2_000

# A full-corpus load above this many rows must be asked for explicitly. This
# is a row count, not a memory limit: the repository cannot know how much RAM
# a given machine has, so it refuses to pretend. The threshold sits far above
# any fixture or subsampled development run and far below the published
# corpus, so the guard only ever fires for the case it is about.
FULL_CORPUS_ROW_LIMIT = 500_000


class SparkovSchemaError(DatasetContractError):
    """The source file does not look like the Sparkov dataset we expect."""


class CorpusTooLargeError(DatasetContractError):
    """A full-corpus load was attempted without asking for one."""


@dataclass(frozen=True)
class UnixTimeOffsetCount:
    """One distinct offset and the number of rows observed at it."""

    offset_seconds: int | float
    rows: int


@dataclass(frozen=True)
class UnixTimeOffsetDistribution:
    """Every distinct offset between the wall clock and `unix_time`, with counts.

    Offsets are listed most frequent first, ties broken by the smaller offset,
    so the same rows serialise identically whatever order the files hold them
    in. At most `listing_limit` offsets are listed; `rows_compared`,
    `distinct_offsets` and the minimum and maximum always describe every row
    that has a numeric `unix_time`.

    This is an observation, not a verdict. What a given distribution means is
    decided by inspecting it, never here.
    """

    rows_compared: int
    rows_without_unix_time: int
    distinct_offsets: int
    min_offset_seconds: int | float | None
    max_offset_seconds: int | float | None
    offsets: tuple[UnixTimeOffsetCount, ...]
    listing_limit: int

    @property
    def unlisted_offsets(self) -> int:
        return self.distinct_offsets - len(self.offsets)

    @property
    def unlisted_rows(self) -> int:
        return self.rows_compared - sum(entry.rows for entry in self.offsets)

    @property
    def truncated(self) -> bool:
        return self.unlisted_offsets > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset_definition": UNIX_TIME_OFFSET_DEFINITION,
            "rows_compared": self.rows_compared,
            "rows_without_unix_time": self.rows_without_unix_time,
            "distinct_offsets": self.distinct_offsets,
            "min_offset_seconds": self.min_offset_seconds,
            "max_offset_seconds": self.max_offset_seconds,
            "listing_limit": self.listing_limit,
            "offsets": [
                {"offset_seconds": entry.offset_seconds, "rows": entry.rows}
                for entry in self.offsets
            ],
            "truncated": self.truncated,
            "unlisted_offsets": self.unlisted_offsets,
            "unlisted_rows": self.unlisted_rows,
        }


@dataclass(frozen=True)
class UnixTimeCheck:
    """How `unix_time` relates to the parsed wall clock across a corpus."""

    mismatches: int
    modal_offset_seconds: int | None
    rows_at_modal_offset: int
    offset_distribution: UnixTimeOffsetDistribution

    @property
    def is_uniform_offset(self) -> bool:
        """True when every row with a numeric `unix_time` has the same offset.

        Read from the complete distribution, so any second offset makes it
        false, however few rows it holds and however close it sits to the
        first. With no comparable rows there is no shared offset, so it is
        false then too.
        """
        return self.offset_distribution.distinct_offsets == 1


@dataclass(frozen=True)
class SparkovLoadResult:
    """A canonical dataset plus the accounting behind it."""

    dataset: CanonicalDataset
    raw_row_count: int
    exclusions: tuple[ExclusionRecord, ...]
    field_notes: tuple[FieldNote, ...]
    unix_time: UnixTimeCheck
    multi_category_merchants: int = 0

    @property
    def excluded_row_count(self) -> int:
        return sum(record.count for record in self.exclusions)


class SparkovAdapter:
    """Loads the Sparkov CSVs into a `CanonicalDataset`."""

    name = DATASET_NAME

    def __init__(
        self,
        *,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        verify_hashes: bool = True,
        allow_full_corpus: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.verify_hashes = verify_hashes
        self.allow_full_corpus = allow_full_corpus

    def load(
        self,
        root: Path,
        *,
        max_entities: int | None = None,
        seed: int = 42,
    ) -> CanonicalDataset:
        """Protocol entry point — the canonical dataset only."""
        return self.load_detailed(root, max_entities=max_entities, seed=seed).dataset

    def load_detailed(
        self,
        root: Path,
        *,
        max_entities: int | None = None,
        seed: int = 42,
    ) -> SparkovLoadResult:
        """Load, and keep the exclusion and field accounting for the report."""
        entry = load_manifest_entry(self.name, path=self.manifest_path)
        file_digests = (
            require_verified(entry, root)
            if self.verify_hashes
            else _digests_without_verification(entry, root)
        )

        frame = _read_frames(root, entry.filenames)
        raw_row_count = len(frame)
        log.info("Read %d rows from %s", raw_row_count, ", ".join(entry.filenames))

        frame = parse_timestamps(frame)
        frame, exclusions = _apply_exclusions(frame)
        frame, subsample = _subsample_by_card(frame, max_entities=max_entities, seed=seed)
        _guard_projected_size(
            len(frame), subsample=subsample, allow_full_corpus=self.allow_full_corpus
        )

        customers = _build_customers(frame["cc_num"])
        merchants, multi_category_merchants = _build_merchants(frame[["merchant", "category"]])
        transactions, labels = _build_transactions(frame)
        unix_time = check_unix_time(frame)

        provenance = _build_provenance(
            entry=entry,
            file_digests=file_digests,
            transactions=transactions,
            labels=labels,
            subsample=subsample,
            exclusions=exclusions,
        )

        dataset = CanonicalDataset(
            customers=customers,
            merchants=merchants,
            transactions=transactions,
            labels=labels,
            provenance=provenance,
        ).validate()

        return SparkovLoadResult(
            dataset=dataset,
            raw_row_count=raw_row_count,
            exclusions=exclusions,
            field_notes=field_inventory(),
            unix_time=unix_time,
            multi_category_merchants=multi_category_merchants,
        )


def field_inventory() -> tuple[FieldNote, ...]:
    """What happened to every source field, and what canonical fields lack a source.

    This inventory is the honest half of an adapter: it is the difference
    between "the model scored X on Sparkov" and "the model scored X on Sparkov
    with four of its seventeen features held constant".
    """
    dropped_pii = ("first", "last", "gender", "street", "city", "zip", "job", "dob")
    notes: list[FieldNote] = [
        FieldNote("trans_date_trans_time", FieldStatus.MAPPED, "Transaction.created_at, UTC"),
        FieldNote("cc_num", FieldStatus.MAPPED, "uuid5 -> Customer.id; raw value never stored"),
        FieldNote("merchant", FieldStatus.MAPPED, "uuid5 -> Merchant.id; 'fraud_' prefix stripped"),
        FieldNote("category", FieldStatus.MAPPED, "Merchant.category via the 14 -> 12 taxonomy map"),
        FieldNote("amt", FieldStatus.MAPPED, "Transaction.amount, Decimal to 2dp"),
        FieldNote("trans_num", FieldStatus.MAPPED, "uuid5 -> Transaction.id; idempotency_key"),
        FieldNote("is_fraud", FieldStatus.MAPPED, "labels map, never an attribute of a row"),
        FieldNote("unix_time", FieldStatus.DERIVED, "cross-check against the parsed timestamp only"),
    ]
    notes += [
        FieldNote(name, FieldStatus.DROPPED, "cardholder detail with no canonical field")
        for name in dropped_pii
    ]
    notes += [
        FieldNote("state", FieldStatus.DROPPED, "superseded by a constant country"),
        FieldNote("city_pop", FieldStatus.DROPPED, "no canonical field"),
        FieldNote("lat/long", FieldStatus.DROPPED, "no canonical coordinate fields (v3 candidate)"),
        FieldNote("merch_lat/merch_long", FieldStatus.DROPPED, "no canonical coordinate fields"),
        FieldNote(
            "Customer.country / Merchant.country / Transaction.country",
            FieldStatus.DERIVED,
            f"constant {_DEFAULT_COUNTRY!r}: the source is US-only, so both country-mismatch "
            "features are structurally zero here",
        ),
        FieldNote(
            "Transaction.is_card_present",
            FieldStatus.DERIVED,
            "from the category suffix: _pos -> True, _net -> False, otherwise True by assumption",
        ),
        FieldNote(
            "Merchant.risk_rating / mcc",
            FieldStatus.DERIVED,
            "from the mapped category's taxonomy entry — never from observed fraud rate",
        ),
        FieldNote(
            "Customer.risk_tier",
            FieldStatus.UNAVAILABLE,
            f"constant {_DEFAULT_RISK_TIER!r}: no equivalent in the source, so the risk-tier "
            "feature is constant",
        ),
        FieldNote(
            "Customer.account_age_days",
            FieldStatus.UNAVAILABLE,
            "constant 0: the source has a birth date, not an account age; deriving one from "
            "dob would invent signal",
        ),
        FieldNote(
            "Customer.email / full_name",
            FieldStatus.DERIVED,
            "placeholders built from the hashed id; no source identity is carried over",
        ),
        FieldNote(
            "Transaction.currency / payment_method / status",
            FieldStatus.DERIVED,
            f"constants {_DEFAULT_CURRENCY!r} / {_DEFAULT_PAYMENT_METHOD!r} / {_DEFAULT_STATUS!r}",
        ),
    ]
    return tuple(notes)


def _read_frames(root: Path, filenames: Sequence[str]) -> pd.DataFrame:
    frames = [_read_one(root / filename) for filename in filenames]
    return pd.concat(frames, ignore_index=True)


def _read_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SparkovSchemaError(f"Expected source file not found: {path}")

    header = pd.read_csv(path, nrows=0)
    present = {str(column) for column in header.columns}
    missing = [column for column in SPARKOV_COLUMNS if column not in present]
    if missing:
        raise SparkovSchemaError(
            f"{path.name} is missing expected Sparkov columns: {', '.join(missing)}. "
            "The publisher may have changed the file; re-check the dataset card before loading."
        )

    return pd.read_csv(
        path,
        usecols=list(_USED_COLUMNS),
        dtype={
            "cc_num": "string",
            "merchant": "string",
            "category": "string",
            "trans_num": "string",
            "trans_date_trans_time": "string",
        },
    )


def parse_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse `trans_date_trans_time` once, with one explicit format.

    Every later step — the exclusion gate, the transaction objects, the
    unix_time cross-check — reads this column instead of re-parsing. Parsing
    the same field twice with two different parsers is how a row passes
    validation and then fails, or worse, is interpreted two ways.

    Unparsable values become NaT here and are excluded (and counted) below,
    rather than raising midway through object construction.
    """
    parsed = pd.to_datetime(
        frame["trans_date_trans_time"], format=SPARKOV_TIMESTAMP_FORMAT, errors="coerce"
    )
    return frame.assign(**{_PARSED_AT_COLUMN: parsed})


def _apply_exclusions(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[ExclusionRecord, ...]]:
    """Remove rows that cannot become valid canonical transactions.

    Every removal is counted and reasoned. Rows are never dropped silently,
    and the order of the checks is fixed so two runs exclude the same rows.
    """
    exclusions: list[ExclusionRecord] = []

    required = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "trans_num"]
    missing_mask = frame[required].isna().any(axis=1)
    frame, record = _exclude(frame, missing_mask, "missing_required_field", "trans_num")
    exclusions.append(record)

    amounts = pd.to_numeric(frame["amt"], errors="coerce")
    invalid_amount = amounts.isna() | (amounts <= 0)
    frame, record = _exclude(frame, invalid_amount, "non_positive_or_unparsable_amount", "trans_num")
    exclusions.append(record)

    labels = pd.to_numeric(frame["is_fraud"], errors="coerce")
    invalid_label = ~labels.isin([0, 1])
    frame, record = _exclude(frame, invalid_label, "label_not_zero_or_one", "trans_num")
    exclusions.append(record)

    frame, record = _exclude(
        frame, frame[_PARSED_AT_COLUMN].isna(), "unparsable_timestamp", "trans_num"
    )
    exclusions.append(record)

    # Keep the first occurrence in file order so a re-run keeps the same row.
    duplicates = frame["trans_num"].duplicated(keep="first")
    frame, record = _exclude(frame, duplicates, "duplicate_trans_num", "trans_num")
    exclusions.append(record)

    unknown = ~frame["category"].isin(list(CATEGORY_MAP))
    if bool(unknown.any()):
        unseen = sorted({str(value) for value in frame.loc[unknown, "category"].unique()})
        raise SparkovSchemaError(
            f"Unmapped Sparkov categories: {', '.join(unseen)}. Every category must have a "
            "canonical home — extend CATEGORY_MAP deliberately rather than dropping the rows."
        )

    return frame.reset_index(drop=True), tuple(exclusions)


def _exclude(
    frame: pd.DataFrame, mask: pd.Series[bool], reason: str, id_column: str
) -> tuple[pd.DataFrame, ExclusionRecord]:
    count = int(mask.sum())
    examples: tuple[str, ...] = ()
    if count:
        examples = tuple(str(value) for value in frame.loc[mask, id_column].head(3))
        log.warning("Excluding %d row(s): %s", count, reason)
    return frame.loc[~mask], ExclusionRecord(reason=reason, count=count, examples=examples)


def _subsample_by_card(
    frame: pd.DataFrame, *, max_entities: int | None, seed: int
) -> tuple[pd.DataFrame, Subsample]:
    """Keep whole card histories, never individual rows.

    Sampling rows at random would delete a card's past, and every velocity and
    recency feature is computed from exactly that past. The unit of sampling
    has to be the entity that owns the history.
    """
    cards = sorted({str(value) for value in frame["cc_num"].unique()})
    if max_entities is None or max_entities >= len(cards):
        return frame, Subsample(
            strategy="none",
            seed=seed,
            max_entities=max_entities,
            selected_entities=len(cards) or None,
        )

    chosen = set(Random(seed).sample(cards, max_entities))
    subset = frame[frame["cc_num"].isin(chosen)].reset_index(drop=True)
    log.info("Subsampled %d of %d cards (seed=%d)", len(chosen), len(cards), seed)
    return subset, Subsample(
        strategy="cards",
        seed=seed,
        max_entities=max_entities,
        selected_entities=len(chosen),
    )


def projected_bytes(rows: int) -> int:
    """Rough memory the canonical objects for `rows` will occupy."""
    return rows * ESTIMATED_BYTES_PER_ROW


def _guard_projected_size(
    rows: int, *, subsample: Subsample, allow_full_corpus: bool
) -> None:
    """Refuse a large unsubsampled load unless it was asked for explicitly.

    Canonical objects are built in memory, so a full-corpus load is a
    multi-gigabyte decision. Rather than guess how much RAM is available —
    which this code cannot know — it declines to make that decision silently
    on the caller's behalf.
    """
    log.info(
        "Projected canonical footprint: %d rows x ~%d bytes = %.2f GB",
        rows,
        ESTIMATED_BYTES_PER_ROW,
        projected_bytes(rows) / 1e9,
    )
    if allow_full_corpus or subsample.strategy != "none" or rows <= FULL_CORPUS_ROW_LIMIT:
        return
    raise CorpusTooLargeError(
        f"Loading {rows:,} rows without subsampling needs roughly "
        f"{projected_bytes(rows) / 1e9:.1f} GB for the canonical objects alone, "
        f"plus the source frame and the feature matrix. Pass --max-cards N for a "
        f"sampled run (histories are kept whole), or opt in explicitly with "
        f"--full-corpus once you know the machine can take it."
    )


def customer_id_for(cc_num: str) -> str:
    """Deterministic, non-reversible id for a card number."""
    return str(uuid.uuid5(SPARKOV_NAMESPACE, f"card:{cc_num}"))


def merchant_id_for(merchant_name: str, category: str) -> str:
    """Merchant identity is (name, canonical category).

    A canonical `Merchant` carries exactly one category. Keying on the name
    alone would let a name that appears under two categories take whichever
    one happened to be seen first — an arbitrary attribute silently attached
    to an id. Including the category makes the identity honest; the two source
    grocery channels still collapse into one merchant because both map to the
    same canonical category.
    """
    return str(uuid.uuid5(SPARKOV_NAMESPACE, f"merchant:{merchant_name}|{category}"))


def transaction_id_for(trans_num: str) -> str:
    return str(uuid.uuid5(SPARKOV_NAMESPACE, f"tx:{trans_num}"))


def clean_merchant_name(raw: str) -> str:
    """Strip the generator's `fraud_` prefix, which is on every merchant."""
    return raw[len(_MERCHANT_PREFIX) :] if raw.startswith(_MERCHANT_PREFIX) else raw


def is_card_present_for(category: str) -> bool:
    for suffix, value in _CARD_PRESENT_SUFFIX.items():
        if category.endswith(suffix):
            return value
    return True


def _build_customers(cc_nums: Iterable[object]) -> dict[str, Customer]:
    customers: dict[str, Customer] = {}
    for raw in {str(value) for value in cc_nums}:
        customer_id = customer_id_for(raw)
        customers[customer_id] = Customer(
            id=customer_id,
            email=f"{customer_id}@sparkov.invalid",
            full_name=f"Sparkov cardholder {customer_id[:8]}",
            country=_DEFAULT_COUNTRY,
            risk_tier=_DEFAULT_RISK_TIER,
            account_age_days=_DEFAULT_ACCOUNT_AGE_DAYS,
        )
    return customers


def _build_merchants(frame: pd.DataFrame) -> tuple[dict[str, Merchant], int]:
    """Build merchants and count names that span more than one category."""
    merchants: dict[str, Merchant] = {}
    pairs = {
        (clean_merchant_name(str(row.merchant)), CATEGORY_MAP[str(row.category)])
        for row in frame.itertuples(index=False)
    }
    categories_per_name: dict[str, set[str]] = {}
    for name, category in sorted(pairs):
        categories_per_name.setdefault(name, set()).add(category)
        merchant_id = merchant_id_for(name, category)
        taxonomy = CATEGORIES[category]
        merchants[merchant_id] = Merchant(
            id=merchant_id,
            name=name,
            category=category,
            mcc=str(taxonomy["mcc"]),
            country=_DEFAULT_COUNTRY,
            risk_rating=str(taxonomy["risk"]),
        )
    spanning = sum(1 for categories in categories_per_name.values() if len(categories) > 1)
    if spanning:
        log.info("%d merchant name(s) appear under more than one canonical category", spanning)
    return merchants, spanning


def _build_transactions(frame: pd.DataFrame) -> tuple[list[Transaction], dict[str, int]]:
    transactions: list[Transaction] = []
    labels: dict[str, int] = {}

    for row in frame.itertuples(index=False):
        trans_num = str(row.trans_num)
        tx_id = transaction_id_for(trans_num)
        merchant_name = clean_merchant_name(str(row.merchant))
        source_category = str(row.category)

        transactions.append(
            Transaction(
                id=tx_id,
                idempotency_key=trans_num,
                customer_id=customer_id_for(str(row.cc_num)),
                merchant_id=merchant_id_for(merchant_name, CATEGORY_MAP[source_category]),
                amount=_to_amount(row.amt),
                currency=_DEFAULT_CURRENCY,
                status=_DEFAULT_STATUS,
                payment_method=_DEFAULT_PAYMENT_METHOD,
                country=_DEFAULT_COUNTRY,
                is_card_present=is_card_present_for(source_category),
                created_at=_to_utc(getattr(row, _PARSED_AT_COLUMN)),
            )
        )
        # The label is written to a separate map, never onto the row above.
        labels[tx_id] = int(row.is_fraud)

    transactions.sort(key=lambda tx: (tx.created_at, tx.id))
    return transactions, labels


def _to_amount(value: object) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as exc:  # pragma: no cover - excluded upstream
        raise SparkovSchemaError(f"Unparsable amount {value!r}") from exc


def _to_utc(value: Any) -> datetime:
    """Normalise one already-parsed timestamp to a tz-aware UTC datetime.

    The generator emits a naive wall clock (see the clock-authority note in
    the module docstring), so UTC is applied explicitly here rather than being
    inherited from whatever locale the reader happens to run in.
    """
    timestamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if not isinstance(timestamp, datetime):  # pragma: no cover - excluded upstream
        raise SparkovSchemaError(f"Unparsable timestamp {value!r}")
    return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)


def check_unix_time(
    frame: pd.DataFrame, *, listing_limit: int = UNIX_TIME_OFFSET_LISTING_LIMIT
) -> UnixTimeCheck:
    """Compare `unix_time` against the parsed wall clock and record what is seen.

    A row's offset is its parsed wall clock, in UTC epoch seconds, minus its
    `unix_time`. Two views of those offsets come back: the modal offset with
    its row count and the rows more than a second out, as before, and the
    complete distribution of distinct offsets.

    Nothing here judges the result. One shared offset, several offsets and
    scattered values are all reported the same way; what they mean is decided
    by inspecting the distribution observed on the real corpus.
    """
    if listing_limit < 1:
        raise ValueError(f"listing_limit must be at least 1, got {listing_limit}.")
    if frame.empty:
        return UnixTimeCheck(
            mismatches=0,
            modal_offset_seconds=None,
            rows_at_modal_offset=0,
            offset_distribution=_offset_distribution(
                pd.Series(dtype="float64"), listing_limit=listing_limit
            ),
        )

    # Cast to second resolution before taking the integer view: the underlying
    # unit of a datetime64 column is version-dependent (ns or us), and dividing
    # by a hard-coded factor silently turns every row into a mismatch.
    epoch_seconds = frame[_PARSED_AT_COLUMN].astype("datetime64[s]").astype("int64")
    deltas = epoch_seconds - pd.to_numeric(frame["unix_time"], errors="coerce")
    mismatches = int((deltas.abs() > _UNIX_TIME_TOLERANCE_SECONDS).sum())
    distribution = _offset_distribution(deltas, listing_limit=listing_limit)

    modes = deltas.mode(dropna=True)
    if modes.empty:
        return UnixTimeCheck(
            mismatches=mismatches,
            modal_offset_seconds=None,
            rows_at_modal_offset=0,
            offset_distribution=distribution,
        )
    modal_offset = int(modes.iloc[0])
    return UnixTimeCheck(
        mismatches=mismatches,
        modal_offset_seconds=modal_offset,
        rows_at_modal_offset=int((deltas == modal_offset).sum()),
        offset_distribution=distribution,
    )


def _offset_distribution(
    deltas: pd.Series[float], *, listing_limit: int
) -> UnixTimeOffsetDistribution:
    """Count every distinct offset; rows without a numeric `unix_time` are counted apart."""
    observed = deltas.dropna()
    counts = sorted(
        (
            UnixTimeOffsetCount(offset_seconds=_as_seconds(offset), rows=int(rows))
            for offset, rows in observed.value_counts().items()
        ),
        key=lambda entry: (-entry.rows, entry.offset_seconds),
    )
    has_values = not observed.empty
    return UnixTimeOffsetDistribution(
        rows_compared=len(observed),
        rows_without_unix_time=len(deltas) - len(observed),
        distinct_offsets=len(counts),
        min_offset_seconds=_as_seconds(observed.min()) if has_values else None,
        max_offset_seconds=_as_seconds(observed.max()) if has_values else None,
        offsets=tuple(counts[:listing_limit]),
        listing_limit=listing_limit,
    )


def _as_seconds(value: Any) -> int | float:
    """An offset as an exact JSON number: integral values as int, others unrounded."""
    number = float(value)
    return int(number) if number.is_integer() else number


def _build_provenance(
    *,
    entry: DatasetManifestEntry,
    file_digests: Mapping[str, str],
    transactions: Sequence[Transaction],
    labels: Mapping[str, int],
    subsample: Subsample,
    exclusions: Sequence[ExclusionRecord],
) -> DatasetProvenance:
    period_start = transactions[0].created_at if transactions else None
    period_end = transactions[-1].created_at if transactions else None

    preprocessing = [
        "ids derived with uuid5; raw card numbers never stored",
        "merchant names stripped of the generator's 'fraud_' prefix",
        "categories mapped onto the 12-category canonical taxonomy",
        "amounts cast to Decimal at 2dp",
        "naive timestamps interpreted as UTC",
        "card-present derived from the category suffix",
    ]
    preprocessing += [
        f"excluded {record.count} row(s): {record.reason}"
        for record in exclusions
        if record.count
    ]

    return DatasetProvenance(
        name=entry.name,
        version=DATASET_VERSION,
        origin=DataOrigin.SYNTHETIC,
        source_url=entry.source_url,
        citation=entry.citation,
        license=entry.license,
        label_field=entry.label_field,
        label_definition=entry.label_definition,
        retrieved_at=datetime.now(UTC).replace(microsecond=0),
        files=dict(file_digests),
        row_count=len(transactions),
        fraud_count=sum(labels.values()),
        period_start=period_start,
        period_end=period_end,
        subsample=subsample,
        preprocessing=tuple(preprocessing),
        notes=entry.notes,
    )


def _digests_without_verification(entry: DatasetManifestEntry, root: Path) -> dict[str, str]:
    """Hash what we actually read, without comparing against the pin.

    Used for fixtures and for a first look at freshly downloaded files. The
    digest still lands in provenance, so the run records which bytes produced
    it even when nothing was there to check them against.
    """
    return {
        name: sha256_file(root / name) for name in entry.filenames if (root / name).exists()
    }


register_adapter(SparkovAdapter())


def build_report(result: SparkovLoadResult) -> QualityReport:
    """Summarise a load, including the checks that are Sparkov-specific."""
    return build_quality_report(
        result.dataset,
        raw_row_count=result.raw_row_count,
        exclusions=result.exclusions,
        field_notes=result.field_notes,
        extra={
            "unix_time_mismatches": result.unix_time.mismatches,
            "unix_time_modal_offset_seconds": result.unix_time.modal_offset_seconds,
            "unix_time_rows_at_modal_offset": result.unix_time.rows_at_modal_offset,
            "unix_time_offset_is_uniform": result.unix_time.is_uniform_offset,
            "unix_time_offset_distribution": result.unix_time.offset_distribution.to_dict(),
            "authoritative_clock": "trans_date_trans_time",
            "multi_category_merchant_names": result.multi_category_merchants,
        },
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the Sparkov benchmark into canonical form")
    parser.add_argument(
        "--root",
        type=Path,
        default=RAW_DATA_DIR / DATASET_NAME,
        help="Directory holding the source CSVs (default: ml/data/raw/sparkov)",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Keep only N cards, with their full histories (default: every card)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for card subsampling")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write quality_report.json into the run directory",
    )
    parser.add_argument(
        "--run-name",
        default=f"{DATASET_NAME}_v1",
        help="Run directory under ml/artifacts/runs/ for the report",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Load without verifying against the manifest (development only)",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help=(
            "Allow an unsubsampled load above "
            f"{FULL_CORPUS_ROW_LIMIT:,} rows (multi-gigabyte)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: load, summarise, and optionally write the quality report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = _parse_args(argv)

    adapter = SparkovAdapter(
        verify_hashes=not args.skip_hash_check,
        allow_full_corpus=args.full_corpus,
    )
    result = adapter.load_detailed(args.root, max_entities=args.max_cards, seed=args.seed)
    report = build_report(result)

    log.info(
        "Loaded %d of %d rows | %d cards | %d merchants | fraud %d (%.3f%%)",
        report.kept_row_count,
        report.raw_row_count,
        report.customer_count,
        report.merchant_count,
        report.fraud_count,
        report.fraud_rate * 100,
    )
    for warning in report.warnings:
        log.warning("%s", warning)

    if args.report:
        written = report.write(run_dir(args.run_name))
        log.info("Quality report written to %s", written)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
