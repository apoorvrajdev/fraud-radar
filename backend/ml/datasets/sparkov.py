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


class SparkovSchemaError(DatasetContractError):
    """The source file does not look like the Sparkov dataset we expect."""


@dataclass(frozen=True)
class SparkovLoadResult:
    """A canonical dataset plus the accounting behind it."""

    dataset: CanonicalDataset
    raw_row_count: int
    exclusions: tuple[ExclusionRecord, ...]
    field_notes: tuple[FieldNote, ...]
    unix_time_mismatches: int
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
    ) -> None:
        self.manifest_path = manifest_path
        self.verify_hashes = verify_hashes

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

        frame, exclusions = _apply_exclusions(frame)
        frame, subsample = _subsample_by_card(frame, max_entities=max_entities, seed=seed)

        customers = _build_customers(frame["cc_num"])
        merchants, multi_category_merchants = _build_merchants(frame[["merchant", "category"]])
        transactions, labels = _build_transactions(frame)
        mismatches = _count_unix_time_mismatches(frame)

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
            unix_time_mismatches=mismatches,
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

    timestamps = pd.to_datetime(frame["trans_date_trans_time"], errors="coerce", format="mixed")
    frame, record = _exclude(frame, timestamps.isna(), "unparsable_timestamp", "trans_num")
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
                created_at=_to_utc(str(row.trans_date_trans_time)),
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


def _to_utc(raw: str) -> datetime:
    """Source timestamps are naive; the dataset documents them as a single clock."""
    parsed = datetime.fromisoformat(raw)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _count_unix_time_mismatches(frame: pd.DataFrame) -> int:
    """How often `unix_time` disagrees with the parsed timestamp.

    Reported rather than enforced: a systematic offset is a property of the
    generator worth stating, not a reason to reject the corpus.
    """
    if frame.empty:
        return 0
    parsed = pd.to_datetime(frame["trans_date_trans_time"], errors="coerce", format="mixed")
    # Cast to second resolution before taking the integer view: the underlying
    # unit of a datetime64 column is version-dependent (ns or us), and dividing
    # by a hard-coded factor silently turns every row into a mismatch.
    epoch_seconds = parsed.astype("datetime64[s]").astype("int64")
    delta = (epoch_seconds - pd.to_numeric(frame["unix_time"], errors="coerce")).abs()
    return int((delta > _UNIX_TIME_TOLERANCE_SECONDS).sum())


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
            "unix_time_mismatches": result.unix_time_mismatches,
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI: load, summarise, and optionally write the quality report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    args = _parse_args(argv)

    adapter = SparkovAdapter(verify_hashes=not args.skip_hash_check)
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
