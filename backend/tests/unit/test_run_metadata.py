"""Phase 5A — run records and the offline directory layout.

A benchmark number is only worth quoting if someone else can reconstruct the
conditions that produced it. These tests hold `run.json` to that standard:
the dataset bytes, the slice, the feature contract, the seed and the code
version survive a write/read cycle intact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ml import paths
from ml.datasets.base import DataOrigin, DatasetContractError, DatasetProvenance, Subsample
from ml.paths import InvalidRunNameError, ensure_dir, run_dir, validate_run_name
from ml.runs import (
    RUN_METADATA_FILENAME,
    RUN_METADATA_VERSION,
    TRANSACTION_IDS_DIGEST_DEFINITION,
    FoldIdentity,
    RunMetadata,
    SplitPeriod,
    current_git_commit,
    identify_test_fold,
    load_run_metadata,
    save_run_metadata,
    split_periods,
    transaction_ids_digest,
)
from ml.splits import SplitIndices, chronological_split

ANCHOR = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
DIGEST = "b" * 64
TEST_IDS = ("tx-8", "tx-9", "tx-10")


def _provenance() -> DatasetProvenance:
    return DatasetProvenance(
        name="fixture",
        version="v1",
        origin=DataOrigin.SYNTHETIC,
        source_url="https://example.test/fixture",
        citation="Fixture dataset",
        license="CC0",
        label_field="is_fraud",
        label_definition="1 = confirmed fraudulent authorisation",
        retrieved_at=ANCHOR,
        files={"fixture.csv": DIGEST},
        row_count=1000,
        fraud_count=15,
        period_start=ANCHOR - timedelta(days=90),
        period_end=ANCHOR,
        subsample=Subsample(strategy="cards", seed=7, max_entities=200, selected_entities=200),
        preprocessing=("timestamps normalised to UTC",),
    )


def _metadata(**overrides: object) -> RunMetadata:
    payload: dict[str, object] = {
        "run_name": "fixture-run",
        "dataset": _provenance(),
        "featureset_version": "v1",
        "code_version": "0" * 40,
        "library_versions": {"xgboost": "3.2.0"},
        "splits": (
            SplitPeriod("train", ANCHOR - timedelta(days=90), ANCHOR - timedelta(days=30)),
            SplitPeriod("val", ANCHOR - timedelta(days=30), ANCHOR - timedelta(days=15)),
            SplitPeriod("test", ANCHOR - timedelta(days=15), ANCHOR),
        ),
        "test_fold_identity": FoldIdentity.of("test", TEST_IDS),
    }
    payload.update(overrides)
    return RunMetadata(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_the_four_stages_have_distinct_locations() -> None:
    locations = [paths.RAW_DATA_DIR, paths.CACHE_DIR, paths.RUNS_ROOT, paths.ARTIFACTS_DIR]
    assert len(set(locations)) == len(locations)


def test_every_location_lives_under_the_ml_package() -> None:
    for location in (
        paths.DATA_DIR,
        paths.RAW_DATA_DIR,
        paths.CACHE_DIR,
        paths.FEATURE_CACHE_DIR,
        paths.ARTIFACTS_DIR,
        paths.RUNS_ROOT,
    ):
        assert paths.ML_ROOT in location.parents


def test_caches_are_separate_from_committed_outputs() -> None:
    """Deleting the cache must not touch a run's record."""
    assert paths.ARTIFACTS_DIR not in paths.CACHE_DIR.parents
    assert paths.CACHE_DIR not in paths.RUNS_ROOT.parents


def test_runs_live_under_the_artifact_directory() -> None:
    assert paths.RUNS_ROOT.parent == paths.ARTIFACTS_DIR


def test_run_dir_is_a_single_segment_under_the_runs_root(tmp_path: Path) -> None:
    assert run_dir("sparkov-v1", runs_root=tmp_path) == tmp_path / "sparkov-v1"


@pytest.mark.parametrize(
    "bad_name", ["", "..", "../escape", "has space", "UPPER", "/absolute", ".leading-dot"]
)
def test_unsafe_run_names_are_refused(bad_name: str) -> None:
    with pytest.raises(InvalidRunNameError):
        validate_run_name(bad_name)


@pytest.mark.parametrize("good_name", ["synthetic-v1", "run.2026", "ulb_baseline", "a1"])
def test_reasonable_run_names_are_accepted(good_name: str) -> None:
    assert validate_run_name(good_name) == good_name


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    assert ensure_dir(target) == target
    assert ensure_dir(target).is_dir()


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------


def test_run_metadata_round_trips_through_disk(tmp_path: Path) -> None:
    original = _metadata()
    written = save_run_metadata(original, runs_root=tmp_path)

    assert written == tmp_path / "fixture-run" / RUN_METADATA_FILENAME
    assert load_run_metadata("fixture-run", runs_root=tmp_path) == original


def test_saved_payload_records_every_reproducibility_field(tmp_path: Path) -> None:
    save_run_metadata(_metadata(), runs_root=tmp_path)
    payload = json.loads((tmp_path / "fixture-run" / RUN_METADATA_FILENAME).read_text("utf-8"))

    assert set(payload) == {
        "metadata_version",
        "run_name",
        "created_at_utc",
        "dataset",
        "featureset_version",
        "seed",
        "code_version",
        "library_versions",
        "splits",
        "test_fold_identity",
        "notes",
    }
    assert payload["metadata_version"] == RUN_METADATA_VERSION
    assert payload["seed"] == 7
    assert payload["featureset_version"] == "v1"


def test_saved_payload_carries_the_dataset_provenance(tmp_path: Path) -> None:
    """The input bytes, licence and slice travel with the run, not beside it."""
    save_run_metadata(_metadata(), runs_root=tmp_path)
    dataset = json.loads(
        (tmp_path / "fixture-run" / RUN_METADATA_FILENAME).read_text("utf-8")
    )["dataset"]

    assert dataset["files"] == {"fixture.csv": DIGEST}
    assert dataset["license"] == "CC0"
    assert dataset["origin"] == "synthetic"
    assert dataset["fraud_rate"] == pytest.approx(0.015)
    assert dataset["subsample"] == {
        "strategy": "cards",
        "seed": 7,
        "max_entities": 200,
        "selected_entities": 200,
    }


def test_seed_is_the_seed_the_subsample_was_drawn_with() -> None:
    metadata = _metadata()

    assert metadata.dataset.subsample is not None
    assert metadata.seed == metadata.dataset.subsample.seed == 7


def test_seed_is_null_when_the_dataset_has_no_subsample_record(tmp_path: Path) -> None:
    """No subsample, no subsample seed — never a stand-in such as the model random state."""
    original = _metadata(dataset=replace(_provenance(), subsample=None))
    save_run_metadata(original, runs_root=tmp_path)
    payload = json.loads((tmp_path / "fixture-run" / RUN_METADATA_FILENAME).read_text("utf-8"))

    assert original.seed is None
    assert payload["seed"] is None
    assert load_run_metadata("fixture-run", runs_root=tmp_path) == original


@pytest.mark.parametrize(
    ("subsample", "expected_seed"),
    [
        pytest.param(Subsample(strategy="cards", seed=7, max_entities=200), 7, id="subsampled"),
        pytest.param(None, None, id="no-subsample-record"),
    ],
)
def test_a_version_1_record_reads_its_seed_from_the_subsample_record(
    tmp_path: Path, subsample: Subsample | None, expected_seed: int | None
) -> None:
    """Version 1 stored the model random state as seed when there was no subsample."""
    payload = _metadata(dataset=replace(_provenance(), subsample=subsample)).to_dict()
    payload.update(metadata_version="1", seed=42)
    target = ensure_dir(tmp_path / "fixture-run") / RUN_METADATA_FILENAME
    target.write_text(json.dumps(payload), encoding="utf-8")

    record = load_run_metadata("fixture-run", runs_root=tmp_path)

    assert RUN_METADATA_VERSION == "3"
    assert record.metadata_version == "1"
    assert record.seed == expected_seed


def test_split_periods_survive_the_round_trip(tmp_path: Path) -> None:
    save_run_metadata(_metadata(), runs_root=tmp_path)
    reloaded = load_run_metadata("fixture-run", runs_root=tmp_path)

    train = reloaded.split("train")
    assert train is not None
    assert train.start == ANCHOR - timedelta(days=90)
    assert reloaded.split("nonexistent") is None


def test_missing_run_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No run\.json for run 'ghost'"):
        load_run_metadata("ghost", runs_root=tmp_path)


def test_created_at_defaults_to_now_in_utc() -> None:
    assert _metadata().created_at_utc.endswith("+00:00")


def test_run_name_is_validated_at_construction() -> None:
    with pytest.raises(InvalidRunNameError):
        _metadata(run_name="../escape")


def test_featureset_version_is_required() -> None:
    with pytest.raises(DatasetContractError, match="featureset_version"):
        _metadata(featureset_version="  ")


def test_duplicate_split_names_are_refused() -> None:
    with pytest.raises(DatasetContractError, match="Duplicate split names"):
        _metadata(
            splits=(
                SplitPeriod("train", ANCHOR - timedelta(days=2), ANCHOR - timedelta(days=1)),
                SplitPeriod("train", ANCHOR - timedelta(days=1), ANCHOR),
            )
        )


def test_inverted_split_period_is_refused() -> None:
    with pytest.raises(DatasetContractError, match="after it ends"):
        SplitPeriod("train", ANCHOR, ANCHOR - timedelta(days=1))


def test_naive_split_boundaries_are_refused() -> None:
    with pytest.raises(DatasetContractError, match="timezone-aware"):
        SplitPeriod("train", datetime(2026, 1, 1), datetime(2026, 2, 1))


def _timestamps(hours: list[int]) -> np.ndarray:
    return np.asarray([ANCHOR + timedelta(hours=hour) for hour in hours], dtype=object)


def test_split_periods_cover_what_each_fold_holds_in_time_order() -> None:
    """Twenty hourly rows split 14 / 3 / 3, deliberately stored out of time order."""
    hours = [19, 3, 11, 0, 7, 15, 2, 18, 9, 13, 5, 16, 1, 10, 17, 6, 12, 4, 14, 8]
    timestamps = _timestamps(hours)

    periods = split_periods(timestamps, chronological_split(timestamps))

    assert [period.name for period in periods] == ["train", "val", "test"]
    assert [(period.start, period.end) for period in periods] == [
        (ANCHOR, ANCHOR + timedelta(hours=13)),
        (ANCHOR + timedelta(hours=14), ANCHOR + timedelta(hours=16)),
        (ANCHOR + timedelta(hours=17), ANCHOR + timedelta(hours=19)),
    ]


def test_split_periods_round_trip_through_the_run_record(tmp_path: Path) -> None:
    timestamps = _timestamps(list(range(20)))
    periods = split_periods(timestamps, chronological_split(timestamps))

    save_run_metadata(_metadata(splits=periods), runs_root=tmp_path)

    assert load_run_metadata("fixture-run", runs_root=tmp_path).splits == periods


def test_split_periods_allow_folds_to_share_a_boundary_instant() -> None:
    """Transactions at the same instant can fall on both sides of a split."""
    timestamps = _timestamps([0, 1, 1, 2])
    splits = SplitIndices(train=np.array([0, 1]), val=np.array([2]), test=np.array([3]))

    periods = split_periods(timestamps, splits)

    assert periods[0].end == periods[1].start == ANCHOR + timedelta(hours=1)


def test_split_periods_refuse_naive_timestamps() -> None:
    timestamps = np.asarray([datetime(2026, 1, 1, hour) for hour in range(10)], dtype=object)

    with pytest.raises(DatasetContractError, match="timezone-aware"):
        split_periods(timestamps, chronological_split(timestamps))


def test_split_periods_refuse_an_empty_fold() -> None:
    timestamps = _timestamps([0, 1, 2])
    splits = SplitIndices(train=np.array([0, 1]), val=np.array([], dtype=int), test=np.array([2]))

    with pytest.raises(DatasetContractError, match="'val' is empty"):
        split_periods(timestamps, splits)


def test_split_periods_refuse_folds_out_of_time_order() -> None:
    timestamps = _timestamps([0, 1, 2, 3])
    splits = SplitIndices(train=np.array([2, 3]), val=np.array([0]), test=np.array([1]))

    with pytest.raises(DatasetContractError, match="not in time order"):
        split_periods(timestamps, splits)


def test_library_versions_cannot_be_mutated_after_construction() -> None:
    versions = {"xgboost": "3.2.0"}
    metadata = _metadata(library_versions=versions)
    versions["injected"] = "9.9.9"
    assert dict(metadata.library_versions) == {"xgboost": "3.2.0"}


def test_code_version_is_optional(tmp_path: Path) -> None:
    """A run outside a git checkout still records everything else."""
    save_run_metadata(_metadata(code_version=None), runs_root=tmp_path)
    assert load_run_metadata("fixture-run", runs_root=tmp_path).code_version is None


def test_current_git_commit_returns_a_sha_or_none() -> None:
    commit = current_git_commit()
    assert commit is None or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit))


# ---------------------------------------------------------------------------
# Test fold identity (version 3)
# ---------------------------------------------------------------------------


def test_the_ids_digest_is_the_sha256_of_the_documented_encoding() -> None:
    """Recomputable from the stated definition alone, without this code."""
    ids = ["tx-1", "tx-2", "tx-3"]

    expected = hashlib.sha256(b'["tx-1","tx-2","tx-3"]').hexdigest()

    assert transaction_ids_digest(ids) == expected
    assert "JSON array" in TRANSACTION_IDS_DIGEST_DEFINITION
    assert "in fold order" in TRANSACTION_IDS_DIGEST_DEFINITION


def test_the_ids_digest_is_deterministic() -> None:
    ids = [f"tx-{index}" for index in range(50)]

    assert transaction_ids_digest(ids) == transaction_ids_digest(list(ids))
    assert transaction_ids_digest(ids) == transaction_ids_digest(tuple(ids))
    assert FoldIdentity.of("test", ids) == FoldIdentity.of("test", list(ids))


def test_a_changed_id_changes_the_digest() -> None:
    ids = ["tx-1", "tx-2", "tx-3"]

    assert transaction_ids_digest(ids) != transaction_ids_digest(["tx-1", "tx-2", "tx-4"])


def test_a_changed_order_changes_the_digest() -> None:
    ids = ["tx-1", "tx-2", "tx-3"]

    assert transaction_ids_digest(ids) != transaction_ids_digest(["tx-1", "tx-3", "tx-2"])


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(["a,b"], ["a", "b"], id="comma-inside-an-id"),
        pytest.param(["a\nb"], ["a", "b"], id="newline-inside-an-id"),
        pytest.param(["ab"], ["a", "b"], id="concatenation"),
    ],
)
def test_different_id_sequences_never_encode_alike(first: list[str], second: list[str]) -> None:
    assert transaction_ids_digest(first) != transaction_ids_digest(second)


def test_a_fold_identity_counts_the_ids_it_digests() -> None:
    identity = FoldIdentity.of("test", TEST_IDS)

    assert identity.fold == "test"
    assert identity.transaction_count == len(TEST_IDS)
    assert identity.transaction_ids_sha256 == transaction_ids_digest(TEST_IDS)


def test_the_test_fold_is_identified_in_fold_order() -> None:
    """Rows stored out of time order are identified in the order the fold holds them."""
    hours = [19, 3, 11, 0, 7, 15, 2, 18, 9, 13, 5, 16, 1, 10, 17, 6, 12, 4, 14, 8]
    timestamps = _timestamps(hours)
    ids = [f"tx-at-{hour}" for hour in hours]
    splits = chronological_split(timestamps)

    identity = identify_test_fold(ids, splits)

    assert identity == FoldIdentity.of("test", ["tx-at-17", "tx-at-18", "tx-at-19"])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("fold", "", "must be non-empty", id="no-fold"),
        pytest.param("transaction_count", 0, "at least one transaction", id="empty"),
        pytest.param("transaction_ids_sha256", "abc", "64-character hex", id="short-digest"),
        pytest.param("transaction_ids_sha256", "B" * 64, "64-character hex", id="uppercase"),
    ],
)
def test_a_malformed_fold_identity_is_refused(field: str, value: object, match: str) -> None:
    with pytest.raises(DatasetContractError, match=match):
        replace(FoldIdentity.of("test", TEST_IDS), **{field: value})


def test_a_version_3_record_round_trips_its_test_fold_identity(tmp_path: Path) -> None:
    original = _metadata()
    save_run_metadata(original, runs_root=tmp_path)
    payload = json.loads((tmp_path / "fixture-run" / RUN_METADATA_FILENAME).read_text("utf-8"))

    assert payload["metadata_version"] == "3"
    assert payload["test_fold_identity"] == {
        "fold": "test",
        "transaction_count": 3,
        "transaction_ids_sha256": transaction_ids_digest(TEST_IDS),
        "digest_definition": TRANSACTION_IDS_DIGEST_DEFINITION,
    }
    reloaded = load_run_metadata("fixture-run", runs_root=tmp_path)
    assert reloaded.test_fold_identity == FoldIdentity.of("test", TEST_IDS)
    assert reloaded == original


def test_a_version_3_record_with_a_test_fold_must_identify_it() -> None:
    with pytest.raises(DatasetContractError, match="must identify the fold's transactions"):
        _metadata(test_fold_identity=None)


def test_a_version_3_record_on_disk_without_the_identity_is_refused(tmp_path: Path) -> None:
    payload = _metadata().to_dict()
    payload.pop("test_fold_identity")
    target = ensure_dir(tmp_path / "fixture-run") / RUN_METADATA_FILENAME
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetContractError, match="must identify the fold's transactions"):
        load_run_metadata("fixture-run", runs_root=tmp_path)


def test_a_record_without_folds_needs_no_identity() -> None:
    """A feature-build record chooses no folds, so there is no test fold to identify."""
    record = _metadata(splits=(), test_fold_identity=None)

    assert record.test_fold_identity is None
    assert record.to_dict()["test_fold_identity"] is None


@pytest.mark.parametrize("version", ["1", "2"])
def test_an_earlier_record_reads_back_without_a_test_fold_identity(
    tmp_path: Path, version: str
) -> None:
    """Older records are still read, and say plainly that they identify no test fold."""
    payload = _metadata().to_dict()
    payload.pop("test_fold_identity")
    payload["metadata_version"] = version
    target = ensure_dir(tmp_path / "fixture-run") / RUN_METADATA_FILENAME
    target.write_text(json.dumps(payload), encoding="utf-8")

    record = load_run_metadata("fixture-run", runs_root=tmp_path)

    assert record.metadata_version == version
    assert record.test_fold_identity is None
    assert record.split("test") is not None


def test_an_identity_for_another_fold_is_refused() -> None:
    with pytest.raises(DatasetContractError, match="describes fold 'val'"):
        _metadata(test_fold_identity=FoldIdentity.of("val", TEST_IDS))


def test_an_identity_digested_some_other_way_is_refused() -> None:
    payload = FoldIdentity.of("test", TEST_IDS).to_dict()
    payload["digest_definition"] = "SHA-256 of the ids joined by newlines"

    with pytest.raises(DatasetContractError, match="cannot be checked"):
        FoldIdentity.from_dict(payload)
