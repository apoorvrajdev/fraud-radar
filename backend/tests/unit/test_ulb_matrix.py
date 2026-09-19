"""Phase 5E — the `ulb_pca_v1` matrix, on fixture files.

Three promises are held here. The featureset is exactly the published columns
it is defined as, and stays out of the production registry. The matrix keeps
the load order the chronological split relies on, and no row's features
depend on any other row. And a run recorded on it can be neither promoted nor
analysed as a v1 run (decisions 4 and 12): those guards are exercised on a
real run directory recorded as a ULB run would be.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.fraud.explainer import load_explainer
from app.fraud.feature_spec import FEATURESETS, feature_names
from ml.datasets.manifest import sha256_file
from ml.loading import DataRequest, load_run_data
from ml.paths import FEATURE_CACHE_DIR
from ml.promote import PromotionError, promote_run
from ml.run_analysis import analyze_run
from ml.run_verification import RunVerificationError, check_recorded_run, read_recorded_run
from ml.runs import FoldIdentity, identify_test_fold, split_periods
from ml.splits import assert_no_temporal_leakage, chronological_split
from ml.tracks.ulb.load import ELAPSED_TIME_ORIGIN, SOURCE_FILENAME, UlbSource
from ml.tracks.ulb.matrix import (
    MATRIX_PREPROCESSING,
    ULB_PCA_V1,
    ULB_PCA_V1_FEATURES,
    UlbMatrix,
    build_matrix,
)
from ml.tracks.ulb.quality import build_quality_report
from tests.unit.test_run_verification import RUN, train_fixture_run
from tests.unit.test_ulb_load import load_fixture, ulb_row

ULB_RUN = "ulb_pca_v1_seed42"


def _corpus(
    n_rows: int = 20,
    *,
    fraud_times: Sequence[int] = (2, 15, 18),
    amount: Callable[[int], str] = lambda index: f"{index + 1}.25",
) -> list[list[str]]:
    """`n_rows` distinct rows at Times 0..n-1: train holds 0-13, val 14-16, test 17-19."""
    return [
        ulb_row(
            str(index),
            marker=index,
            amount=amount(index),
            label="1" if index in fraud_times else "0",
        )
        for index in range(n_rows)
    ]


def _matrix(tmp_path: Path, rows: Sequence[Sequence[str]]) -> tuple[UlbSource, UlbMatrix]:
    source = load_fixture(tmp_path, rows)
    return source, build_matrix(source)


# ---------------------------------------------------------------------------
# The featureset
# ---------------------------------------------------------------------------


def test_ulb_pca_v1_is_the_published_components_and_amount_in_source_order() -> None:
    """Pinned literally, as Phase 5E decision 4 defines it."""
    assert ULB_PCA_V1 == "ulb_pca_v1"
    assert ULB_PCA_V1_FEATURES == (
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount",
    )  # fmt: skip


def test_time_and_class_are_not_features() -> None:
    assert "Time" not in ULB_PCA_V1_FEATURES
    assert "Class" not in ULB_PCA_V1_FEATURES


def test_ulb_pca_v1_stays_out_of_the_production_registry() -> None:
    assert ULB_PCA_V1 not in FEATURESETS
    with pytest.raises(ValueError, match="Unknown featureset version 'ulb_pca_v1'"):
        feature_names(ULB_PCA_V1)
    # v1 is untouched, and no ULB column can be joined to it by name.
    assert len(FEATURESETS["v1"]) == 17
    assert set(ULB_PCA_V1_FEATURES).isdisjoint(FEATURESETS["v1"])


# ---------------------------------------------------------------------------
# Shape, order and values
# ---------------------------------------------------------------------------


def test_matrix_shape_and_types(tmp_path: Path) -> None:
    _, matrix = _matrix(tmp_path, _corpus())
    ds = matrix.ds

    assert ds.X.shape == (20, 29)
    assert ds.X.dtype == np.float64
    assert ds.X.flags.c_contiguous
    assert ds.y.shape == (20,)
    assert ds.y.dtype == np.int64
    assert len(ds.timestamps) == len(ds.transaction_ids) == ds.n_rows == 20
    assert all(moment.tzinfo is not None for moment in ds.timestamps)
    assert ds.feature_names == list(ULB_PCA_V1_FEATURES)


def test_each_column_holds_its_named_source_column_as_published(tmp_path: Path) -> None:
    source, matrix = _matrix(tmp_path, _corpus())
    X = matrix.ds.X

    for column, name in enumerate(ULB_PCA_V1_FEATURES[:-1]):
        assert X[:, column].tolist() == source.components[:, column].tolist(), name
    assert X[:, -1].tolist() == source.amounts.tolist()
    # Row 3 carries marker 3: V1 is 3.00, V28 is 3.27, Amount is 4.25.
    assert (X[3, 0], X[3, 27], X[3, 28]) == (3.0, 3.27, 4.25)
    assert matrix.ds.y.tolist() == source.labels.tolist()


def test_amount_is_used_as_published_not_log_transformed(tmp_path: Path) -> None:
    amounts = ["149.62", "0.00", "2.69"]
    _, matrix = _matrix(
        tmp_path, [ulb_row(str(i), marker=i, amount=value) for i, value in enumerate(amounts)]
    )

    assert matrix.ds.X[:, -1].tolist() == [149.62, 0.0, 2.69]
    assert matrix.ds.X[0, -1] != np.log1p(149.62)


def test_rows_follow_load_order_time_then_file_position(tmp_path: Path) -> None:
    times = ["5", "3", "5", "1e+05", "1", "3"]
    _, matrix = _matrix(tmp_path, [ulb_row(time, marker=i) for i, time in enumerate(times)])
    ds = matrix.ds

    assert ds.transaction_ids == ["5", "2", "6", "1", "3", "4"]
    # Each row's values travel with it: V1 is the row's marker.
    assert ds.X[:, 0].tolist() == [4.0, 1.0, 5.0, 0.0, 2.0, 3.0]
    seconds = [(moment - ELAPSED_TIME_ORIGIN).total_seconds() for moment in ds.timestamps]
    assert seconds == [1.0, 3.0, 3.0, 5.0, 5.0, 100000.0]


def test_exact_duplicates_stay_as_identical_rows_with_their_own_ids(tmp_path: Path) -> None:
    same = ulb_row("7", marker=7, label="1")
    _, matrix = _matrix(tmp_path, [ulb_row("0", marker=0), same, same])

    assert matrix.ds.n_rows == 3
    assert matrix.ds.X[1].tolist() == matrix.ds.X[2].tolist()
    assert matrix.ds.y.tolist() == [0, 1, 1]
    assert matrix.ds.transaction_ids == ["1", "2", "3"]


def test_excluded_rows_are_absent_and_ids_stay_file_positions(tmp_path: Path) -> None:
    rows = [ulb_row("0", marker=0), ulb_row("1", marker=1, amount="-5.00"), ulb_row("2", marker=2)]
    _, matrix = _matrix(tmp_path, rows)

    assert matrix.ds.transaction_ids == ["1", "3"]
    assert matrix.ds.X[:, 0].tolist() == [0.0, 2.0]


# ---------------------------------------------------------------------------
# Determinism and provenance
# ---------------------------------------------------------------------------


def test_the_same_file_always_gives_the_same_matrix(tmp_path: Path) -> None:
    rows = _corpus()
    _, first = _matrix(tmp_path / "first", rows)
    _, second = _matrix(tmp_path / "second", rows)

    assert first.ds.X.tobytes() == second.ds.X.tobytes()
    assert first.ds.y.tobytes() == second.ds.y.tobytes()
    assert first.ds.transaction_ids == second.ds.transaction_ids
    assert first.ds.timestamps.tolist() == second.ds.timestamps.tolist()
    assert first.ds.feature_names == second.ds.feature_names
    assert first.provenance.preprocessing == second.provenance.preprocessing
    assert dict(first.provenance.files) == dict(second.provenance.files)


def test_provenance_adds_the_matrix_step_to_the_loaded_rows_provenance(tmp_path: Path) -> None:
    source, matrix = _matrix(tmp_path, _corpus())

    assert matrix.provenance.preprocessing == (
        *source.provenance.preprocessing,
        MATRIX_PREPROCESSING,
    )
    assert "featureset ulb_pca_v1" in MATRIX_PREPROCESSING
    assert "neither is a feature" in MATRIX_PREPROCESSING
    assert "no transform" in MATRIX_PREPROCESSING
    # Everything else is the loaded rows' record, unchanged: the pinned bytes and the licence.
    assert dict(matrix.provenance.files) == {
        SOURCE_FILENAME: sha256_file(tmp_path / "raw" / "ulb" / SOURCE_FILENAME)
    }
    assert matrix.provenance.license == source.provenance.license
    assert matrix.provenance.row_count == matrix.ds.n_rows
    assert matrix.provenance.fraud_count == int(matrix.ds.y.sum())
    assert MATRIX_PREPROCESSING not in source.provenance.preprocessing


def test_building_the_matrix_writes_nothing(tmp_path: Path) -> None:
    """The matrix exists in memory only: no cache entry, no run file, no copy of a row."""

    def files(root: Path) -> set[Path]:
        return {path for path in root.rglob("*") if path.is_file()} if root.exists() else set()

    source = load_fixture(tmp_path, _corpus())
    before = files(tmp_path), files(FEATURE_CACHE_DIR)
    build_matrix(source)

    assert (files(tmp_path), files(FEATURE_CACHE_DIR)) == before


# ---------------------------------------------------------------------------
# The chronological split, and no information from later rows
# ---------------------------------------------------------------------------


def test_the_split_of_the_matrix_is_the_one_the_quality_report_describes(tmp_path: Path) -> None:
    source, matrix = _matrix(tmp_path, _corpus())
    ds = matrix.ds
    splits = chronological_split(ds.timestamps)
    assert_no_temporal_leakage(ds.timestamps, splits)
    report = build_quality_report(source)

    assert [(fold.rows, fold.frauds) for fold in report.split.folds] == [
        (len(indices), int(ds.y[indices].sum()))
        for indices in (splits.train, splits.val, splits.test)
    ]
    # Rows are already in time order, so each fold is a contiguous run of rows.
    assert splits.train.tolist() == list(range(14))
    assert splits.val.tolist() == list(range(14, 17))
    assert splits.test.tolist() == list(range(17, 20))
    assert identify_test_fold(ds.transaction_ids, splits) == FoldIdentity.of(
        "test", ["18", "19", "20"]
    )
    periods = split_periods(ds.timestamps, splits)
    assert [period.start for period in periods] == [
        ELAPSED_TIME_ORIGIN + timedelta(seconds=seconds) for seconds in (0, 14, 17)
    ]


def test_folds_are_disjoint_ordered_and_cover_every_row(tmp_path: Path) -> None:
    rows = _corpus()
    rows[14] = ulb_row("13", marker=14)  # a Time shared across the train/val boundary
    _, matrix = _matrix(tmp_path, rows)
    ds = matrix.ds
    splits = chronological_split(ds.timestamps)
    folds = [set(splits.train.tolist()), set(splits.val.tolist()), set(splits.test.tolist())]

    assert folds[0].isdisjoint(folds[1]) and folds[1].isdisjoint(folds[2])
    assert folds[0].isdisjoint(folds[2])
    assert set().union(*folds) == set(range(ds.n_rows))
    assert max(ds.timestamps[splits.train]) <= min(ds.timestamps[splits.val])
    assert max(ds.timestamps[splits.val]) <= min(ds.timestamps[splits.test])
    # The tie falls on both sides, in file order, as recorded rather than corrected.
    assert ds.transaction_ids[13:15] == ["14", "15"]
    assert ds.timestamps[13] == ds.timestamps[14]


def test_changing_later_rows_changes_no_earlier_features(tmp_path: Path) -> None:
    """Every feature is the row's own value: the test fold cannot reach train or val."""
    original = _corpus()
    altered = _corpus(fraud_times=(2, 15, 17, 18, 19), amount=lambda index: f"{900 + index}.99")
    for index in range(17):
        altered[index] = original[index]
    for index in range(17, 20):
        altered[index] = [altered[index][0], *[f"-{value}" for value in altered[index][1:29]],
                          *altered[index][29:]]  # fmt: skip
    _, before = _matrix(tmp_path / "original", original)
    _, after = _matrix(tmp_path / "altered", altered)

    assert before.ds.X[:17].tobytes() == after.ds.X[:17].tobytes()
    assert before.ds.y[:17].tolist() == after.ds.y[:17].tolist()
    assert before.ds.transaction_ids[:17] == after.ds.transaction_ids[:17]
    assert before.ds.X[17:].tobytes() != after.ds.X[17:].tobytes()
    assert chronological_split(before.ds.timestamps).sizes == (
        chronological_split(after.ds.timestamps).sizes
    )


def test_the_first_rows_of_a_file_give_the_first_rows_of_its_matrix(tmp_path: Path) -> None:
    rows = _corpus(40, fraud_times=(2, 30, 36))
    _, whole = _matrix(tmp_path / "whole", rows)
    _, prefix = _matrix(tmp_path / "prefix", rows[:25])

    assert prefix.ds.X.tobytes() == whole.ds.X[:25].tobytes()
    assert prefix.ds.y.tolist() == whole.ds.y[:25].tolist()
    assert prefix.ds.transaction_ids == whole.ds.transaction_ids[:25]


# ---------------------------------------------------------------------------
# A ULB run can be neither promoted nor analysed as v1 (decisions 4 and 12)
# ---------------------------------------------------------------------------


def _edit(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def trained_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_fixture_run(tmp_path_factory.mktemp("ulb-guards")).runs_root


@pytest.fixture
def ulb_runs(trained_runs: Path, tmp_path: Path) -> Path:
    """A complete run directory, recorded as a ULB run is: name, featureset, feature list, data.

    The model inside is the pipeline fixture's; the guards read the record,
    and refuse before any model is loaded or scored.
    """
    runs_root = tmp_path / "runs"
    shutil.copytree(trained_runs / RUN, runs_root / ULB_RUN)
    _, matrix = _matrix(tmp_path / "data", _corpus())
    _edit(
        runs_root / ULB_RUN / "run.json",
        lambda record: record.update(
            run_name=ULB_RUN,
            featureset_version=ULB_PCA_V1,
            dataset=matrix.provenance.to_dict(),
        ),
    )
    _edit(
        runs_root / ULB_RUN / "feature_list.json",
        lambda record: record.update(features=list(ULB_PCA_V1_FEATURES)),
    )
    return runs_root


def _relabel_as_v1(runs_root: Path) -> None:
    _edit(runs_root / ULB_RUN / "run.json", lambda record: record.update(featureset_version="v1"))


def test_promotion_refuses_a_ulb_run(ulb_runs: Path, tmp_path: Path) -> None:
    served = tmp_path / "served"
    served.mkdir()
    (served / "model.json").write_text('{"previously": "served"}', encoding="utf-8")
    before = _snapshot(served)

    with pytest.raises(PromotionError, match="featureset 'ulb_pca_v1', which is not registered"):
        promote_run(ULB_RUN, runs_root=ulb_runs, artifacts_dir=served)
    assert _snapshot(served) == before


def test_promotion_refuses_a_ulb_run_relabelled_as_v1(ulb_runs: Path, tmp_path: Path) -> None:
    _relabel_as_v1(ulb_runs)
    served = tmp_path / "served"
    served.mkdir()

    with pytest.raises(PromotionError, match="does not list featureset 'v1'"):
        promote_run(ULB_RUN, runs_root=ulb_runs, artifacts_dir=served)
    assert not any(served.iterdir())


def test_analysis_refuses_a_ulb_run_before_loading_any_data(ulb_runs: Path) -> None:
    before = _snapshot(ulb_runs)

    with pytest.raises(
        RunVerificationError, match="featureset 'ulb_pca_v1', which is not registered"
    ):
        analyze_run(ULB_RUN, runs_root=ulb_runs)
    assert _snapshot(ulb_runs) == before


def test_verification_refuses_a_ulb_run_relabelled_as_v1(ulb_runs: Path) -> None:
    _relabel_as_v1(ulb_runs)

    with pytest.raises(RunVerificationError, match="does not list featureset 'v1'"):
        check_recorded_run(read_recorded_run(ULB_RUN, runs_root=ulb_runs))


def test_the_serving_explainer_refuses_a_ulb_run(ulb_runs: Path) -> None:
    with pytest.raises(ValueError, match=r"feature_list\.json has 29 features"):
        load_explainer(ulb_runs / ULB_RUN)


def test_the_registered_dataset_loader_refuses_ulb() -> None:
    """ULB has no adapter, so the v1 loading path cannot build a matrix from it."""
    with pytest.raises(ValueError, match="Unknown dataset 'ulb'"):
        load_run_data(DataRequest(dataset="ulb", featureset=ULB_PCA_V1))
