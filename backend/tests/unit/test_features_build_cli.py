"""Phase 5C — the adapter → canonical → matrix → cache seam.

Runs end to end on the Sparkov CSV fixtures, so CI exercises the whole path
without the real corpus. Evaluation deliberately sits outside this: the CLI
stops once the matrix is cached.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.fraud.feature_spec import FEATURESETS
from ml.features import build as build_cli
from ml.runs import (
    RUN_METADATA_FILENAME,
    RunMetadata,
    load_run_metadata,
    save_run_metadata,
)
from tests.unit.test_dataset_sparkov import build_fixture_corpus, fixture_adapter


@pytest.fixture
def sparkov_root(tmp_path: Path) -> Path:
    return build_fixture_corpus(tmp_path)


@pytest.fixture
def patched_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve 'sparkov' to a fixture-backed adapter with hash checks off."""
    adapter = fixture_adapter(tmp_path)
    monkeypatch.setattr(build_cli, "get_adapter", lambda name: adapter)


@pytest.mark.usefixtures("patched_adapter")
def test_cli_builds_a_cache_from_the_adapter(tmp_path: Path, sparkov_root: Path) -> None:
    cache_root = tmp_path / "cache"
    exit_code = build_cli.main(
        ["--dataset", "sparkov", "--root", str(sparkov_root), "--cache-root", str(cache_root)]
    )

    assert exit_code == 0
    caches = list(cache_root.glob("*.npz"))
    assert len(caches) == 1
    assert caches[0].name.startswith("sparkov_v1_")


@pytest.mark.usefixtures("patched_adapter")
def test_cached_matrix_has_the_frozen_feature_width(
    tmp_path: Path, sparkov_root: Path
) -> None:
    cache_root = tmp_path / "cache"
    build_cli.main(
        ["--dataset", "sparkov", "--root", str(sparkov_root), "--cache-root", str(cache_root)]
    )
    cache_file = next(iter(cache_root.glob("*.npz")))

    with np.load(cache_file, allow_pickle=False) as archive:
        assert archive["X"].shape[1] == len(FEATURESETS["v1"])
        assert archive["X"].shape[0] == archive["y"].shape[0]
        assert [str(name) for name in archive["feature_names"]] == FEATURESETS["v1"]


@pytest.mark.usefixtures("patched_adapter")
def test_second_invocation_reuses_the_cache(tmp_path: Path, sparkov_root: Path) -> None:
    cache_root = tmp_path / "cache"
    args = ["--dataset", "sparkov", "--root", str(sparkov_root), "--cache-root", str(cache_root)]

    build_cli.main(args)
    written_at = next(iter(cache_root.glob("*.npz"))).stat().st_mtime_ns
    build_cli.main(args)

    assert next(iter(cache_root.glob("*.npz"))).stat().st_mtime_ns == written_at


@pytest.mark.usefixtures("patched_adapter")
def test_subsampling_produces_a_separate_cache_entry(
    tmp_path: Path, sparkov_root: Path
) -> None:
    """A subsampled run must never answer from the full corpus's matrix."""
    cache_root = tmp_path / "cache"
    base = ["--dataset", "sparkov", "--root", str(sparkov_root), "--cache-root", str(cache_root)]

    build_cli.main(base)
    build_cli.main([*base, "--max-cards", "1"])

    assert len(list(cache_root.glob("*.npz"))) == 2


@pytest.mark.usefixtures("patched_adapter")
def test_run_record_captures_the_inputs(
    tmp_path: Path, sparkov_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "cache"
    runs_root = tmp_path / "runs"

    def write_to_tmp(record: RunMetadata) -> Path:
        return save_run_metadata(record, runs_root=runs_root)

    monkeypatch.setattr(build_cli, "save_run_metadata", write_to_tmp)
    build_cli.main(
        [
            "--dataset",
            "sparkov",
            "--root",
            str(sparkov_root),
            "--cache-root",
            str(cache_root),
            "--seed",
            "11",
            "--max-cards",
            "2",
            "--run-name",
            "sparkov-fixture",
        ]
    )

    record = load_run_metadata("sparkov-fixture", runs_root=runs_root)
    assert record.featureset_version == "v1"
    assert record.seed == 11
    assert record.dataset.name == "sparkov"
    assert record.dataset.origin.value == "synthetic"
    assert record.dataset.subsample is not None
    assert record.dataset.subsample.max_entities == 2
    assert record.splits == ()  # fold choice belongs to evaluation, not here

    payload = json.loads(
        (runs_root / "sparkov-fixture" / RUN_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["dataset"]["license"].startswith("CC0")


def test_unknown_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown dataset 'ghost'"):
        build_cli.main(["--dataset", "ghost"])
