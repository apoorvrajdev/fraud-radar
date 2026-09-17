"""Phase 5D — promotion copies a run's model artifacts only behind its guard.

Every test works on a fixture run and a temporary served directory. The real
`ml/artifacts/` is never written: the last test checks it is byte-for-byte
unchanged after the command line has promoted a run into its stand-in.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.fraud.feature_spec import FEATURESETS
from ml import promote
from ml.paths import ARTIFACTS_DIR
from ml.promote import PROMOTED_FILES, PromotionError, check_promotable, promote_run
from tests.unit.test_run_verification import RUN, train_fixture_run


def _snapshot(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class Setup:
    runs_root: Path
    served: Path

    @property
    def run(self) -> Path:
        return self.runs_root / RUN

    def edit(self, filename: str, change: Callable[[dict[str, Any]], None]) -> None:
        path = self.run / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def refused(self, match: str, run: str = RUN) -> None:
        before = _snapshot(self.served)
        with pytest.raises(PromotionError, match=match):
            promote_run(run, runs_root=self.runs_root, artifacts_dir=self.served)
        assert _snapshot(self.served) == before, "a refused promotion changed the served files"


@pytest.fixture(scope="module")
def trained_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train_fixture_run(tmp_path_factory.mktemp("promote")).runs_root


@pytest.fixture
def setup(trained_runs: Path, tmp_path: Path) -> Setup:
    runs_root = tmp_path / "runs"
    shutil.copytree(trained_runs, runs_root)
    served = tmp_path / "served"
    served.mkdir()
    (served / "model.json").write_text('{"previously": "served"}', encoding="utf-8")
    (served / "calibration_metrics.json").write_text('{"left": "alone"}', encoding="utf-8")
    # Files a run directory may hold beside its model artifacts.
    for extra in ("quality_report.json", "transfer_metrics.json", "rules_audit.json"):
        (runs_root / RUN / extra).write_text("{}", encoding="utf-8")
    return Setup(runs_root=runs_root, served=served)


# ---------------------------------------------------------------------------
# What is copied
# ---------------------------------------------------------------------------


def test_exactly_the_five_model_artifacts_are_copied_byte_for_byte(setup: Setup) -> None:
    promotion = promote_run(RUN, runs_root=setup.runs_root, artifacts_dir=setup.served)

    assert PROMOTED_FILES == (
        "model.json",
        "feature_list.json",
        "threshold.json",
        "metrics.json",
        "training_metadata.json",
    )
    assert [path.name for path in promotion.copied] == list(PROMOTED_FILES)
    for name in PROMOTED_FILES:
        assert (setup.served / name).read_bytes() == (setup.run / name).read_bytes(), name
    assert sorted(path.name for path in setup.served.iterdir()) == sorted(
        [*PROMOTED_FILES, "calibration_metrics.json"]
    )


def test_nothing_else_in_the_served_directory_is_touched(setup: Setup) -> None:
    promote_run(RUN, runs_root=setup.runs_root, artifacts_dir=setup.served)

    assert (setup.served / "calibration_metrics.json").read_text("utf-8") == '{"left": "alone"}'
    for name in ("run.json", "quality_report.json", "transfer_metrics.json", "rules_audit.json"):
        assert not (setup.served / name).exists(), name
    assert not list(setup.served.glob("*.promoting"))


def test_the_run_itself_is_left_unchanged(setup: Setup) -> None:
    before = _snapshot(setup.run)

    promote_run(RUN, runs_root=setup.runs_root, artifacts_dir=setup.served)

    assert _snapshot(setup.run) == before


def test_the_promotion_names_the_run_and_its_featureset(setup: Setup) -> None:
    promotion = promote_run(RUN, runs_root=setup.runs_root, artifacts_dir=setup.served)

    assert (promotion.run_name, promotion.featureset_version) == (RUN, "v1")
    assert (promotion.source, promotion.destination) == (setup.run, setup.served)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_a_featureset_absent_from_the_registry_is_refused(setup: Setup) -> None:
    """Read from run.json, even though feature_list.json still lists the v1 columns."""
    setup.edit("run.json", lambda record: record.update(featureset_version="v9"))

    assert json.loads((setup.run / "feature_list.json").read_text("utf-8"))["features"] == (
        FEATURESETS["v1"]
    )
    setup.refused("featureset 'v9', which is not registered")


def test_a_run_on_a_registered_featureset_passes_the_guard(setup: Setup) -> None:
    record = check_promotable(RUN, runs_root=setup.runs_root)

    assert record.featureset_version in FEATURESETS


def test_a_run_without_a_run_record_is_refused(setup: Setup) -> None:
    (setup.run / "run.json").unlink()

    setup.refused("has no run.json")


def test_a_missing_run_is_refused(setup: Setup) -> None:
    setup.refused("No run named 'no-such-run'", run="no-such-run")


@pytest.mark.parametrize("name", PROMOTED_FILES)
def test_a_run_missing_any_promoted_file_is_refused(setup: Setup, name: str) -> None:
    (setup.run / name).unlink()

    setup.refused(f"is missing {name}")


def test_a_feature_list_out_of_registered_order_is_refused(setup: Setup) -> None:
    setup.edit("feature_list.json", lambda payload: payload["features"].reverse())

    setup.refused("registered order")


def test_a_model_file_other_than_the_recorded_one_is_refused(setup: Setup) -> None:
    (setup.run / "model.json").write_text('{"another": "model"}', encoding="utf-8")

    setup.refused("it is not the model the run saved")


def test_a_run_record_naming_another_run_is_refused(setup: Setup) -> None:
    setup.edit("run.json", lambda record: record.update(run_name="another-run"))

    setup.refused("names run 'another-run'")


def test_a_malformed_run_record_is_refused(setup: Setup) -> None:
    (setup.run / "run.json").write_text('{"run_name": "synthetic-fixture"}', encoding="utf-8")

    setup.refused("malformed run.json")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_cli_promotes_into_the_served_directory(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(promote, "RUNS_ROOT", setup.runs_root)
    monkeypatch.setattr(promote, "ARTIFACTS_DIR", setup.served)

    promote.main([RUN])

    assert (setup.served / "model.json").read_bytes() == (setup.run / "model.json").read_bytes()


def test_the_cli_exits_with_an_error_when_the_guard_refuses(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(promote, "RUNS_ROOT", setup.runs_root)
    monkeypatch.setattr(promote, "ARTIFACTS_DIR", setup.served)
    setup.edit("run.json", lambda record: record.update(featureset_version="ulb"))
    before = _snapshot(setup.served)

    with pytest.raises(SystemExit) as refused:
        promote.main([RUN])

    assert refused.value.code == 1
    assert "was not promoted" in caplog.text
    assert _snapshot(setup.served) == before


@pytest.mark.parametrize("argv", [["../escape"], []], ids=["unsafe-name", "no-run"])
def test_the_cli_refuses_arguments_before_any_work(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(promote, "promote_run", lambda *a, **k: calls.append(a))

    with pytest.raises(SystemExit) as refused:
        promote.main(argv)

    assert refused.value.code == 2
    assert calls == []


def test_the_real_served_artifacts_are_never_written_by_these_tests(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _snapshot(ARTIFACTS_DIR)
    monkeypatch.setattr(promote, "RUNS_ROOT", setup.runs_root)
    monkeypatch.setattr(promote, "ARTIFACTS_DIR", setup.served)

    promote.main([RUN])

    assert _snapshot(ARTIFACTS_DIR) == before
