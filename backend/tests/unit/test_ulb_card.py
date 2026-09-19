"""Phase 5E — the generated ULB benchmark card.

The card is built from the three committed ULB run records, which are
aggregate-only, so these tests run on copies of the real records rather than
hand-written ones. Each refusal test changes one record and requires the card
to refuse it and say why.

The licence tests are the reason this suite exists as much as the layout: the
ODbL notice and the method offer must be on every card generated, word for
word, and cannot drop out without a test failing.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ml.benchmark_card import BenchmarkCardError
from ml.paths import ML_ROOT, RUNS_ROOT
from ml.tracks.ulb import card
from ml.tracks.ulb.load import LICENCE_NOTICE, METHOD_OFFER
from ml.tracks.ulb.train import ULB_SEEDS, run_name_for

ULB_RUNS = [run_name_for(seed) for seed in ULB_SEEDS]
PRIMARY, SECOND, THIRD = ULB_RUNS
DATA_LICENSES = ML_ROOT.parents[1] / "docs" / "DATA_LICENSES.md"


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """A copy of the committed ULB records, safe to edit."""
    root = tmp_path / "runs"
    for name in ULB_RUNS:
        (root / name).mkdir(parents=True)
        for filename in card.RUN_RECORD_FILES:
            shutil.copyfile(RUNS_ROOT / name / filename, root / name / filename)
    return root


def edit(root: Path, source: str, change: Callable[[dict[str, Any]], None]) -> None:
    path = root / source
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build(root: Path) -> str:
    return card.build_ulb_card(card.load_ulb_records(root))


def refused(root: Path, match: str) -> None:
    with pytest.raises(BenchmarkCardError, match=match):
        card.load_ulb_records(root)


def section(text: str, heading: str) -> str:
    """The text of one numbered section, up to the next heading."""
    start = text.index(heading)
    following = text.find("\n## ", start + len(heading))
    return text[start:] if following == -1 else text[start:following]


def result_rows(text: str) -> list[str]:
    results = section(text, "## 2. Results")
    return [line for line in results.splitlines() if line.startswith("| `ulb_pca_v1_seed")]


# ---------------------------------------------------------------------------
# The committed card
# ---------------------------------------------------------------------------


def test_the_committed_card_is_what_the_committed_records_produce() -> None:
    """Generated, never hand-edited: read as text, so LF and CRLF checkouts both hold."""
    committed = card.ULB_CARD_PATH.read_text(encoding="utf-8")

    assert committed == card.build_ulb_card(card.load_ulb_records(RUNS_ROOT))


def test_the_same_records_always_give_the_same_card(runs_root: Path) -> None:
    assert build(runs_root) == build(runs_root)


# ---------------------------------------------------------------------------
# Licence: the notice and the method offer cannot drop out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["generated", "committed"])
def test_the_card_carries_the_odbl_and_dbcl_notice_word_for_word(
    runs_root: Path, which: str
) -> None:
    text = build(runs_root) if which == "generated" else card.ULB_CARD_PATH.read_text("utf-8")

    assert f"> {card.LICENCE_NOTICE_MARKDOWN}" in text.splitlines()
    assert "[Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/)" in text
    assert "[Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/)" in text


def test_the_card_notice_is_the_one_data_licenses_and_the_records_carry() -> None:
    documented = [
        line[2:]
        for line in DATA_LICENSES.read_text(encoding="utf-8").splitlines()
        if line.startswith("> Contains information from the")
    ]
    as_plain_text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", card.LICENCE_NOTICE_MARKDOWN)

    assert documented == [card.LICENCE_NOTICE_MARKDOWN]
    assert as_plain_text == LICENCE_NOTICE


@pytest.mark.parametrize("which", ["generated", "committed"])
def test_the_card_carries_the_method_offer_at_the_recorded_code_version(
    runs_root: Path, which: str
) -> None:
    text = build(runs_root) if which == "generated" else card.ULB_CARD_PATH.read_text("utf-8")
    offer = next(line for line in text.splitlines() if line.startswith("**Method offer.**"))
    records = card.load_ulb_records(runs_root)

    assert METHOD_OFFER in offer
    assert f"[`backend/ml/tracks/ulb/`]({card.TRACK_LINK})" in offer
    assert (ML_ROOT / card.TRACK_LINK).is_dir()
    for run in records.runs:
        assert run.record.code_version is not None
        assert f"`{run.record.code_version}`" in offer


def test_the_notice_and_the_offer_come_before_any_result(runs_root: Path) -> None:
    text = build(runs_root)
    first_result = text.index("## 2. Results")

    assert text.index(card.LICENCE_NOTICE_MARKDOWN) < first_result
    assert text.index(METHOD_OFFER) < first_result


def test_runs_recording_different_code_versions_are_each_named(runs_root: Path) -> None:
    edit(runs_root, f"{SECOND}/run.json", lambda record: record.update(code_version="f" * 40))
    offer = next(line for line in build(runs_root).splitlines() if "**Method offer.**" in line)

    assert f"`{'f' * 40}` (`{SECOND}`)" in offer
    assert f"(`{PRIMARY}`)" in offer and f"(`{THIRD}`)" in offer


def test_a_run_record_without_the_notice_is_refused(runs_root: Path) -> None:
    edit(runs_root, f"{PRIMARY}/run.json", lambda record: record.update(notes="no notice"))

    refused(runs_root, "does not carry the licence notice and method offer")


def test_a_quality_report_without_the_notice_is_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{THIRD}/ulb_quality_report.json",
        lambda report: report["licence"].update(notice="removed"),
    )

    refused(runs_root, "does not carry the licence notice")


# ---------------------------------------------------------------------------
# What the card says
# ---------------------------------------------------------------------------


def test_the_three_random_states_are_shown_apart_with_42_primary(runs_root: Path) -> None:
    text = build(runs_root)
    rows = result_rows(text)

    assert [row.split(" | ")[0] for row in rows] == [f"| `{name}`" for name in ULB_RUNS]
    assert [row.split(" | ")[2] for row in rows] == [
        "primary",
        "pre-registered repeat",
        "pre-registered repeat",
    ]
    assert "Random state 42 is the primary result" in text
    assert "never averaged" in text
    assert not re.search(r"\b(mean|average)\b", section(text, "## 2. Results").lower())


def test_every_result_shows_its_featureset_test_size_frauds_and_prevalence(
    runs_root: Path,
) -> None:
    for row in result_rows(build(runs_root)):
        cells = row.strip("|").split(" | ")
        assert cells[3].strip() == "`ulb_pca_v1`"
        assert (cells[4], cells[5], cells[6]) == ("42,722", "52", "0.0012")


def test_values_are_read_from_the_records(runs_root: Path) -> None:
    edit(runs_root, f"{SECOND}/metrics.json", lambda metrics: metrics.update(test_pr_auc=0.1234))

    assert "| 0.1234 |" in result_rows(build(runs_root))[1]


def test_the_protocol_states_the_chronological_split_and_its_shared_boundary(
    runs_root: Path,
) -> None:
    protocol = section(build(runs_root), "## 1. Protocol")

    assert "chronologically 70/15/15" in protocol
    assert "| test | 42,722 | 52 |" in protocol
    boundary = next(line for line in protocol.splitlines() if line.startswith("| val → test"))
    assert boundary == (
        "| val → test | 151,328 s (42.04 h) | 151,328 s (42.04 h) | yes | 2 / 1 | 0 |"
    )
    assert "recorded, not corrected" in protocol


def test_the_card_states_ulb_is_real_and_apart_from_the_synthetic_tracks(runs_root: Path) -> None:
    text = build(runs_root)

    assert "ULB is real card-transaction data, anonymised by its publisher" in text
    assert "neither the in-house synthetic data nor the Sparkov simulation" in text
    assert "kept outside the production featureset registry" in text


def test_the_limitations_name_the_test_frauds_and_the_split(runs_root: Path) -> None:
    limitations = section(build(runs_root), "## 9. Accepted limitations")

    assert "holding 52 frauds among 42,722 rows" in limitations
    assert "not comparable to ULB results from random splits" in limitations
    assert "**Unknown PCA fit.**" in limitations
    assert "**Reused val fold.**" in limitations


# ---------------------------------------------------------------------------
# No row of the database
# ---------------------------------------------------------------------------


def test_the_card_is_built_from_the_aggregate_records_alone(runs_root: Path) -> None:
    records = card.load_ulb_records(runs_root)

    assert [record.source for record in records.files] == [
        f"{name}/{filename}" for name in ULB_RUNS for filename in card.RUN_RECORD_FILES
    ]
    assert "model.json" not in card.RUN_RECORD_FILES


def test_the_card_module_never_reads_the_source_file_a_matrix_or_a_model() -> None:
    tree = ast.parse((ML_ROOT / "tracks" / "ulb" / "card.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert not imported & {"load_ulb", "build_matrix", "train_ulb_run", "verify_ulb_run"}
    assert "xgboost" not in {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", card.RUN_RECORD_FILES)
def test_a_missing_record_is_refused_by_name(runs_root: Path, filename: str) -> None:
    (runs_root / THIRD / filename).unlink()

    refused(runs_root, f"{THIRD}/{filename} is missing")


def test_the_record_of_another_run_is_refused(runs_root: Path) -> None:
    edit(runs_root, f"{SECOND}/run.json", lambda record: record.update(run_name=PRIMARY))

    refused(runs_root, f"is the record of run '{PRIMARY}', not '{SECOND}'")


def test_a_run_on_another_featureset_is_refused(runs_root: Path) -> None:
    edit(runs_root, f"{PRIMARY}/run.json", lambda record: record.update(featureset_version="v1"))

    refused(runs_root, "records featureset 'v1', not 'ulb_pca_v1'")


def test_a_feature_list_out_of_order_is_refused(runs_root: Path) -> None:
    def swap(record: dict[str, Any]) -> None:
        record["features"][0], record["features"][1] = record["features"][1], record["features"][0]

    edit(runs_root, f"{PRIMARY}/feature_list.json", swap)

    refused(runs_root, "features in their order")


def test_a_fit_under_another_random_state_is_refused(runs_root: Path) -> None:
    edit(runs_root, f"{THIRD}/training_metadata.json", lambda fit: fit.update(random_state=42))

    refused(runs_root, "records random state 42, but run 'ulb_pca_v1_seed44'")


def test_metrics_measured_at_another_threshold_are_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{PRIMARY}/metrics.json",
        lambda metrics: metrics["at_operating_threshold"].update(threshold=0.5),
    )

    refused(runs_root, "was measured at threshold 0.5")


def test_a_recall_not_read_on_the_test_roc_curve_is_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{PRIMARY}/metrics.json",
        lambda metrics: metrics["threshold_source"].update(recall_at_1pct_fpr="val"),
    )

    refused(runs_root, "not a recall read on the test ROC curve")


def test_a_quality_report_recording_a_stop_is_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{PRIMARY}/ulb_quality_report.json",
        lambda report: report.update(stops=["The test fold holds no fraud."]),
    )

    refused(runs_root, "records stops")


def test_a_quality_report_of_other_folds_is_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{PRIMARY}/ulb_quality_report.json",
        lambda report: report["chronological_split"]["folds"][2].update(frauds=51),
    )

    refused(runs_root, "describes folds")


def test_calibration_measured_on_other_rows_is_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{SECOND}/calibration_metrics.json",
        lambda calibration: calibration.update(n_test_samples=1),
    )

    refused(runs_root, "was measured on 1 rows")


def test_runs_on_different_source_data_are_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{THIRD}/run.json",
        lambda record: record["dataset"]["files"].update({"creditcard.csv": "0" * 64}),
    )

    refused(runs_root, "record different source data")


def test_runs_scored_on_different_test_folds_are_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{SECOND}/run.json",
        lambda record: record["test_fold_identity"].update(transaction_ids_sha256="f" * 64),
    )

    refused(runs_root, "record different test fold")


def test_runs_fitted_with_different_settings_are_refused(runs_root: Path) -> None:
    edit(
        runs_root,
        f"{SECOND}/training_metadata.json",
        lambda fit: fit.update(tuning_iterations=10),
    )

    refused(runs_root, "record different fit settings")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_the_command_writes_the_card_with_lf_line_endings(runs_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "ULB_BENCHMARK_CARD.md"

    assert card.main(["--runs-root", str(runs_root), "--output", str(output)]) == 0
    assert b"\r\n" not in output.read_bytes()
    assert output.read_text(encoding="utf-8") == build(runs_root)


def test_the_command_writes_nothing_when_the_records_are_refused(
    runs_root: Path, tmp_path: Path
) -> None:
    (runs_root / PRIMARY / "metrics.json").unlink()
    output = tmp_path / "ULB_BENCHMARK_CARD.md"

    assert card.main(["--runs-root", str(runs_root), "--output", str(output)]) == 1
    assert not output.exists()
