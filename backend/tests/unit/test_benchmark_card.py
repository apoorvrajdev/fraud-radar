"""The Phase 5D benchmark card is built from the recorded runs, and only from them.

The fixture writes a complete set of benchmark records into a temporary runs
root: run records through `RunMetadata`, the other records as the JSON the
pipeline writes. Every run carries distinct values, so a value on the card can
be traced to the one record it was read from.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.fraud.feature_spec import FEATURESETS
from ml import benchmark_card as card
from ml.datasets.base import DataOrigin, DatasetProvenance, Subsample
from ml.reporting import LABEL_DELAY_NOTE
from ml.runs import FoldIdentity, RunMetadata, SplitPeriod, save_run_metadata

SYNTHETIC, DEV, FULL = "synthetic_v1", "sparkov_v1_200cards", "sparkov_v1_full"
FEATURES = FEATURESETS["v1"]
CODE = "b" * 40
LIBRARIES = {"numpy": "2.4.6", "python": "3.13.14", "scikit-learn": "1.8.0", "xgboost": "3.2.0"}
START = datetime(2019, 1, 1, tzinfo=UTC)
SPARKOV_FILES = {"fraudTest.csv": "1" * 64, "fraudTrain.csv": "2" * 64}
SPARKOV_CONSTANT = [
    "country_mismatch_customer",
    "country_mismatch_merchant",
    "customer_account_age_days",
    "customer_risk_tier_encoded",
    "is_high_risk_category",
]

# One index per run, so each run's values differ from every other run's.
INDEX = {SYNTHETIC: 1, DEV: 2, FULL: 3}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def edit_json(path: Path, change: Any) -> None:
    payload = read_json(path)
    change(payload)
    write_json(path, payload)


def values(name: str) -> dict[str, Any]:
    """The distinct numbers the fixture records for one run."""
    k = INDEX[name]
    return {
        "rows": 1000 * k,
        "train": 700 * k,
        "val": 150 * k,
        "test": 150 * k,
        "train_frauds": 70 + k,
        "val_frauds": 15 + k,
        "test_frauds": 10 + k,
        "pr_auc": 0.9 + k * 0.0111,
        "roc_auc": 0.99 + k * 0.0011,
        "recall_1": 0.8 + k * 0.0111,
        "recall_5": 0.85 + k * 0.0111,
        "threshold": 0.1 * k + 0.0123,
        "val_fpr": 0.005 + k * 0.0011,
        "precision": 0.2 + k * 0.0111,
        "recall": 0.9 + k * 0.0011,
        "f1": 0.3 + k * 0.0111,
        "tp": 9 + k,
        "fp": 20 + k,
        "tn": 110 * k,
        "fn": 1,
        "test_fpr": 0.006 + k * 0.0011,
        "prevalence": 0.003 + k * 0.0011,
        "cv_pr_auc": 0.88 + k * 0.0011,
        "best_iteration": 100 * k,
    }


def provenance(name: str) -> DatasetProvenance:
    v = values(name)
    if name == SYNTHETIC:
        dataset, version, files, subsample = (
            "synthetic",
            "v1",
            {"synthetic_transactions.csv": "3" * 64},
            None,
        )
    else:
        dataset, version, files = "sparkov", "kaggle-2020-08-05", dict(SPARKOV_FILES)
        subsample = (
            Subsample("cards", 42, 200, 200) if name == DEV else Subsample("none", 42, None, 999)
        )
    return DatasetProvenance(
        name=dataset,
        version=version,
        origin=DataOrigin.SYNTHETIC,
        source_url="https://example.test/data",
        citation=f"fixture citation for {name}",
        license="CC0",
        label_field="is_fraud",
        label_definition="1 = fraud",
        files=files,
        row_count=v["rows"],
        fraud_count=v["train_frauds"] + v["val_frauds"] + v["test_frauds"],
        period_start=START,
        period_end=START + timedelta(days=100),
        subsample=subsample,
        preprocessing=("fixture preprocessing",),
        notes=f"fixture notes for {name}",
    )


def write_run(root: Path, name: str) -> None:
    """Every record one benchmark run writes into its directory."""
    v = values(name)
    k = INDEX[name]
    directory = root / name
    record = RunMetadata(
        run_name=name,
        dataset=provenance(name),
        featureset_version="v1",
        created_at_utc="2026-09-17T00:00:00+00:00",
        code_version=CODE,
        library_versions=LIBRARIES,
        splits=(
            SplitPeriod("train", START, START + timedelta(days=70)),
            SplitPeriod("val", START + timedelta(days=70), START + timedelta(days=85)),
            SplitPeriod("test", START + timedelta(days=85), START + timedelta(days=100)),
        ),
        test_fold_identity=FoldIdentity.of("test", [f"{name}-tx{i}" for i in range(v["test"])]),
        notes=f"fixture run {name}",
    )
    save_run_metadata(record, runs_root=root)

    metrics: dict[str, Any] = {
        "test_pr_auc": v["pr_auc"],
        "test_roc_auc": v["roc_auc"],
        "recall_at_1pct_fpr": v["recall_1"],
        "recall_at_5pct_fpr": v["recall_5"],
        "at_operating_threshold": {
            "threshold": v["threshold"],
            "precision": v["precision"],
            "recall": v["recall"],
            "f1": v["f1"],
            "true_positives": v["tp"],
            "false_positives": v["fp"],
            "true_negatives": v["tn"],
            "false_negatives": v["fn"],
        },
        "threshold_source": {
            "at_operating_threshold": "threshold.json",
            "recall_at_1pct_fpr": "test_roc_curve",
            "recall_at_5pct_fpr": "test_roc_curve",
        },
        "context": {
            "test_prevalence": v["prevalence"],
            "fraud_counts": {
                "train": v["train_frauds"],
                "val": v["val_frauds"],
                "test": v["test_frauds"],
            },
            "realised_fpr_on_test_at_operating_threshold": v["test_fpr"],
            "label_delay_note": LABEL_DELAY_NOTE,
        },
        "best_cv_pr_auc": v["cv_pr_auc"],
    }
    if name == SYNTHETIC:
        metrics |= {"target_pr_auc": 0.75, "target_recall_at_1pct_fpr": 0.6}
    write_json(directory / "metrics.json", metrics)
    write_json(
        directory / "threshold.json",
        {
            "value": v["threshold"],
            "target_fpr": 0.01,
            "realised_fpr_on_val": v["val_fpr"],
            "fallback_used": False,
        },
    )

    constant = [] if name == SYNTHETIC else SPARKOV_CONSTANT
    write_json(
        directory / "training_metadata.json",
        {
            "dataset_size": v["rows"],
            "train_size": v["train"],
            "val_size": v["val"],
            "test_size": v["test"],
            "best_iteration": v["best_iteration"],
            "best_hyperparameters": {"max_depth": 3 + k, "n_estimators": 400},
            "early_stopping_rounds": 50,
            "random_state": 42,
            "tuning_iterations": 25,
            "tuning_cv_folds": 4,
            "scale_pos_weight": 100.5 + k,
            "live_feature_count": len(FEATURES) - len(constant),
            "constant_features": constant,
            "live_feature_count_whole_matrix": len(FEATURES) - len(constant),
            "constant_features_whole_matrix": constant,
            "model_sha256": str(k) * 64,
        },
    )
    write_json(directory / "feature_list.json", {"features": FEATURES})
    write_json(
        directory / "calibration_metrics.json",
        {
            "brier_score": 0.004 + k * 0.0011,
            "positive_class_brier": 0.02 + k * 0.0111,
            "expected_calibration_error": 0.007 + k * 0.0011,
            "positive_class_ece": 0.4 + k * 0.0111,
            "n_test_samples": v["test"],
            "worst_positive_bin": {
                "bin_index": 8,
                "mean_predicted": 0.85 + k * 0.0011,
                "mean_observed": 0.1 + k * 0.0111,
                "n_samples": 20 + k,
                "n_positives": 2 + k,
                "gap": 0.7,
            },
        },
    )
    ranked = list(reversed(FEATURES)) if name == SYNTHETIC else FEATURES
    write_json(
        directory / "feature_importance.json",
        {
            "features": [
                {
                    "feature": feature,
                    "rank": rank,
                    "mean_abs_shap": 0.0 if feature in constant else 3.0 - rank * 0.1 - k * 0.0011,
                }
                for rank, feature in enumerate(ranked, start=1)
            ],
            "method": "mean absolute SHAP value over test set",
            "n_test_samples": v["test"],
            "units": "log-odds",
        },
    )
    if name != SYNTHETIC:
        write_json(directory / "quality_report.json", quality_report(name))


def quality_report(name: str) -> dict[str, Any]:
    v = values(name)
    k = INDEX[name]
    cards = 200 if name == DEV else 999
    return {
        "dataset": "sparkov",
        "rows": {"raw": 9999, "kept": v["rows"], "excluded": 0},
        "entities": {"customers": cards, "merchants": 690 + k},
        "label": {"fraud_count": v["train_frauds"] + v["val_frauds"] + v["test_frauds"]},
        "fraud_distribution": {
            "cards": {
                "with_transactions": cards,
                "with_fraud": 190 + k,
                "frauds_per_card_with_fraud": {"p50": 9.0 + k, "p99": 16.0, "max": 19.0},
            },
            "chronological_split": {
                "available": True,
                "folds": [
                    {"name": "train", "cards_with_fraud": 160 + k},
                    {"name": "val", "cards_with_fraud": 20 + k},
                    {"name": "test", "cards_with_fraud": 10 + k},
                ],
                "boundaries": [
                    {"between": ["train", "val"], "cards_with_fraud_on_both_sides": k},
                    {"between": ["val", "test"], "cards_with_fraud_on_both_sides": k + 1},
                ],
            },
        },
        "extra": {"authoritative_clock": "trans_date_trans_time"},
    }


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    for name in (SYNTHETIC, DEV, FULL):
        write_run(root, name)
    return root


def load(root: Path, name: str = FULL) -> card.RecordedBenchmarkRun:
    return card.load_recorded_run(
        root,
        name,
        dataset="synthetic" if name == SYNTHETIC else "sparkov",
        with_quality_report=name != SYNTHETIC,
    )


# ---------------------------------------------------------------------------
# Reading one run's records
# ---------------------------------------------------------------------------


def test_a_run_is_read_with_every_record_it_wrote(runs_root: Path) -> None:
    run = load(runs_root, DEV)

    assert run.name == DEV
    assert run.record.run_name == DEV
    assert run.metrics.get("test_pr_auc") == values(DEV)["pr_auc"]
    assert [f.source for f in run.files] == [
        *(f"{DEV}/{filename}" for filename in card.RUN_RECORD_FILES),
        f"{DEV}/quality_report.json",
    ]


def test_a_run_without_a_quality_report_is_read_when_none_is_asked_for(runs_root: Path) -> None:
    run = load(runs_root, SYNTHETIC)

    assert run.quality is None
    assert len(run.files) == len(card.RUN_RECORD_FILES)


def test_each_record_is_identified_by_the_digest_of_its_lf_normalised_text(
    runs_root: Path, tmp_path: Path
) -> None:
    crlf_root = tmp_path / "crlf"
    for filename in (*card.RUN_RECORD_FILES, "quality_report.json"):
        lf = (runs_root / FULL / filename).read_bytes().replace(b"\r\n", b"\n")
        target = crlf_root / FULL / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(lf.replace(b"\n", b"\r\n"))
    lf_metrics = (runs_root / FULL / "metrics.json").read_bytes()

    lf_run, crlf_run = load(runs_root), load(crlf_root)

    assert b"\r\n" not in lf_metrics
    assert b"\r\n" in (crlf_root / FULL / "metrics.json").read_bytes()
    expected = hashlib.sha256(lf_metrics).hexdigest()
    assert crlf_run.metrics.sha256 == lf_run.metrics.sha256 == expected


@pytest.mark.parametrize("filename", [*card.RUN_RECORD_FILES, "quality_report.json"])
def test_a_missing_record_is_refused_by_name(runs_root: Path, filename: str) -> None:
    (runs_root / FULL / filename).unlink()

    with pytest.raises(card.BenchmarkCardError, match=re.escape(f"{FULL}/{filename} is missing")):
        load(runs_root)


def test_a_record_with_a_non_json_number_is_refused(runs_root: Path) -> None:
    path = runs_root / FULL / "calibration_metrics.json"
    path.write_text('{"brier_score": NaN}\n', encoding="utf-8")

    with pytest.raises(card.BenchmarkCardError, match="is not strict JSON"):
        load(runs_root)


def test_a_record_that_is_not_a_json_object_is_refused(runs_root: Path) -> None:
    (runs_root / FULL / "threshold.json").write_text("[0.5]\n", encoding="utf-8")

    with pytest.raises(card.BenchmarkCardError, match="does not hold a JSON object"):
        load(runs_root)


def test_a_run_record_that_cannot_be_read_as_one_is_refused(runs_root: Path) -> None:
    edit_json(runs_root / FULL / "run.json", lambda payload: payload.pop("dataset"))

    with pytest.raises(card.BenchmarkCardError, match="is not a run record"):
        load(runs_root)


def test_the_record_of_another_run_is_refused(runs_root: Path) -> None:
    (runs_root / FULL / "run.json").write_bytes((runs_root / DEV / "run.json").read_bytes())

    with pytest.raises(card.BenchmarkCardError, match="is the record of run 'sparkov_v1_200cards'"):
        load(runs_root)


def test_a_run_on_another_dataset_is_refused(runs_root: Path) -> None:
    with pytest.raises(card.BenchmarkCardError, match="records dataset 'sparkov'"):
        card.load_recorded_run(runs_root, FULL, dataset="synthetic", with_quality_report=True)


def test_a_feature_list_out_of_registered_order_is_refused(runs_root: Path) -> None:
    write_json(runs_root / FULL / "feature_list.json", {"features": list(reversed(FEATURES))})

    with pytest.raises(card.BenchmarkCardError, match="registered order"):
        load(runs_root)


def test_metrics_measured_at_another_threshold_are_refused(runs_root: Path) -> None:
    edit_json(runs_root / FULL / "threshold.json", lambda payload: payload.update(value=0.99))

    with pytest.raises(card.BenchmarkCardError, match="was measured at threshold"):
        load(runs_root)


def test_a_value_a_record_does_not_hold_is_refused_with_its_path(runs_root: Path) -> None:
    run = load(runs_root)

    expected = re.escape(f"{FULL}/metrics.json has no context.missing")
    with pytest.raises(card.BenchmarkCardError, match=expected):
        run.metrics.get("context", "missing")
