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


def fold_identity(name: str) -> FoldIdentity:
    return FoldIdentity.of("test", [f"{name}-tx{i}" for i in range(values(name)["test"])])


def transfer_record() -> dict[str, Any]:
    """The transfer of the synthetic run's model onto the full run's test fold."""
    source, target = values(SYNTHETIC), values(FULL)
    identity = fold_identity(FULL)
    confusion = {
        "threshold": source["threshold"],
        "precision": 0.0031,
        "recall": 0.0622,
        "f1": 0.0055,
        "true_positives": 5,
        "false_positives": 1900,
        "true_negatives": 25000,
        "false_negatives": 70,
    }
    return {
        "transfer_metrics_version": "1",
        "source_run": SYNTHETIC,
        "target_run": FULL,
        "method_note": "fixture method note",
        "source_featureset": "v1",
        "target_featureset": "v1",
        "source_model_sha256": str(INDEX[SYNTHETIC]) * 64,
        "source_threshold": {
            "value": source["threshold"],
            "selected_on": f"the val fold of source run {SYNTHETIC!r}",
            "fpr_ceiling_on_source_val": 0.01,
            "realised_fpr_on_source_val": source["val_fpr"],
            "fallback_used": False,
        },
        "threshold_free": {
            "pr_auc": 0.0444,
            "roc_auc": 0.7444,
            "recall_at_1pct_fpr": 0.0333,
            "recall_at_5pct_fpr": 0.0555,
            "threshold_source": {
                "recall_at_1pct_fpr": "target_test_roc_curve",
                "recall_at_5pct_fpr": "target_test_roc_curve",
            },
            "note": "fixture threshold-free note",
        },
        "at_source_threshold": {
            **confusion,
            "realised_fpr_on_target_test": 0.0707,
            "threshold_source": "source_run_threshold",
            "note": "fixture source-threshold note",
        },
        "context": {
            "target_test_rows": target["test"],
            "target_test_prevalence": 0.0077,
            "target_test_fraud_count": target["test_frauds"],
            "source_train_fraud_count": source["train_frauds"],
            "source_val_fraud_count": source["val_frauds"],
            "realised_fpr_on_target_test_at_source_threshold": 0.0707,
            "source_threshold_fallback_used": False,
            "label_delay_note": LABEL_DELAY_NOTE,
            "target_test_transaction_count": identity.transaction_count,
            "target_test_transaction_ids_sha256": identity.transaction_ids_sha256,
        },
        "features": {
            "featureset_version": "v1",
            "scored_columns": FEATURES,
            "constant_in_source_training_fold": [],
            "constant_in_target_test_fold": SPARKOV_CONSTANT,
            "by_feature": [
                {
                    "feature": feature,
                    "target_test_rows_outside_source_training_range": (
                        target["test"] if feature == "customer_account_age_days" else 0
                    ),
                }
                for feature in FEATURES
            ],
        },
        "verification": {
            side: {
                "run": run,
                "metrics_reproduced": True,
                "model_digest_verified": True,
                "test_fold_identity_verified": True,
            }
            for side, run in (("source", SYNTHETIC), ("target", FULL))
        },
    }


RULES = (
    ("velocity_burst", "HARD_BLOCK", 17, 3),
    ("geo_velocity_impossible", "HARD_BLOCK", 0, 0),
    ("amount_ceiling", "REVIEW", 19, 0),
    ("high_risk_country", "REVIEW", 0, 0),
    ("dormant_account_high_value", None, None, None),
    ("off_hours_high_value", "REVIEW", 131, 29),
)


def rules_audit_record() -> dict[str, Any]:
    """The final rules audit over every row of the full run."""
    full = values(FULL)
    frauds = full["train_frauds"] + full["val_frauds"] + full["test_frauds"]
    rules: list[dict[str, Any]] = []
    for name, severity, fired, on_fraud in RULES:
        if severity is None:
            rules.append(
                {
                    "rule": name,
                    "evaluable": False,
                    "reason": "not evaluable: no account-open timestamp",
                }
            )
            continue
        rules.append(
            {
                "rule": name,
                "evaluable": True,
                "severity": severity,
                "rows_evaluated": full["rows"],
                "rows_fired": fired,
                "fired_on_fraud": on_fraud,
                "precision": on_fraud / fired if fired else None,
                "fraud_recall": on_fraud / frauds,
            }
        )
    return {
        "rules_audit_version": "1",
        "run": FULL,
        "population": {
            "run": FULL,
            "rows": "all",
            "row_count": full["rows"],
            "fraud_count": frauds,
            "fraud_rate": frauds / full["rows"],
            "first_timestamp": START.isoformat(),
            "last_timestamp": (START + timedelta(days=100)).isoformat(),
            "note": "fixture population note",
        },
        "context": {
            "history_window_days": 180,
            "definition": "fixture context definition",
            "history_drawn_from": "every row of the run's dataset",
        },
        "rules": rules,
        "rules_only_outcome": {
            "rules_used": [name for name, severity, _, _ in RULES if severity is not None],
            "rules_not_evaluable": ["dormant_account_high_value"],
            "note": "Computed from the 5 evaluable rules of 6. Not evaluable: "
            "dormant_account_high_value (not evaluable: no account-open timestamp).",
            "decision_rule": "fixture decision rule",
            "by_outcome": {
                "DECLINE": {"rows": 17, "frauds": 3, "legitimate": 14},
                "REVIEW": {"rows": 150, "frauds": 29, "legitimate": 121},
                "APPROVE": {
                    "rows": full["rows"] - 167,
                    "frauds": frauds - 32,
                    "legitimate": full["rows"] - 167 - (frauds - 32),
                },
            },
        },
        "verification": {
            "metrics_reproduced": True,
            "model_digest_verified": True,
            "test_fold_identity_verified": True,
        },
    }


def drift_record() -> dict[str, Any]:
    """The drift experiment on the full corpus, with a final month that holds no fraud."""
    months = []
    for month in range(1, 13):
        frauds = 0 if month == 12 else 30 + month
        months.append(
            {
                "name": f"2020-{month:02d}",
                "start": datetime(2020, month, 1, tzinfo=UTC).isoformat(),
                "end": (
                    datetime(2021, 1, 1, tzinfo=UTC)
                    if month == 12
                    else datetime(2020, month + 1, 1, tzinfo=UTC)
                ).isoformat(),
                "interval": "half-open: start <= timestamp < end",
                "volume": 5000 + month,
                "fraud_count": frauds,
                "fraud_rate": frauds / (5000 + month),
                "pr_auc": None if month == 12 else 0.8 + month * 0.0101,
                "recall": None if month == 12 else 0.9 + month * 0.0011,
                "precision": None if month == 12 else 0.3 + month * 0.0011,
                "realised_fpr": 0.009 + month * 0.0001,
                "true_positives": 0 if month == 12 else frauds - 1,
                "false_positives": 0 if month == 12 else 40 + month,
                "true_negatives": 5000 + month - frauds - (0 if month == 12 else 40 + month),
                "false_negatives": 0 if month == 12 else 1,
            }
        )
    return {
        "drift_metrics_version": "1",
        "run_name": "sparkov_v1_drift",
        "note": "fixture drift note",
        "dataset": provenance(FULL).to_dict(),
        "featureset_version": "v1",
        "periods": {
            "train": {
                "name": "train",
                "start": "2019-01-01T00:00:00+00:00",
                "end": "2019-11-01T00:00:00+00:00",
                "rows": 7001,
                "fraud_count": 41,
            },
            "val": {
                "name": "val",
                "start": "2019-11-01T00:00:00+00:00",
                "end": "2020-01-01T00:00:00+00:00",
                "rows": 2001,
                "fraud_count": 11,
            },
        },
        "rows_outside_periods": 0,
        "tuning": {
            "iterations": 25,
            "cv_folds": 4,
            "random_state": 42,
            "best_hyperparameters": {"max_depth": 8, "n_estimators": 400},
            "best_cv_pr_auc": 0.8999,
            "note": "fixture tuning note",
        },
        "fit": {
            "scale_pos_weight": 166.25,
            "early_stopping_rounds": 50,
            "best_iteration": 397,
            "early_stopping_against": "val period",
        },
        "threshold": {
            "value": 0.1382,
            "target_fpr": 0.01,
            "realised_fpr_on_val": 0.0093,
            "fallback_used": False,
            "selected_on": "val period",
            "note": "fixture threshold note",
        },
        "months": months,
        "undefined_metrics_note": "fixture undefined-metrics note",
        "label_delay_note": LABEL_DELAY_NOTE,
    }


def write_benchmark(root: Path) -> None:
    """Every record the benchmark card reads."""
    for name in (SYNTHETIC, DEV, FULL):
        write_run(root, name)
    write_json(root / FULL / "transfer_metrics.json", transfer_record())
    write_json(root / FULL / "rules_audit.json", rules_audit_record())
    write_json(root / "sparkov_v1_drift" / "drift_metrics.json", drift_record())


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    write_benchmark(root)
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


# ---------------------------------------------------------------------------
# Reading the benchmark: every experiment record must belong to its runs
# ---------------------------------------------------------------------------

DRIFT_RECORD = "sparkov_v1_drift/drift_metrics.json"
EXPERIMENT_RECORDS = [f"{FULL}/transfer_metrics.json", f"{FULL}/rules_audit.json", DRIFT_RECORD]


def test_the_benchmark_is_read_with_its_runs_and_experiment_records(runs_root: Path) -> None:
    records = card.load_benchmark(runs_root)

    assert (records.synthetic.name, records.dev.name, records.full.name) == (SYNTHETIC, DEV, FULL)
    assert [f.source for f in records.files[-3:]] == EXPERIMENT_RECORDS
    assert len({f.source for f in records.files}) == len(records.files)


@pytest.mark.parametrize("source", EXPERIMENT_RECORDS)
def test_a_missing_experiment_record_is_refused_by_name(runs_root: Path, source: str) -> None:
    (runs_root / source).unlink()

    with pytest.raises(card.BenchmarkCardError, match=re.escape(f"{source} is missing")):
        card.load_benchmark(runs_root)


def test_runs_on_different_source_data_are_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "run.json",
        lambda payload: payload["dataset"]["files"].update({"fraudTest.csv": "9" * 64}),
    )

    with pytest.raises(card.BenchmarkCardError, match="record different source data"):
        card.load_benchmark(runs_root)


def test_a_development_run_without_a_card_subsample_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "run.json", lambda payload: payload["dataset"].update(subsample=None)
    )

    with pytest.raises(card.BenchmarkCardError, match="does not record a card subsample"):
        card.load_benchmark(runs_root)


def test_a_subsampled_full_run_is_refused(runs_root: Path) -> None:
    subsample = {"strategy": "cards", "seed": 42, "max_entities": 500, "selected_entities": 500}
    edit_json(
        runs_root / FULL / "run.json",
        lambda payload: payload["dataset"].update(subsample=subsample),
    )

    with pytest.raises(card.BenchmarkCardError, match="the full run covers the whole corpus"):
        card.load_benchmark(runs_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_run", DEV), ("target_run", DEV)],
    ids=["another-source", "another-target"],
)
def test_a_transfer_between_other_runs_is_refused(runs_root: Path, field: str, value: str) -> None:
    edit_json(
        runs_root / FULL / "transfer_metrics.json", lambda payload: payload.update({field: value})
    )

    with pytest.raises(card.BenchmarkCardError, match="the benchmark's transfer measures"):
        card.load_benchmark(runs_root)


def test_a_transfer_that_scored_another_model_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / FULL / "transfer_metrics.json",
        lambda payload: payload.update(source_model_sha256="f" * 64),
    )

    with pytest.raises(card.BenchmarkCardError, match="scored with a model other than"):
        card.load_benchmark(runs_root)


def test_a_transfer_at_another_threshold_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / FULL / "transfer_metrics.json",
        lambda payload: payload["source_threshold"].update(value=0.5),
    )

    with pytest.raises(card.BenchmarkCardError, match="applied a threshold other than"):
        card.load_benchmark(runs_root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("target_test_transaction_count", 1), ("target_test_transaction_ids_sha256", "e" * 64)],
    ids=["count", "digest"],
)
def test_a_transfer_on_other_test_rows_is_refused(runs_root: Path, field: str, value: Any) -> None:
    edit_json(
        runs_root / FULL / "transfer_metrics.json",
        lambda payload: payload["context"].update({field: value}),
    )

    with pytest.raises(card.BenchmarkCardError, match="did not score exactly the test fold"):
        card.load_benchmark(runs_root)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize(
    "flag", ["metrics_reproduced", "model_digest_verified", "test_fold_identity_verified"]
)
def test_a_transfer_with_a_failed_verification_is_refused(
    runs_root: Path, side: str, flag: str
) -> None:
    edit_json(
        runs_root / FULL / "transfer_metrics.json",
        lambda payload: payload["verification"][side].update({flag: False}),
    )

    with pytest.raises(card.BenchmarkCardError, match=f"checks as passed: {flag}"):
        card.load_benchmark(runs_root)


def test_a_rules_audit_of_another_run_is_refused(runs_root: Path) -> None:
    edit_json(runs_root / FULL / "rules_audit.json", lambda payload: payload.update(run=DEV))

    with pytest.raises(card.BenchmarkCardError, match="audits another run"):
        card.load_benchmark(runs_root)


def test_a_rules_audit_of_one_fold_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / FULL / "rules_audit.json",
        lambda payload: payload["population"].update(rows="test"),
    )

    with pytest.raises(card.BenchmarkCardError, match="the final audit covers every row"):
        card.load_benchmark(runs_root)


def test_a_rules_audit_of_a_different_row_count_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / FULL / "rules_audit.json",
        lambda payload: payload["population"].update(row_count=1),
    )

    with pytest.raises(card.BenchmarkCardError, match="counts 1 rows"):
        card.load_benchmark(runs_root)


def test_a_rules_audit_with_a_failed_verification_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / FULL / "rules_audit.json",
        lambda payload: payload["verification"].update(metrics_reproduced=False),
    )

    with pytest.raises(card.BenchmarkCardError, match="checks as passed: metrics_reproduced"):
        card.load_benchmark(runs_root)


def test_a_drift_record_of_another_run_is_refused(runs_root: Path) -> None:
    edit_json(runs_root / DRIFT_RECORD, lambda payload: payload.update(run_name="elsewhere"))

    with pytest.raises(card.BenchmarkCardError, match="names run 'elsewhere'"):
        card.load_benchmark(runs_root)


def test_a_drift_record_on_another_dataset_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DRIFT_RECORD,
        lambda payload: payload["dataset"].update(row_count=payload["dataset"]["row_count"] + 1),
    )

    with pytest.raises(card.BenchmarkCardError, match=r"ran on another dataset .*row_count"):
        card.load_benchmark(runs_root)


def test_a_drift_record_with_another_featureset_is_refused(runs_root: Path) -> None:
    edit_json(runs_root / DRIFT_RECORD, lambda payload: payload.update(featureset_version="v2"))

    with pytest.raises(card.BenchmarkCardError, match="records another featureset"):
        card.load_benchmark(runs_root)


def test_a_drift_record_differing_only_in_retrieval_time_is_read(runs_root: Path) -> None:
    edit_json(
        runs_root / DRIFT_RECORD,
        lambda payload: payload["dataset"].update(retrieved_at="2026-09-18T00:00:00+00:00"),
    )

    assert card.load_benchmark(runs_root).drift.get("run_name") == "sparkov_v1_drift"
