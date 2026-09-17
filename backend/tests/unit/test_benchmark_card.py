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
            "bin_counts": [v["test"] - 45, *([5] * 9)],
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
                "interval": "half-open: start <= timestamp < end",
                "rows": 7001,
                "fraud_count": 41,
            },
            "val": {
                "name": "val",
                "start": "2019-11-01T00:00:00+00:00",
                "end": "2020-01-01T00:00:00+00:00",
                "interval": "half-open: start <= timestamp < end",
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


# ---------------------------------------------------------------------------
# Rendering: helpers that read the card back
# ---------------------------------------------------------------------------


def build(root: Path) -> str:
    return card.build_benchmark_card(card.load_benchmark(root))


def section(text: str, heading: str) -> str:
    """The body of the `## heading` section, up to the next `## ` heading."""
    start = text.index(f"\n## {heading}")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def row(text: str, label: str) -> list[str]:
    """The cells of the first table row whose first cell is `label`."""
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("| ") and cells[0] == label:
            return cells
    raise AssertionError(f"No table row labelled {label!r}.")


def f4(value: float) -> str:
    return f"{value:.4f}"


RESULTS = "1. Results — never merged"
THRESHOLDS = "2. At each operating threshold"
IN_DOMAIN_LABELS = {
    SYNTHETIC: "Synthetic baseline",
    DEV: "Sparkov in-domain, development (200 cards)",
    FULL: "Sparkov in-domain, full corpus (999 cards)",
}


# ---------------------------------------------------------------------------
# Rendering: results, thresholds and context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [SYNTHETIC, DEV, FULL])
def test_each_in_domain_result_is_read_from_its_own_run(runs_root: Path, name: str) -> None:
    v = values(name)

    cells = row(section(build(runs_root), RESULTS), IN_DOMAIN_LABELS[name])

    assert cells[1:5] == [
        f"`{name}`",
        f"`{name}` train fold",
        f"`{name}` test fold",
        f"`{name}` val fold",
    ]
    assert cells[5:] == [
        f"{v['test']:,}",
        f"{v['test_frauds']:,}",
        f4(v["prevalence"]),
        f4(v["pr_auc"]),
        f4(v["roc_auc"]),
        f4(v["recall_1"]),
        f4(v["recall_5"]),
    ]


def test_the_transfer_result_is_read_from_the_transfer_record(runs_root: Path) -> None:
    transfer = transfer_record()
    free, context = transfer["threshold_free"], transfer["context"]

    cells = row(section(build(runs_root), RESULTS), "Cross-generator transfer")

    assert cells[1:5] == [
        f"`{SYNTHETIC}` → `{FULL}`",
        f"`{SYNTHETIC}` train fold, not retrained",
        f"`{FULL}` test fold",
        f"`{SYNTHETIC}` val fold; nothing selected on `{FULL}`",
    ]
    assert cells[5:] == [
        f"{context['target_test_rows']:,}",
        f"{context['target_test_fraud_count']:,}",
        f4(context["target_test_prevalence"]),
        f4(free["pr_auc"]),
        f4(free["roc_auc"]),
        f4(free["recall_at_1pct_fpr"]),
        f4(free["recall_at_5pct_fpr"]),
    ]


def test_a_changed_record_value_changes_that_value_on_the_card(runs_root: Path) -> None:
    edit_json(runs_root / FULL / "metrics.json", lambda payload: payload.update(test_pr_auc=0.1234))

    cells = row(section(build(runs_root), RESULTS), IN_DOMAIN_LABELS[FULL])

    assert cells[8] == "0.1234"


def test_the_results_are_four_rows_that_are_never_merged(runs_root: Path) -> None:
    results = section(build(runs_root), RESULTS)
    header, *rows = [line for line in results.splitlines() if line.startswith("| ")]

    assert [cells.split("|")[1].strip() for cells in rows] == [
        *IN_DOMAIN_LABELS.values(),
        "Cross-generator transfer",
    ]
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    assert columns.index("Test prevalence") + 1 == columns.index("PR-AUC")
    assert "ULB" not in results


def test_the_development_run_is_labelled_with_the_cards_its_record_selected(
    runs_root: Path,
) -> None:
    edit_json(
        runs_root / DEV / "run.json",
        lambda payload: payload["dataset"]["subsample"].update(selected_entities=150),
    )

    rows = section(build(runs_root), RESULTS)

    assert row(rows, "Sparkov in-domain, development (150 cards)")[1] == f"`{DEV}`"


def test_a_recall_not_read_on_a_test_roc_curve_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "metrics.json",
        lambda payload: payload["threshold_source"].update(recall_at_1pct_fpr="threshold.json"),
    )

    with pytest.raises(card.BenchmarkCardError, match="not a recall read on a test ROC curve"):
        build(runs_root)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        (SYNTHETIC, "Synthetic baseline"),
        (DEV, "Sparkov in-domain, development"),
        (FULL, "Sparkov in-domain, full corpus"),
    ],
)
def test_each_in_domain_threshold_row_is_read_from_its_own_run(
    runs_root: Path, name: str, label: str
) -> None:
    v = values(name)

    cells = row(section(build(runs_root), THRESHOLDS), label)

    assert cells[1:] == [
        f4(v["threshold"]),
        f"`{name}` val fold",
        "0.0100",
        f4(v["val_fpr"]),
        "no",
        f4(v["precision"]),
        f4(v["recall"]),
        f4(v["f1"]),
        f"{v['tp']:,}",
        f"{v['fp']:,}",
        f"{v['tn']:,}",
        f"{v['fn']:,}",
        f4(v["test_fpr"]),
    ]


def test_the_transfer_is_measured_at_the_source_runs_threshold(runs_root: Path) -> None:
    transfer = transfer_record()
    at, threshold = transfer["at_source_threshold"], transfer["source_threshold"]

    thresholds = section(build(runs_root), THRESHOLDS)
    cells = row(thresholds, "Cross-generator transfer, at the source run's threshold")

    assert cells[1:] == [
        f4(values(SYNTHETIC)["threshold"]),
        threshold["selected_on"],
        f4(threshold["fpr_ceiling_on_source_val"]),
        f4(threshold["realised_fpr_on_source_val"]),
        "no",
        f4(at["precision"]),
        f4(at["recall"]),
        f4(at["f1"]),
        f"{at['true_positives']:,}",
        f"{at['false_positives']:,}",
        f"{at['true_negatives']:,}",
        f"{at['false_negatives']:,}",
        f4(transfer["context"]["realised_fpr_on_target_test_at_source_threshold"]),
    ]


def test_fraud_counts_are_named_as_source_or_target_for_the_transfer(runs_root: Path) -> None:
    context = transfer_record()["context"]

    counts = section(build(runs_root), THRESHOLDS).split("**Fraud counts per fold.**")[1]

    assert row(counts, "Cross-generator transfer")[1:] == [
        f"{context['source_train_fraud_count']} (source `{SYNTHETIC}`)",
        f"{context['source_val_fraud_count']} (source `{SYNTHETIC}`)",
        f"{context['target_test_fraud_count']} (target `{FULL}`)",
    ]
    for name, label in (
        (DEV, "Sparkov in-domain, development"),
        (FULL, "Sparkov in-domain, full corpus"),
    ):
        v = values(name)
        assert row(counts, label)[1:] == [
            str(v["train_frauds"]),
            str(v["val_frauds"]),
            str(v["test_frauds"]),
        ]


def test_the_label_delay_note_is_stated_once(runs_root: Path) -> None:
    text = build(runs_root)

    assert text.count(LABEL_DELAY_NOTE) == 1


def test_synthetic_era_targets_never_appear_on_the_card(runs_root: Path) -> None:
    text = build(runs_root)

    synthetic_metrics = read_json(runs_root / SYNTHETIC / "metrics.json")
    assert {"target_pr_auc", "target_recall_at_1pct_fpr"} <= set(synthetic_metrics)
    assert "target_pr_auc" not in text and "target_recall" not in text
    assert f4(synthetic_metrics["target_pr_auc"]) not in text
    assert f4(synthetic_metrics["target_recall_at_1pct_fpr"]) not in text


def test_every_record_read_is_listed_with_its_digest(runs_root: Path) -> None:
    records = card.load_benchmark(runs_root)

    text = card.build_benchmark_card(records)

    assert f"Records read ({len(records.files)})" in text
    for record in records.files:
        assert f"| `{record.source}` | `{record.sha256}` |" in text


def test_the_card_is_the_same_bytes_every_time_it_is_built(runs_root: Path) -> None:
    assert build(runs_root) == build(runs_root)
    assert "Generated at" not in build(runs_root)


# ---------------------------------------------------------------------------
# Rendering: live features, calibration and feature importance
# ---------------------------------------------------------------------------

LIVE = "3. Live features"
CALIBRATION = "4. Calibration — test folds, no calibrator fitted"
IMPORTANCE = "5. Global feature importance — test folds"


def test_live_features_are_read_from_each_runs_fit_record(runs_root: Path) -> None:
    live = section(build(runs_root), LIVE)

    assert row(live, f"`{SYNTHETIC}`")[1:] == ["17", "17", "none", "17", "none"]
    constant = ", ".join(f"`{name}`" for name in SPARKOV_CONSTANT)
    for name in (DEV, FULL):
        assert row(live, f"`{name}`")[1:] == ["17", "12", constant, "12", constant]


def test_a_changed_live_count_changes_that_count_on_the_card(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "training_metadata.json",
        lambda payload: payload.update(live_feature_count=11),
    )

    assert row(section(build(runs_root), LIVE), f"`{DEV}`")[2] == "11"


def test_the_transfer_feature_context_is_read_from_the_transfer_record(runs_root: Path) -> None:
    live = section(build(runs_root), LIVE)

    assert f"Constant in its training fold: none. Constant in the `{FULL}` test fold: " in live
    assert ", ".join(f"`{name}`" for name in SPARKOV_CONSTANT) in live.split("**Transfer.**")[1]
    expected = f"`customer_account_age_days` {values(FULL)['test']:,}."
    assert f"Target test rows outside the source training range, by feature: {expected}" in live


@pytest.mark.parametrize("name", [SYNTHETIC, DEV, FULL])
def test_calibration_is_read_from_each_runs_calibration_record(runs_root: Path, name: str) -> None:
    calibration = read_json(runs_root / name / "calibration_metrics.json")
    worst = calibration["worst_positive_bin"]

    cells = row(section(build(runs_root), CALIBRATION), f"`{name}`")

    assert cells[1:] == [
        f"{values(name)['test']:,}",
        f4(calibration["brier_score"]),
        f4(calibration["positive_class_brier"]),
        f4(calibration["expected_calibration_error"]),
        f4(calibration["positive_class_ece"]),
        f"bin 8: mean score {f4(worst['mean_predicted'])}, fraud rate "
        f"{f4(worst['mean_observed'])} ({worst['n_samples']} rows, {worst['n_positives']} frauds)",
    ]


def test_the_number_of_calibration_bins_is_read_from_the_records(runs_root: Path) -> None:
    assert "with 10 equal-width score bins" in section(build(runs_root), CALIBRATION)


def test_calibration_records_with_different_bin_counts_are_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "calibration_metrics.json",
        lambda payload: payload.update(bin_counts=payload["bin_counts"][:5]),
    )

    with pytest.raises(card.BenchmarkCardError, match="different numbers of bins"):
        build(runs_root)


def test_a_run_without_a_bin_holding_frauds_shows_no_gap(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "calibration_metrics.json",
        lambda payload: payload.update(worst_positive_bin=None),
    )

    assert row(section(build(runs_root), CALIBRATION), f"`{DEV}`")[-1] == "—"


@pytest.mark.parametrize("filename", ["calibration_metrics.json", "feature_importance.json"])
def test_an_analysis_of_another_number_of_test_rows_is_refused(
    runs_root: Path, filename: str
) -> None:
    edit_json(runs_root / FULL / filename, lambda payload: payload.update(n_test_samples=7))

    with pytest.raises(card.BenchmarkCardError, match="was measured on 7 rows"):
        build(runs_root)


def test_feature_importance_is_ranked_by_each_runs_recorded_rank(runs_root: Path) -> None:
    importance = section(build(runs_root), IMPORTANCE)

    assert "in log-odds" in importance
    first = read_json(runs_root / SYNTHETIC / "feature_importance.json")["features"][0]
    assert row(importance, "1")[1] == f"`{first['feature']}` {f4(first['mean_abs_shap'])}"
    assert row(importance, "1")[2].startswith(f"`{FEATURES[0]}` ")
    assert row(importance, "17")[3] == "`is_high_risk_category` 0.0000"


def test_feature_importance_ignores_the_order_entries_are_stored_in(runs_root: Path) -> None:
    before = section(build(runs_root), IMPORTANCE)
    edit_json(
        runs_root / DEV / "feature_importance.json",
        lambda payload: payload.update(features=list(reversed(payload["features"]))),
    )

    assert section(build(runs_root), IMPORTANCE) == before


def test_feature_importance_in_different_units_is_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DEV / "feature_importance.json",
        lambda payload: payload.update(units="probability"),
    )

    with pytest.raises(card.BenchmarkCardError, match="use different units"):
        build(runs_root)


# ---------------------------------------------------------------------------
# Rendering: temporal drift
# ---------------------------------------------------------------------------

DRIFT = "6. Temporal drift"


def test_the_drift_periods_are_read_from_the_drift_record(runs_root: Path) -> None:
    periods = drift_record()["periods"]

    drift = section(build(runs_root), DRIFT)

    for name in ("train", "val"):
        period = periods[name]
        assert row(drift, name)[1:] == [
            period["start"],
            period["end"],
            period["interval"],
            f"{period['rows']:,}",
            f"{period['fraud_count']:,}",
        ]


def test_the_drift_procedure_is_read_from_the_drift_record(runs_root: Path) -> None:
    record = drift_record()
    tuning, fit, threshold = record["tuning"], record["fit"], record["threshold"]

    drift = section(build(runs_root), DRIFT)

    assert row(drift, "Hyperparameter search")[1] == (
        "25 iterations over 4 folds of the train period, random state 42"
    )
    assert row(drift, "Cross-validated PR-AUC")[1] == (
        f"{f4(tuning['best_cv_pr_auc'])} — {tuning['note']}"
    )
    assert row(drift, "Chosen hyperparameters")[1] == "`max_depth=8`, `n_estimators=400`"
    assert row(drift, "scale_pos_weight")[1] == f4(fit["scale_pos_weight"])
    assert row(drift, "Early stopping")[1] == (
        "after 50 rounds without improvement on the val period; best round 397"
    )
    assert row(drift, "Operating threshold")[1] == (
        f"{f4(threshold['value'])}, selected on the val period for a target FPR of 0.0100, "
        f"realising {f4(threshold['realised_fpr_on_val'])} there; fallback used: no. "
        f"{threshold['note']}"
    )
    assert row(drift, "Rows outside every period")[1] == "0"
    assert record["note"] in drift


def test_every_drift_month_is_read_in_calendar_order(runs_root: Path) -> None:
    months = drift_record()["months"]

    drift = section(build(runs_root), DRIFT)

    labels = [
        line.split("|")[1].strip() for line in drift.splitlines() if line.startswith("| `2020")
    ]
    assert labels == [f"`{month['name']}`" for month in months]
    march = months[2]
    assert row(drift, "`2020-03`")[1:] == [
        march["interval"],
        f"{march['volume']:,}",
        f"{march['fraud_count']:,}",
        f4(march["fraud_rate"]),
        f4(march["pr_auc"]),
        f4(march["recall"]),
        f4(march["precision"]),
        f4(march["realised_fpr"]),
    ]


def test_a_month_that_leaves_metrics_undefined_shows_them_as_dashes(runs_root: Path) -> None:
    december = row(section(build(runs_root), DRIFT), "`2020-12`")

    realised_fpr = f4(drift_record()["months"][11]["realised_fpr"])
    assert december[3:] == ["0", f4(0.0), "—", "—", "—", realised_fpr]
    assert "fixture undefined-metrics note" in section(build(runs_root), DRIFT)


def test_drift_months_out_of_calendar_order_are_refused(runs_root: Path) -> None:
    edit_json(
        runs_root / DRIFT_RECORD,
        lambda payload: payload.update(months=list(reversed(payload["months"]))),
    )

    with pytest.raises(card.BenchmarkCardError, match="not in calendar order"):
        build(runs_root)


def test_a_drift_month_missing_a_metric_is_refused_by_its_position(runs_root: Path) -> None:
    edit_json(runs_root / DRIFT_RECORD, lambda payload: payload["months"][4].pop("recall"))

    with pytest.raises(card.BenchmarkCardError, match=re.escape("months[4] has no recall")):
        build(runs_root)


def test_the_drift_label_delay_note_is_not_repeated_when_it_matches(runs_root: Path) -> None:
    text = build(runs_root)

    assert "**Label delay.** As stated in section 2." in section(text, DRIFT)
    assert text.count(LABEL_DELAY_NOTE) == 1


# ---------------------------------------------------------------------------
# Rendering: rules audit
# ---------------------------------------------------------------------------

AUDIT = "7. Rules audit"


def test_the_audited_population_is_stated_from_the_audit_record(runs_root: Path) -> None:
    population = rules_audit_record()["population"]

    audit = section(build(runs_root), AUDIT)

    assert f"on every row of `{FULL}`: {population['row_count']:,} rows holding " in audit
    assert f"{population['fraud_count']:,} frauds (rate {f4(population['fraud_rate'])})" in audit
    assert f"{population['first_timestamp']} to {population['last_timestamp']}" in audit
    assert population["note"] in audit
    assert "over 180 days" in audit


@pytest.mark.parametrize(
    ("rule", "severity", "fired", "on_fraud"),
    [entry for entry in RULES if entry[1] is not None],
)
def test_each_evaluable_rule_is_read_from_the_audit_record(
    runs_root: Path, rule: str, severity: str, fired: int, on_fraud: int
) -> None:
    recorded = next(entry for entry in rules_audit_record()["rules"] if entry["rule"] == rule)

    cells = row(section(build(runs_root), AUDIT), f"`{rule}`")

    assert cells[1:] == [
        severity,
        "yes",
        f"{fired:,}",
        f"{on_fraud:,}",
        f4(recorded["precision"]) if recorded["precision"] is not None else "—",
        f4(recorded["fraud_recall"]),
    ]


def test_a_rule_the_data_cannot_support_is_shown_as_not_evaluable(runs_root: Path) -> None:
    cells = row(section(build(runs_root), AUDIT), "`dormant_account_high_value`")

    assert cells[1:] == [
        "—",
        "no — not evaluable: no account-open timestamp",
        "—",
        "—",
        "—",
        "—",
    ]


def test_the_rules_only_outcome_is_read_from_the_audit_record(runs_root: Path) -> None:
    outcome = rules_audit_record()["rules_only_outcome"]

    audit = section(build(runs_root), AUDIT)

    assert outcome["note"] in audit and outcome["decision_rule"] in audit
    declines = [
        line.split("|")[1].strip() for line in audit.splitlines() if line.startswith("| DE")
    ]
    assert declines == ["DECLINE"]
    for name, counts in outcome["by_outcome"].items():
        assert row(audit, name)[1:] == [
            f"{counts['rows']:,}",
            f"{counts['frauds']:,}",
            f"{counts['legitimate']:,}",
        ]


def test_the_outcomes_are_listed_from_the_strictest_down(runs_root: Path) -> None:
    audit = section(build(runs_root), AUDIT)

    outcomes = [
        line.split("|")[1].strip()
        for line in audit.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip() in {"DECLINE", "REVIEW", "APPROVE"}
    ]
    assert outcomes == ["DECLINE", "REVIEW", "APPROVE"]


def test_a_rule_entry_missing_a_count_is_refused_by_its_position(runs_root: Path) -> None:
    edit_json(
        runs_root / FULL / "rules_audit.json",
        lambda payload: payload["rules"][0].pop("rows_fired"),
    )

    with pytest.raises(card.BenchmarkCardError, match=re.escape("rules[0] has no rows_fired")):
        build(runs_root)


# ---------------------------------------------------------------------------
# Rendering: data quality, provenance and limitations
# ---------------------------------------------------------------------------

QUALITY = "8. Data quality — Sparkov runs"
PROVENANCE = "9. Provenance"
LIMITATIONS = "10. Accepted limitations"


@pytest.mark.parametrize("name", [DEV, FULL])
def test_each_sparkov_load_is_read_from_its_quality_report(runs_root: Path, name: str) -> None:
    report = quality_report(name)

    cells = row(section(build(runs_root), QUALITY), f"`{name}`")

    assert cells[1:7] == [
        f"{report['rows']['raw']:,}",
        f"{report['rows']['kept']:,}",
        "0",
        f"{report['entities']['customers']:,}",
        f"{report['entities']['merchants']:,}",
        f"{report['label']['fraud_count']:,}",
    ]


@pytest.mark.parametrize("name", [DEV, FULL])
def test_fraud_concentration_is_read_from_the_quality_report(runs_root: Path, name: str) -> None:
    distribution = quality_report(name)["fraud_distribution"]
    folds = {fold["name"]: fold for fold in distribution["chronological_split"]["folds"]}

    quality = section(build(runs_root), QUALITY)
    cells = [line for line in quality.splitlines() if line.startswith(f"| `{name}` |")][1]

    assert [cell.strip() for cell in cells.strip("|").split("|")][1:] == [
        f"{distribution['cards']['with_fraud']:,}",
        f"{int(distribution['cards']['frauds_per_card_with_fraud']['p50']):,}",
        f"{folds['train']['cards_with_fraud']:,}",
        f"{folds['val']['cards_with_fraud']:,}",
        f"{folds['test']['cards_with_fraud']:,}",
        "; ".join(
            f"{'/'.join(boundary['between'])}: {boundary['cards_with_fraud_on_both_sides']}"
            for boundary in distribution["chronological_split"]["boundaries"]
        ),
    ]


def test_provenance_names_each_runs_code_featureset_dataset_and_libraries(runs_root: Path) -> None:
    provenance_section = section(build(runs_root), PROVENANCE)

    assert row(provenance_section, f"`{SYNTHETIC}`")[1:] == [
        f"`{CODE}`",
        "`v1`",
        "`synthetic` `v1`",
        "none",
        "numpy 2.4.6, python 3.13.14, scikit-learn 1.8.0, xgboost 3.2.0",
    ]
    assert row(provenance_section, f"`{DEV}`")[4] == "cards, seed 42, 200 of at most 200"
    assert row(provenance_section, f"`{FULL}`")[4] == "none (seed 42, 999 entities)"


def test_the_source_files_are_listed_with_their_digests(runs_root: Path) -> None:
    provenance_section = section(build(runs_root), PROVENANCE)

    cells = row(provenance_section, "`sparkov` `kaggle-2020-08-05`")
    assert cells[1:5] == ["CC0", "https://example.test/data", f"{values(FULL)['rows']:,}", "104"]
    for name, digest in sorted(SPARKOV_FILES.items()):
        assert f"`{name}` `{digest}`" in cells[5]
    assert f"`sparkov`: {provenance(FULL).notes}" in provenance_section


def test_the_limitations_are_the_ones_every_run_card_states(runs_root: Path) -> None:
    from ml.run_card import _ACCEPTED_LIMITATIONS

    limitations = section(build(runs_root), LIMITATIONS)

    for limitation in _ACCEPTED_LIMITATIONS:
        assert f"- {limitation}" in limitations
    assert "Label delay, as stated in section 2." in limitations
    assert "PHASE_5D_BENCHMARK_METHODOLOGY.md" in limitations
