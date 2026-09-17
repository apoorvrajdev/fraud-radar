"""Phase 5D — transfer: one run's frozen model, measured on another run's test fold.

Two fixture runs are trained once, into one runs directory:

- `synthetic-fixture`, the verification suite's synthetic run, whose rows the
  stubbed database returns;
- `sparkov-fixture`, a Sparkov run trained end to end through the real
  adapter, batch builder and feature cache on the generated fixture corpus.
  Its val fold is too small for its FPR ceiling, so its threshold genuinely
  falls back.

The real Sparkov corpus is never read: the adapter is pointed at a fixture
manifest and corpus in the module's temporary directory, and a spy records
every root it loads from. Each test works on a private copy of the runs, with
every data load, every explainer and every matrix handed to a model recorded.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xgboost as xgb

from app.fraud.explainer import FraudExplainer
from app.fraud.feature_spec import FEATURESETS
from ml import loading, train
from ml.data import LabelledDataset
from ml.datasets.sparkov import SparkovAdapter, transaction_id_for
from ml.evaluation import confusion_at_threshold, pr_auc, recall_at_fpr, roc_auc
from ml.experiments import transfer
from ml.experiments.transfer import (
    FALLBACK_THRESHOLD_METHOD,
    SELECTED_THRESHOLD_METHOD,
    SOURCE_RUN_THRESHOLD_SOURCE,
    TARGET_TEST_ROC_CURVE_SOURCE,
    TRANSFER_METRICS_FILENAME,
    DataLocation,
    TransferError,
    TransferMeasurement,
)
from ml.features.cache import load_feature_cache
from ml.loading import DataRequest
from ml.paths import RAW_DATA_DIR
from ml.reporting import LABEL_DELAY_NOTE
from ml.run_verification import RunVerificationError
from ml.runs import TRANSACTION_IDS_DIGEST_DEFINITION, transaction_ids_digest
from ml.splits import SplitIndices, chronological_split
from ml.tuning import TuningResult
from tests.unit.test_dataset_sparkov import fixture_adapter
from tests.unit.test_run_verification import RUN as SYNTHETIC_RUN
from tests.unit.test_run_verification import TrainedRun, train_fixture_run
from tests.unit.test_train_runs import _benchmark_corpus

SPARKOV_RUN = "sparkov-fixture"

# The columns the methodology expects to be constant on Sparkov.
SPARKOV_CONSTANT_FEATURES = {
    "country_mismatch_customer",
    "country_mismatch_merchant",
    "customer_account_age_days",
    "customer_risk_tier_encoded",
    "is_high_risk_category",
}


# ---------------------------------------------------------------------------
# Fixture runs, trained once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trained:
    root: Path
    synthetic: TrainedRun
    corpus: Path
    adapter: SparkovAdapter

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def cache_root(self) -> Path:
        return self.root / "cache"


def _train_runs(root: Path) -> Trained:
    synthetic = train_fixture_run(root)
    corpus = _benchmark_corpus(root)
    adapter = fixture_adapter(root)

    def stub_tune(*_: object, **__: object) -> TuningResult:
        return TuningResult(
            best_params={"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3},
            best_score=0.5,
            cv_results_summary={"mean_test_score": [0.5], "std_test_score": [0.0]},
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loading, "get_adapter", lambda name: adapter)
        patch.setattr(train, "tune_hyperparameters", stub_tune)
        patch.setattr(train, "RUNS_ROOT", root / "runs")
        train.main(
            [
                "--dataset", "sparkov",
                "--root", str(corpus),
                "--cache-root", str(root / "cache"),
                "--run-name", SPARKOV_RUN,
                "--n-iter", "1",
                "--cv-splits", "2",
                "--target-fpr", "0.05",
            ]
        )
    return Trained(root=root, synthetic=synthetic, corpus=corpus, adapter=adapter)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    return _train_runs(tmp_path_factory.mktemp("transfer"))


# ---------------------------------------------------------------------------
# A private, observed copy per test
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    trained: Trained
    runs_root: Path
    requests: list[DataRequest] = field(default_factory=list)
    adapter_roots: list[Path] = field(default_factory=list)
    scored: list[tuple[Path, np.ndarray]] = field(default_factory=list)
    rows_given_to_xgboost: list[int] = field(default_factory=list)
    other_scoring_calls: list[str] = field(default_factory=list)

    def directory(self, run: str) -> Path:
        return self.runs_root / run

    def location(self) -> DataLocation:
        """Where both fixture datasets are; each run's record picks what applies to it."""
        return DataLocation(
            root=self.trained.corpus,
            cache_root=self.trained.cache_root,
            csv_path=self.trained.synthetic.labels_csv,
        )

    def measure(
        self, source: str = SYNTHETIC_RUN, target: str = SPARKOV_RUN
    ) -> TransferMeasurement:
        return transfer.measure_transfer(
            source,
            target,
            runs_root=self.runs_root,
            source_data=self.location(),
            target_data=self.location(),
        )

    def refused(
        self, match: str, *, source: str = SYNTHETIC_RUN, target: str = SPARKOV_RUN
    ) -> None:
        before = _snapshot(self.runs_root)
        with pytest.raises(RunVerificationError, match=match):
            self.measure(source, target)
        assert _snapshot(self.runs_root) == before, "a refused transfer wrote something"

    def edit(self, run: str, filename: str, change: Callable[[dict[str, Any]], None]) -> None:
        path = self.directory(run) / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        change(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def scored_by(self, run: str) -> list[np.ndarray]:
        return [matrix for directory, matrix in self.scored if directory == self.directory(run)]


@pytest.fixture
def ws(trained: Trained, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    shutil.copytree(trained.runs_root, tmp_path / "runs")
    workspace = Workspace(trained=trained, runs_root=tmp_path / "runs")

    @contextmanager
    def stub_session() -> Iterator[object]:
        yield object()

    monkeypatch.setattr(loading, "SessionLocal", stub_session)
    monkeypatch.setattr(
        loading, "load_dataset_with_csv_labels", lambda *_, **__: trained.synthetic.ds
    )
    monkeypatch.setattr(loading, "get_adapter", lambda name: trained.adapter)

    real_load_detailed = SparkovAdapter.load_detailed

    def load_detailed_spy(self: SparkovAdapter, root: Path, **options: Any) -> Any:
        workspace.adapter_roots.append(Path(root))
        return real_load_detailed(self, root, **options)

    monkeypatch.setattr(SparkovAdapter, "load_detailed", load_detailed_spy)

    real_load_run_data = transfer.load_run_data

    def load_run_data_spy(request: DataRequest, **options: Any) -> Any:
        workspace.requests.append(request)
        return real_load_run_data(request, **options)

    monkeypatch.setattr(transfer, "load_run_data", load_run_data_spy)

    directories: dict[int, Path] = {}
    real_load_explainer = transfer.load_explainer

    def load_explainer_spy(directory: Path | str) -> FraudExplainer:
        explainer = real_load_explainer(directory)
        directories[id(explainer)] = Path(directory)
        return explainer

    monkeypatch.setattr(transfer, "load_explainer", load_explainer_spy)

    real_batch = FraudExplainer.predict_proba_batch

    def batch_spy(self: FraudExplainer, X: np.ndarray) -> np.ndarray:  # noqa: N803
        workspace.scored.append((directories[id(self)], np.array(X)))
        return real_batch(self, X)

    monkeypatch.setattr(FraudExplainer, "predict_proba_batch", batch_spy)
    for name in ("predict_proba", "explain_local", "compute_global_shap"):
        real = getattr(FraudExplainer, name)

        def other_spy(self: FraudExplainer, *args: Any, _name: str = name, _real: Any = real) -> Any:
            workspace.other_scoring_calls.append(_name)
            return _real(self, *args)

        monkeypatch.setattr(FraudExplainer, name, other_spy)

    real_dmatrix = xgb.DMatrix.__init__

    def dmatrix_spy(self: xgb.DMatrix, data: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(data, np.ndarray) and data.ndim == 2:
            workspace.rows_given_to_xgboost.append(int(data.shape[0]))
        real_dmatrix(self, data, *args, **kwargs)

    monkeypatch.setattr(xgb.DMatrix, "__init__", dmatrix_spy)
    return workspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read(directory: Path, name: str) -> Any:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _target_matrix(trained: Trained) -> tuple[LabelledDataset, SplitIndices]:
    """The target run's matrix as training cached it, and its chronological folds."""
    (cache_file,) = trained.cache_root.glob("sparkov_v1_*.npz")
    matrix, _ = load_feature_cache(cache_file)
    return matrix, chronological_split(matrix.timestamps)


def _independent_scores(model_directory: Path, features: np.ndarray) -> np.ndarray:
    """The saved model's scores, through XGBoost directly rather than the explainer."""
    booster = xgb.Booster()
    booster.load_model(str(model_directory / "model.json"))
    return np.asarray(booster.predict(xgb.DMatrix(features)))


def _keys(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        return set(payload) | {key for value in payload.values() for key in _keys(value)}
    if isinstance(payload, list):
        return {key for value in payload for key in _keys(value)}
    return set()


# ---------------------------------------------------------------------------
# Source and target identity
# ---------------------------------------------------------------------------


def test_the_result_names_both_runs_their_datasets_and_featuresets(ws: Workspace) -> None:
    measurement = ws.measure()
    target = ws.directory(SPARKOV_RUN)
    payload = _read(target, TRANSFER_METRICS_FILENAME)
    target_record = _read(target, "run.json")
    source_record = _read(ws.directory(SYNTHETIC_RUN), "run.json")

    assert payload == measurement.payload
    assert payload["transfer_metrics_version"] == "1"
    assert (payload["source_run"], payload["target_run"]) == (SYNTHETIC_RUN, SPARKOV_RUN)
    assert payload["source_dataset"]["name"] == "synthetic"
    assert payload["source_dataset"]["files"] == source_record["dataset"]["files"]
    assert payload["target_dataset"]["name"] == "sparkov"
    assert payload["target_dataset"]["files"] == target_record["dataset"]["files"]
    assert payload["target_dataset"]["subsample"] == target_record["dataset"]["subsample"]
    assert payload["source_featureset"] == payload["target_featureset"] == "v1"
    assert payload["provenance"]["source_code_version"] == source_record["code_version"]
    assert payload["provenance"]["target_code_version"] == target_record["code_version"]
    assert payload["verification"] == {
        "source": {
            "run": SYNTHETIC_RUN,
            "metrics_reproduced": True,
            "model_digest_verified": True,
            "test_fold_identity_verified": True,
        },
        "target": {
            "run": SPARKOV_RUN,
            "metrics_reproduced": True,
            "model_digest_verified": True,
            "test_fold_identity_verified": True,
        },
    }


def test_the_source_model_is_identified_by_the_digest_of_its_saved_file(ws: Workspace) -> None:
    source = ws.directory(SYNTHETIC_RUN)

    payload = ws.measure().payload

    on_disk = hashlib.sha256((source / "model.json").read_bytes()).hexdigest()
    assert payload["source_model_sha256"] == on_disk
    assert _read(source, "training_metadata.json")["model_sha256"] == on_disk


def test_the_target_test_fold_is_identified_by_its_recorded_transaction_ids(
    ws: Workspace,
) -> None:
    """The last fifteen corpus rows, t85 to t99, in the order the target run scored them."""
    measurement = ws.measure()
    matrix, splits = _target_matrix(ws.trained)
    test_ids = [matrix.transaction_ids[row] for row in splits.test]
    recorded = _read(ws.directory(SPARKOV_RUN), "run.json")["test_fold_identity"]

    context = measurement.payload["context"]

    assert test_ids == [transaction_id_for(f"t{index}") for index in range(85, 100)]
    assert context["target_test_transaction_count"] == recorded["transaction_count"] == 15
    assert context["target_test_transaction_ids_sha256"] == transaction_ids_digest(test_ids)
    assert context["target_test_transaction_ids_sha256"] == recorded["transaction_ids_sha256"]
    assert context["target_test_transaction_ids_digest_definition"] == (
        TRANSACTION_IDS_DIGEST_DEFINITION
    )
    assert measurement.target.test_fold_identity_verified is True


# ---------------------------------------------------------------------------
# What is scored
# ---------------------------------------------------------------------------


def test_the_source_model_scores_exactly_the_target_test_rows_with_every_column(
    ws: Workspace,
) -> None:
    """The cached target matrix's test rows, all seventeen v1 columns, constants included."""
    measurement = ws.measure()
    matrix, splits = _target_matrix(ws.trained)
    target_test = matrix.X[splits.test]
    synthetic = ws.trained.synthetic

    by_source = ws.scored_by(SYNTHETIC_RUN)

    assert len(by_source) == 2
    np.testing.assert_array_equal(by_source[0], synthetic.ds.X[synthetic.outcome.splits.test])
    np.testing.assert_array_equal(by_source[1], target_test)
    assert by_source[1].shape == (15, len(FEATURESETS["v1"]))
    assert measurement.target.ds.feature_names == FEATURESETS["v1"]
    np.testing.assert_array_equal(
        measurement.scores, _independent_scores(ws.directory(SYNTHETIC_RUN), target_test)
    )


def test_no_target_train_or_val_row_reaches_either_model(ws: Workspace) -> None:
    """Each model scores its own test fold to verify, and the source model the target's once."""
    ws.measure()
    matrix, splits = _target_matrix(ws.trained)
    synthetic_test_rows = len(ws.trained.synthetic.outcome.splits.test)

    by_target = ws.scored_by(SPARKOV_RUN)

    assert len(by_target) == 1
    np.testing.assert_array_equal(by_target[0], matrix.X[splits.test])
    assert sorted(ws.rows_given_to_xgboost) == sorted([synthetic_test_rows, 15, 15])
    assert sum(ws.rows_given_to_xgboost) == synthetic_test_rows + 2 * len(splits.test)
    assert ws.other_scoring_calls == []


def test_each_run_is_rebuilt_from_its_own_record(ws: Workspace) -> None:
    ws.measure()

    assert [(request.dataset, request.featureset) for request in ws.requests] == [
        ("synthetic", "v1"),
        ("sparkov", "v1"),
    ]
    assert ws.requests[1].root == ws.trained.corpus
    assert ws.requests[1].max_entities is None
    assert ws.requests[1].refresh is False


def test_the_real_sparkov_corpus_is_never_read(ws: Workspace) -> None:
    ws.measure()

    assert ws.adapter_roots, "the target was not loaded through the adapter"
    for root in ws.adapter_roots:
        assert ws.trained.root in root.parents
        assert root != RAW_DATA_DIR / "sparkov"
        assert RAW_DATA_DIR not in root.parents


# ---------------------------------------------------------------------------
# The source threshold
# ---------------------------------------------------------------------------


def test_the_source_threshold_is_applied_unchanged_with_its_provenance(ws: Workspace) -> None:
    payload = ws.measure().payload
    recorded = _read(ws.directory(SYNTHETIC_RUN), "threshold.json")

    source_threshold = payload["source_threshold"]

    assert source_threshold["value"] == recorded["value"]
    assert payload["at_source_threshold"]["threshold"] == recorded["value"]
    assert payload["at_source_threshold"]["threshold_source"] == SOURCE_RUN_THRESHOLD_SOURCE
    assert source_threshold["fpr_ceiling_on_source_val"] == recorded["target_fpr"] == 0.05
    assert source_threshold["realised_fpr_on_source_val"] == recorded["realised_fpr_on_val"]
    assert source_threshold["fallback_used"] is recorded["fallback_used"] is False
    assert payload["context"]["source_threshold_fallback_used"] is False
    assert source_threshold["selection_method"] == SELECTED_THRESHOLD_METHOD
    assert source_threshold["selected_on"] == f"the val fold of source run {SYNTHETIC_RUN!r}"


def test_a_source_threshold_that_fell_back_is_reported_as_the_fallback(ws: Workspace) -> None:
    """The Sparkov fixture's threshold fell back in training; here it is the source."""
    recorded = _read(ws.directory(SPARKOV_RUN), "threshold.json")

    measurement = ws.measure(source=SPARKOV_RUN, target=SYNTHETIC_RUN)
    payload = measurement.payload

    assert recorded["fallback_used"] is True
    assert payload["source_threshold"]["fallback_used"] is True
    assert payload["source_threshold"]["selection_method"] == FALLBACK_THRESHOLD_METHOD
    assert payload["context"]["source_threshold_fallback_used"] is True
    assert payload["at_source_threshold"]["threshold"] == train.FALLBACK_THRESHOLD
    assert measurement.written == ws.directory(SYNTHETIC_RUN) / TRANSFER_METRICS_FILENAME


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def test_threshold_free_results_are_the_existing_metrics_on_the_target_test_fold(
    ws: Workspace,
) -> None:
    payload = ws.measure().payload
    matrix, splits = _target_matrix(ws.trained)
    labels = matrix.y[splits.test]
    scores = _independent_scores(ws.directory(SYNTHETIC_RUN), matrix.X[splits.test])

    free = payload["threshold_free"]

    assert free["pr_auc"] == pr_auc(labels, scores)
    assert free["roc_auc"] == roc_auc(labels, scores)
    assert free["recall_at_1pct_fpr"] == recall_at_fpr(labels, scores, 0.01)[0]
    assert free["recall_at_5pct_fpr"] == recall_at_fpr(labels, scores, 0.05)[0]
    assert free["threshold_source"] == {
        "recall_at_1pct_fpr": TARGET_TEST_ROC_CURVE_SOURCE,
        "recall_at_5pct_fpr": TARGET_TEST_ROC_CURVE_SOURCE,
    }
    assert "point on the target test ROC curve" in free["note"]


def test_results_at_the_source_threshold_are_the_confusion_at_that_threshold(
    ws: Workspace,
) -> None:
    payload = ws.measure().payload
    matrix, splits = _target_matrix(ws.trained)
    labels = matrix.y[splits.test]
    scores = _independent_scores(ws.directory(SYNTHETIC_RUN), matrix.X[splits.test])
    threshold = _read(ws.directory(SYNTHETIC_RUN), "threshold.json")["value"]

    expected = confusion_at_threshold(labels, scores, threshold).as_dict()
    at_threshold = payload["at_source_threshold"]

    assert {key: at_threshold[key] for key in expected} == expected
    assert at_threshold["true_positives"] + at_threshold["false_negatives"] == 3
    assert at_threshold["false_positives"] + at_threshold["true_negatives"] == 12


def test_the_realised_target_fpr_is_measured_at_the_source_threshold(ws: Workspace) -> None:
    payload = ws.measure().payload
    at_threshold = payload["at_source_threshold"]

    legitimate = at_threshold["false_positives"] + at_threshold["true_negatives"]
    expected = at_threshold["false_positives"] / legitimate

    assert at_threshold["realised_fpr_on_target_test"] == expected
    assert payload["context"]["realised_fpr_on_target_test_at_source_threshold"] == expected
    assert "not guaranteed" in at_threshold["note"]


def test_no_result_at_the_source_threshold_is_named_for_a_fixed_fpr(ws: Workspace) -> None:
    """The 1% and 5% name points on the target ROC curve, never the source threshold's FPR."""
    payload = ws.measure().payload
    text = json.dumps(payload)

    fixed_fpr_keys = {key for key in _keys(payload) if "pct_fpr" in key}

    assert fixed_fpr_keys == {"recall_at_1pct_fpr", "recall_at_5pct_fpr"}
    assert not fixed_fpr_keys & _keys(payload["at_source_threshold"])
    assert not fixed_fpr_keys & _keys(payload["context"])
    assert not fixed_fpr_keys & _keys(payload["source_threshold"])
    for wording in ("@1%", "@ 1%", "1% FPR", "target_fpr"):
        assert wording not in text, wording


def test_the_result_is_strict_machine_readable_json(ws: Workspace) -> None:
    written = ws.measure().written
    text = written.read_text(encoding="utf-8")

    for token in ("NaN", "Infinity"):
        assert token not in text
    free = json.loads(text)["threshold_free"]
    assert all(isinstance(free[key], float) for key in ("pr_auc", "roc_auc"))


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_the_context_labels_source_and_target_counts_apart(ws: Workspace) -> None:
    payload = ws.measure().payload
    source_counts = _read(ws.directory(SYNTHETIC_RUN), "metrics.json")["context"]["fraud_counts"]
    target_metrics = _read(ws.directory(SPARKOV_RUN), "metrics.json")["context"]
    target_record = _read(ws.directory(SPARKOV_RUN), "run.json")

    context = payload["context"]

    assert set(context) == {
        "target_test_rows",
        "target_test_prevalence",
        "target_test_fraud_count",
        "source_train_fraud_count",
        "source_val_fraud_count",
        "realised_fpr_on_target_test_at_source_threshold",
        "source_threshold_fallback_used",
        "label_delay_note",
        "target_fold_periods",
        "target_test_transaction_count",
        "target_test_transaction_ids_sha256",
        "target_test_transaction_ids_digest_definition",
    }
    assert context["source_train_fraud_count"] == source_counts["train"]
    assert context["source_val_fraud_count"] == source_counts["val"]
    assert context["target_test_fraud_count"] == target_metrics["fraud_counts"]["test"] == 3
    assert context["target_test_rows"] == 15
    assert context["target_test_prevalence"] == target_metrics["test_prevalence"] == 3 / 15
    assert context["target_fold_periods"] == target_record["splits"]
    assert context["label_delay_note"] == LABEL_DELAY_NOTE
    target_counts = target_metrics["fraud_counts"]
    assert (source_counts["train"], source_counts["val"]) != (
        target_counts["train"],
        target_counts["val"],
    ), "the fixture no longer tells source counts from target counts"


def test_the_feature_context_describes_every_scored_column(ws: Workspace) -> None:
    payload = ws.measure().payload
    matrix, splits = _target_matrix(ws.trained)
    target_test = matrix.X[splits.test]
    synthetic = ws.trained.synthetic
    source_training = synthetic.ds.X[synthetic.outcome.splits.train]

    features = payload["features"]

    assert features["featureset_version"] == "v1"
    assert features["scored_columns"] == FEATURESETS["v1"]
    assert [entry["feature"] for entry in features["by_feature"]] == FEATURESETS["v1"]
    assert set(features["constant_in_target_test_fold"]) >= SPARKOV_CONSTANT_FEATURES
    assert "no column is dropped or remapped" in features["note"]
    for column, entry in enumerate(features["by_feature"]):
        values = target_test[:, column]
        low, high = source_training[:, column].min(), source_training[:, column].max()
        assert entry["constant_in_target_test_fold"] == (np.unique(values).size == 1)
        assert entry["constant_in_source_training_fold"] == (
            np.unique(source_training[:, column]).size == 1
        )
        assert (entry["source_training_fold_min"], entry["source_training_fold_max"]) == (low, high)
        assert (entry["target_test_fold_min"], entry["target_test_fold_max"]) == (
            values.min(),
            values.max(),
        )
        assert entry["target_test_rows_outside_source_training_range"] == int(
            ((values < low) | (values > high)).sum()
        )


# ---------------------------------------------------------------------------
# Where the result goes
# ---------------------------------------------------------------------------


def test_only_the_transfer_result_is_written_and_only_into_the_target_run(
    ws: Workspace,
) -> None:
    before = _snapshot(ws.runs_root)

    ws.measure()

    after = _snapshot(ws.runs_root)
    changed = {path for path in after if before.get(path) != after[path]}
    assert changed == {str(ws.directory(SPARKOV_RUN) / TRANSFER_METRICS_FILENAME)}
    assert set(before) <= set(after)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_the_same_run_as_source_and_target_is_refused(ws: Workspace) -> None:
    with pytest.raises(TransferError, match="both the source and the target"):
        ws.measure(SPARKOV_RUN, SPARKOV_RUN)

    assert ws.requests == []


def test_two_runs_on_the_same_dataset_are_refused_before_loading_data(ws: Workspace) -> None:
    shutil.copytree(ws.directory(SYNTHETIC_RUN), ws.directory("synthetic-copy"))
    ws.edit("synthetic-copy", "run.json", lambda record: record.update(run_name="synthetic-copy"))

    ws.refused("both trained on dataset 'synthetic'", target="synthetic-copy")
    assert ws.requests == []


def test_mismatched_featuresets_are_refused_before_loading_data(ws: Workspace) -> None:
    ws.edit(SPARKOV_RUN, "run.json", lambda record: record.update(featureset_version="v2"))

    ws.refused("uses featureset 'v1', but target run 'sparkov-fixture' uses 'v2'")
    assert ws.requests == []


@pytest.mark.parametrize("missing", ["source", "target"])
def test_a_missing_run_is_refused(ws: Workspace, missing: str) -> None:
    runs = {"source": SYNTHETIC_RUN, "target": SPARKOV_RUN, missing: "no-such-run"}

    ws.refused("No run named 'no-such-run'", source=runs["source"], target=runs["target"])
    assert ws.requests == []


def test_an_incomplete_target_run_is_refused(ws: Workspace) -> None:
    (ws.directory(SPARKOV_RUN) / "metrics.json").unlink()

    ws.refused(r"incomplete: metrics\.json missing")


def test_a_wrong_source_model_of_the_right_shape_is_refused_before_loading_data(
    ws: Workspace,
) -> None:
    """Same rounds, same seventeen inputs, no early-stopping state: only the digest tells."""
    synthetic = ws.trained.synthetic
    train_rows = synthetic.outcome.splits.train
    impostor = xgb.XGBClassifier(
        n_estimators=synthetic.outcome.fit.best_iteration + 1, max_depth=2, random_state=0
    )
    impostor.fit(synthetic.ds.X[train_rows], synthetic.ds.y[train_rows])
    impostor.get_booster().save_model(str(ws.directory(SYNTHETIC_RUN) / "model.json"))

    ws.refused("It is not the model file this run saved")
    assert ws.requests == []
    assert ws.scored == []


def test_another_runs_model_as_the_source_model_is_refused_before_loading_data(
    ws: Workspace,
) -> None:
    shutil.copyfile(
        ws.directory(SPARKOV_RUN) / "model.json", ws.directory(SYNTHETIC_RUN) / "model.json"
    )

    ws.refused(r"Run 'synthetic-fixture': model\.json")
    assert ws.requests == []
    assert ws.scored == []


def test_a_source_run_without_a_model_digest_is_refused_before_loading_data(
    ws: Workspace,
) -> None:
    ws.edit(SYNTHETIC_RUN, "training_metadata.json", lambda metadata: metadata.pop("model_sha256"))

    ws.refused("records no digest of its model.json")
    assert ws.requests == []


def test_a_source_run_that_does_not_verify_is_refused(ws: Workspace) -> None:
    def nudge(metrics: dict[str, Any]) -> None:
        metrics["test_pr_auc"] = float(np.nextafter(metrics["test_pr_auc"], 2.0))

    ws.edit(SYNTHETIC_RUN, "metrics.json", nudge)

    ws.refused("do not reproduce metrics.json exactly")
    assert ws.scored_by(SPARKOV_RUN) == []


def test_target_test_ids_other_than_the_recorded_ones_are_refused(ws: Workspace) -> None:
    other = transaction_ids_digest(["tx-that-was-never-scored"])
    ws.edit(
        SPARKOV_RUN,
        "run.json",
        lambda record: record["test_fold_identity"].update(transaction_ids_sha256=other),
    )

    ws.refused("not the rows, or not the order, the run was evaluated on")
    # The source model scored its own test fold to verify, and never the target's.
    assert len(ws.scored_by(SYNTHETIC_RUN)) == 1


def test_a_target_test_count_other_than_the_recorded_one_is_refused(ws: Workspace) -> None:
    ws.edit(
        SPARKOV_RUN,
        "run.json",
        lambda record: record["test_fold_identity"].update(transaction_count=16),
    )

    ws.refused(r"holds 15 transactions, but run\.json identifies 16")


def test_a_target_run_that_does_not_identify_its_test_fold_is_refused(ws: Workspace) -> None:
    """A version 2 record still verifies on its own, but cannot be a transfer target."""

    def as_version_2(record: dict[str, Any]) -> None:
        record.pop("test_fold_identity")
        record["metadata_version"] = "2"

    ws.edit(SPARKOV_RUN, "run.json", as_version_2)

    ws.refused(r"does not identify its test fold's transactions \(run\.json version 2\)")
    assert ws.requests == []


def test_a_source_threshold_without_a_fallback_record_is_refused(ws: Workspace) -> None:
    ws.edit(SYNTHETIC_RUN, "threshold.json", lambda threshold: threshold.pop("fallback_used"))

    ws.refused("does not record whether its threshold fell back")
    assert ws.requests == []


def test_a_row_limit_for_a_registered_target_is_refused(ws: Workspace) -> None:
    before = _snapshot(ws.runs_root)

    with pytest.raises(RunVerificationError, match="cannot be reloaded"):
        transfer.measure_transfer(
            SYNTHETIC_RUN,
            SPARKOV_RUN,
            runs_root=ws.runs_root,
            source_data=ws.location(),
            target_data=DataLocation(
                root=ws.trained.corpus, cache_root=ws.trained.cache_root, limit=10
            ),
        )

    assert _snapshot(ws.runs_root) == before


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _cli_arguments(ws: Workspace, source: str = SYNTHETIC_RUN) -> list[str]:
    return [
        "--source-run", source,
        "--target-run", SPARKOV_RUN,
        "--cache-root", str(ws.trained.cache_root),
        "--source-csv-path", str(ws.trained.synthetic.labels_csv),
        "--target-root", str(ws.trained.corpus),
    ]


def test_the_cli_measures_a_transfer_into_the_target_run(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transfer, "RUNS_ROOT", ws.runs_root)

    transfer.main(_cli_arguments(ws))

    payload = _read(ws.directory(SPARKOV_RUN), TRANSFER_METRICS_FILENAME)
    assert (payload["source_run"], payload["target_run"]) == (SYNTHETIC_RUN, SPARKOV_RUN)


def test_the_cli_exits_with_an_error_when_a_run_does_not_verify(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(transfer, "RUNS_ROOT", ws.runs_root)
    shutil.copyfile(
        ws.directory(SPARKOV_RUN) / "model.json", ws.directory(SYNTHETIC_RUN) / "model.json"
    )

    with pytest.raises(SystemExit) as refused:
        transfer.main(_cli_arguments(ws))

    assert refused.value.code == 1
    assert "was not measured" in caplog.text
    assert not (ws.directory(SPARKOV_RUN) / TRANSFER_METRICS_FILENAME).exists()


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--source-run", "a", "--target-run", "a"], id="same-run"),
        pytest.param(["--source-run", "../escape", "--target-run", "a"], id="unsafe-source"),
        pytest.param(["--source-run", "a", "--target-run", "UPPER"], id="unsafe-target"),
        pytest.param(["--target-run", "a"], id="no-source"),
    ],
)
def test_the_cli_refuses_arguments_before_any_work(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(transfer, "measure_transfer", lambda *a, **k: calls.append(a))

    with pytest.raises(SystemExit) as refused:
        transfer.main(argv)

    assert refused.value.code == 2
    assert calls == []
