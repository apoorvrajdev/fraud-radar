"""Checking that a named run is the run it claims to be, before analysing it.

Analysis describes a run's test fold: how well its scores are calibrated,
which features drove them. That description is about the run only if it
scores the same rows with the same model the run was evaluated with. A
mutable database, a different subsample, another run's model file or a model
saved with extra boosting rounds would each produce a confident description
of something else.

So every check compares what the run recorded with what is loaded now, and
any mismatch refuses the run with the reason. Nothing is tolerated. The saved
model scores exactly as training did, so a run whose `metrics.json` cannot be
reproduced exactly from its saved model and its test fold is not the run it
claims to be.

Only the test fold is scored. The timestamps and labels of every fold are
read, to check fold periods, sizes and fraud counts, but no train or val row
is ever given to the model.

A run record of version 3 also identifies its test fold's transactions, and
the loaded fold must hold exactly those, in the same order. Older records
carry no identity; they still verify, and `VerifiedRun` says the identity was
not checked rather than implying it was.

The same holds for the saved model: a fit record that carries the SHA-256 of
`model.json` must match the file byte for byte, which is checked on the
record alone, before any data is loaded.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from app.fraud.explainer import FraudExplainer
from app.fraud.feature_spec import FEATURESETS, feature_names
from ml.artifacts import (
    MODEL_FILENAME,
    ThresholdRecord,
    collect_library_versions,
    load_feature_list,
    load_threshold,
)
from ml.data import LabelledDataset
from ml.datasets.base import DatasetContractError
from ml.datasets.manifest import sha256_file
from ml.holdout import evaluate_test_fold
from ml.loading import RunData
from ml.paths import RUNS_ROOT, run_dir
from ml.runs import (
    RUN_METADATA_FILENAME,
    RunMetadata,
    SplitPeriod,
    identify_test_fold,
    split_periods,
)
from ml.splits import SplitIndices, assert_no_temporal_leakage, chronological_split

THRESHOLD_FILENAME = "threshold.json"
FEATURE_LIST_FILENAME = "feature_list.json"
METRICS_FILENAME = "metrics.json"
TRAINING_METADATA_FILENAME = "training_metadata.json"

RECORDED_RUN_FILES = (
    RUN_METADATA_FILENAME,
    MODEL_FILENAME,
    THRESHOLD_FILENAME,
    FEATURE_LIST_FILENAME,
    METRICS_FILENAME,
    TRAINING_METADATA_FILENAME,
)

# The libraries that compute the scores and the recorded metrics. A run is
# reproduced exactly only with the versions it was recorded with.
REPRODUCTION_LIBRARIES = ("xgboost", "scikit-learn", "numpy")

# metrics.json entries that are not test-fold results, so not recomputed: the
# cross-validated search score comes from the training fold, and the synthetic
# targets are goals rather than measurements.
_SEARCH_SCORE = "best_cv_pr_auc"
_TARGET_PREFIX = "target_"

_FIT_FIELDS = ("dataset_size", "train_size", "val_size", "test_size", "best_iteration")


class RunVerificationError(Exception):
    """A run cannot be analysed as the run it claims to be."""


@dataclass(frozen=True)
class RecordedRun:
    """What a run directory recorded, read back without loading any data."""

    name: str
    directory: Path
    record: RunMetadata
    training_metadata: Mapping[str, Any]
    threshold: ThresholdRecord
    metrics: Mapping[str, Any]
    feature_list: list[str]

    @property
    def model_sha256(self) -> str | None:
        """The digest the fit record holds for `model.json`, or None if it holds none."""
        digest = self.training_metadata.get("model_sha256")
        return None if digest is None else str(digest)


@dataclass(frozen=True)
class VerifiedRun:
    """A run whose data, folds, model and metrics all match what it recorded.

    `test_scores` are the saved model's scores for `splits.test`, in that
    order. They reproduce the run's `metrics.json` exactly.
    """

    recorded: RecordedRun
    ds: LabelledDataset
    splits: SplitIndices
    test_scores: np.ndarray

    @property
    def test_fold_identity_verified(self) -> bool:
        """Whether the test fold's transaction ids were checked against the run record.

        `verify_run` checks them whenever the record identifies the fold, so
        this is False only for a record written before identities were
        recorded, whose test fold is then known by period, size and fraud
        count alone.
        """
        return self.recorded.record.test_fold_identity is not None

    @property
    def model_digest_verified(self) -> bool:
        """Whether `model.json` was checked against the digest in the fit record.

        Checked whenever the record carries one, so this is False only for a
        fit record written before model digests were.
        """
        return self.recorded.model_sha256 is not None


def read_recorded_run(run_name: str, *, runs_root: Path = RUNS_ROOT) -> RecordedRun:
    """Read a run directory's records, refusing one that is missing any of them."""
    directory = run_dir(run_name, runs_root=runs_root)
    if not directory.is_dir():
        raise RunVerificationError(f"No run named {run_name!r}: {directory} does not exist.")
    missing = [name for name in RECORDED_RUN_FILES if not (directory / name).exists()]
    if RUN_METADATA_FILENAME in missing:
        raise RunVerificationError(
            f"Run {run_name!r} has no {RUN_METADATA_FILENAME}, so what it was trained on "
            "is unknown. Only a run written by `python -m ml.train --run-name` can be analysed."
        )
    if missing:
        raise RunVerificationError(
            f"Run {run_name!r} is incomplete: {', '.join(missing)} missing from {directory}."
        )

    try:
        record = RunMetadata.from_dict(_read_json(directory / RUN_METADATA_FILENAME))
        return RecordedRun(
            name=run_name,
            directory=directory,
            record=record,
            training_metadata=_read_json(directory / TRAINING_METADATA_FILENAME),
            threshold=load_threshold(directory),
            metrics=_read_json(directory / METRICS_FILENAME),
            feature_list=load_feature_list(directory),
        )
    except (KeyError, TypeError, ValueError, DatasetContractError) as exc:
        raise RunVerificationError(f"Run {run_name!r} has a malformed record: {exc}") from exc


def check_recorded_run(recorded: RecordedRun) -> None:
    """Checks that need no data: name, environment, featureset, fit record and model file."""
    _check_run_name(recorded)
    _check_environment(recorded)
    _check_featureset(recorded)
    _check_fit_record(recorded)
    _check_metrics_record(recorded)
    _check_model_file(recorded)
    _check_model_digest(recorded)


def verify_run(recorded: RecordedRun, data: RunData, explainer: FraudExplainer) -> VerifiedRun:
    """Verify the loaded data and model against the run, scoring its test fold only.

    `data` must be loaded from the request the run record describes, and
    `explainer` from the run's own directory.
    """
    check_recorded_run(recorded)
    ds = data.ds
    _check_provenance(recorded, data)
    _check_loaded_features(recorded, ds, explainer)
    splits = _check_folds(recorded, ds)
    _check_test_fold_identity(recorded, ds, splits)

    test_scores = explainer.predict_proba_batch(ds.X[splits.test])
    _check_metrics_reproduced(recorded, ds, splits, test_scores)
    return VerifiedRun(recorded=recorded, ds=ds, splits=splits, test_scores=test_scores)


# ---------------------------------------------------------------------------
# Checks shared with a verifier outside the featureset registry
# ---------------------------------------------------------------------------
#
# A run on a featureset the registry does not hold, such as the ULB track's,
# is verified by that track's own verifier, which checks the featureset
# against its own definition. The checks below do not depend on the
# featureset, so it uses them as they are. `check_recorded_run` and its
# registry guard are unchanged.


def check_record_apart_from_featureset(recorded: RecordedRun) -> None:
    """Every record-only check but the featureset's: name, environment, fit record and model."""
    _check_run_name(recorded)
    _check_environment(recorded)
    _check_fit_record(recorded)
    _check_metrics_record(recorded)
    _check_model_file(recorded)
    _check_model_digest(recorded)


def check_loaded_folds(recorded: RecordedRun, data: RunData) -> SplitIndices:
    """Refuse unless the loaded data has the run's provenance, folds and test-fold identity."""
    _check_provenance(recorded, data)
    splits = _check_folds(recorded, data.ds)
    _check_test_fold_identity(recorded, data.ds, splits)
    return splits


def check_metrics_reproduced(
    recorded: RecordedRun,
    ds: LabelledDataset,
    splits: SplitIndices,
    test_scores: np.ndarray,
) -> None:
    """Refuse unless `test_scores` reproduce the run's `metrics.json` exactly."""
    _check_metrics_reproduced(recorded, ds, splits, test_scores)


# ---------------------------------------------------------------------------
# Checks on the record alone
# ---------------------------------------------------------------------------


def _check_run_name(recorded: RecordedRun) -> None:
    if recorded.record.run_name != recorded.name:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {RUN_METADATA_FILENAME} names run "
            f"{recorded.record.run_name!r}, so it is the record of another run."
        )


def _check_environment(recorded: RecordedRun) -> None:
    current = collect_library_versions()
    for library in REPRODUCTION_LIBRARIES:
        version = recorded.record.library_versions.get(library)
        if version is None:
            raise RunVerificationError(
                f"Run {recorded.name!r} records no {library} version, so whether this "
                "environment can reproduce its metrics exactly cannot be established."
            )
        if version != current[library]:
            raise RunVerificationError(
                f"Run {recorded.name!r} was recorded with {library} {version}, but this "
                f"environment has {library} {current[library]}. Its metrics.json is only "
                "reproduced exactly with the versions it was recorded with; analyse it in an "
                "environment that has them."
            )


def _check_featureset(recorded: RecordedRun) -> None:
    version = recorded.record.featureset_version
    if version not in FEATURESETS:
        raise RunVerificationError(
            f"Run {recorded.name!r} records featureset {version!r}, which is not registered "
            f"({', '.join(sorted(FEATURESETS))})."
        )
    if recorded.feature_list != feature_names(version):
        raise RunVerificationError(
            f"Run {recorded.name!r}: {FEATURE_LIST_FILENAME} does not list featureset "
            f"{version!r}'s features in their registered order."
        )


def _check_fit_record(recorded: RecordedRun) -> None:
    absent = [name for name in _FIT_FIELDS if name not in recorded.training_metadata]
    if absent:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {TRAINING_METADATA_FILENAME} has no "
            f"{', '.join(absent)}. It was written before the fit record included them, so "
            "the saved model cannot be checked against the fit; retrain the run."
        )


def _check_metrics_record(recorded: RecordedRun) -> None:
    context = recorded.metrics.get("context")
    if not isinstance(context, Mapping) or not isinstance(context.get("fraud_counts"), Mapping):
        raise RunVerificationError(
            f"Run {recorded.name!r}: {METRICS_FILENAME} has no context with fold fraud "
            "counts. It was written before results recorded their context; retrain the run."
        )
    if "threshold_source" not in recorded.metrics:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {METRICS_FILENAME} does not record where its thresholds "
            "came from. It was written before results were labelled; retrain the run."
        )
    at_threshold = recorded.metrics.get("at_operating_threshold")
    measured_at = at_threshold.get("threshold") if isinstance(at_threshold, Mapping) else None
    if measured_at != recorded.threshold.value:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {METRICS_FILENAME} was measured at threshold "
            f"{measured_at}, but {THRESHOLD_FILENAME} holds {recorded.threshold.value}."
        )


def _check_model_file(recorded: RecordedRun) -> None:
    booster = xgb.Booster()
    booster.load_model(str(recorded.directory / MODEL_FILENAME))
    best_iteration = int(recorded.training_metadata["best_iteration"])
    rounds = booster.num_boosted_rounds()
    if rounds != best_iteration + 1:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {MODEL_FILENAME} holds {rounds} boosting rounds, but the "
            f"fit's best round was {best_iteration}, so the model training evaluated has "
            f"{best_iteration + 1}. A model with any other rounds does not score as training did."
        )
    if booster.attr("best_iteration") is not None:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {MODEL_FILENAME} carries early-stopping state, so it was "
            "saved before only the rounds the model predicts with were kept; retrain the run."
        )
    if booster.num_features() != len(recorded.feature_list):
        raise RunVerificationError(
            f"Run {recorded.name!r}: {MODEL_FILENAME} takes {booster.num_features()} features, "
            f"but {FEATURE_LIST_FILENAME} lists {len(recorded.feature_list)}."
        )


def _check_model_digest(recorded: RecordedRun) -> None:
    expected = recorded.model_sha256
    if expected is None:
        # A fit record written before model digests were.
        return
    actual = sha256_file(recorded.directory / MODEL_FILENAME)
    if actual != expected:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {MODEL_FILENAME} hashes to {actual}, but "
            f"{TRAINING_METADATA_FILENAME} records {expected}. It is not the model file this "
            "run saved."
        )


# ---------------------------------------------------------------------------
# Checks against the loaded data
# ---------------------------------------------------------------------------


def _check_provenance(recorded: RecordedRun, data: RunData) -> None:
    if data.provenance is None:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the data was loaded without a provenance record, so it "
            "cannot be matched to run.json."
        )
    # Compared field by field, so datetimes compare as instants. retrieved_at is
    # when the files were read, not which bytes they hold.
    expected = recorded.record.dataset
    loaded = data.provenance
    differing = [
        field.name
        for field in fields(expected)
        if field.name != "retrieved_at"
        and getattr(expected, field.name) != getattr(loaded, field.name)
    ]
    if differing:
        details = "; ".join(
            f"{name}: recorded {getattr(expected, name)!r}, loaded {getattr(loaded, name)!r}"
            for name in differing
        )
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded dataset is not the one run.json records "
            f"({details})."
        )


def _check_loaded_features(
    recorded: RecordedRun, ds: LabelledDataset, explainer: FraudExplainer
) -> None:
    if ds.feature_names != recorded.feature_list:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded matrix has features {ds.feature_names}, but "
            f"the run was trained on {recorded.feature_list}."
        )
    if explainer.feature_names != recorded.feature_list:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the explainer was not loaded from this run's model."
        )
    if explainer.threshold != recorded.threshold.value:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the explainer holds threshold {explainer.threshold}, but "
            f"{THRESHOLD_FILENAME} holds {recorded.threshold.value}; it was not loaded from "
            "this run."
        )


def _check_folds(recorded: RecordedRun, ds: LabelledDataset) -> SplitIndices:
    metadata = recorded.training_metadata
    if ds.n_rows != metadata["dataset_size"]:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded dataset has {ds.n_rows} rows, but the run was "
            f"trained on {metadata['dataset_size']}."
        )

    splits = chronological_split(ds.timestamps)
    try:
        assert_no_temporal_leakage(ds.timestamps, splits)
        periods = split_periods(ds.timestamps, splits)
    except (AssertionError, DatasetContractError) as exc:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded folds have no valid chronological periods: {exc}"
        ) from exc

    if not recorded.record.splits:
        raise RunVerificationError(
            f"Run {recorded.name!r}: run.json records no fold periods, so the folds cannot be "
            "matched to the ones the run was evaluated on."
        )
    if periods != recorded.record.splits:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded folds cover {_describe(periods)}, but run.json "
            f"records {_describe(recorded.record.splits)}."
        )

    sizes = splits.sizes
    recorded_sizes = (metadata["train_size"], metadata["val_size"], metadata["test_size"])
    if sizes != recorded_sizes:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded folds hold {sizes} rows (train, val, test), but "
            f"the run recorded {recorded_sizes}."
        )

    recorded_counts = recorded.metrics["context"]["fraud_counts"]
    for name, indices in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        count = int(ds.y[indices].sum())
        if count != recorded_counts.get(name):
            raise RunVerificationError(
                f"Run {recorded.name!r}: the loaded {name} fold holds {count} frauds, but "
                f"metrics.json records {recorded_counts.get(name)}."
            )
    return splits


def _check_test_fold_identity(
    recorded: RecordedRun, ds: LabelledDataset, splits: SplitIndices
) -> None:
    identity = recorded.record.test_fold_identity
    if identity is None:
        # A record written before identities were; RunMetadata refuses a later one without it.
        return
    loaded = identify_test_fold(ds.transaction_ids, splits)
    if loaded.transaction_count != identity.transaction_count:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded test fold holds {loaded.transaction_count} "
            f"transactions, but {RUN_METADATA_FILENAME} identifies "
            f"{identity.transaction_count}."
        )
    if loaded.transaction_ids_sha256 != identity.transaction_ids_sha256:
        raise RunVerificationError(
            f"Run {recorded.name!r}: the loaded test fold's transaction ids, in fold order, "
            f"digest to {loaded.transaction_ids_sha256}, but {RUN_METADATA_FILENAME} records "
            f"{identity.transaction_ids_sha256}. These are not the rows, or not the order, "
            "the run was evaluated on."
        )


def _check_metrics_reproduced(
    recorded: RecordedRun,
    ds: LabelledDataset,
    splits: SplitIndices,
    test_scores: np.ndarray,
) -> None:
    evaluation = evaluate_test_fold(
        train_labels=ds.y[splits.train],
        val_labels=ds.y[splits.val],
        test_labels=ds.y[splits.test],
        test_scores=test_scores,
        operating_threshold=recorded.threshold.value,
    )
    # Compared as JSON, the form the recorded values were written in.
    recomputed: dict[str, Any] = json.loads(json.dumps(evaluation.metrics()))
    written = dict(recorded.metrics)

    missing = sorted(recomputed.keys() - written.keys())
    if missing:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {METRICS_FILENAME} lacks {', '.join(missing)}; "
            "retrain the run."
        )
    unverifiable = sorted(
        key
        for key in written.keys() - recomputed.keys()
        if key != _SEARCH_SCORE and not key.startswith(_TARGET_PREFIX)
    )
    if unverifiable:
        raise RunVerificationError(
            f"Run {recorded.name!r}: {METRICS_FILENAME} holds entries this analysis cannot "
            f"verify: {', '.join(unverifiable)}."
        )
    differing = sorted(key for key in recomputed if written[key] != recomputed[key])
    if differing:
        details = "; ".join(
            f"{key}: recorded {written[key]!r}, reproduced {recomputed[key]!r}" for key in differing
        )
        raise RunVerificationError(
            f"Run {recorded.name!r}: its saved model and test fold do not reproduce "
            f"{METRICS_FILENAME} exactly ({details})."
        )


def _describe(periods: tuple[SplitPeriod, ...]) -> str:
    return ", ".join(
        f"{period.name} {period.start.isoformat()} → {period.end.isoformat()}"
        for period in periods
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not hold a JSON object.")
    return payload
