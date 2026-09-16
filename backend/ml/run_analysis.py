"""Analysing a named run's test fold, into that run's own directory.

The Phase 5D methodology measures two things about each run beyond its
metrics, both on the test fold: calibration, with no calibrator fitted, and
global mean-absolute SHAP importance. This module produces them, and country
segments where the test fold holds more than one country group.

Nothing is analysed until the run has been verified: its data is rebuilt from
its `run.json`, and its saved model has to reproduce its `metrics.json`
exactly on the re-derived test fold. The scores and rows analysed are then
the verified ones, so every output describes the run it is written beside.

Outputs go into the run's directory only. The served artifacts and
`ml/MODEL_CARD.md` belong to the legacy analysis of the promoted model, and
nothing here writes them.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.fraud.explainer import FraudExplainer, load_explainer
from ml.analysis import (
    BUCKET_ORDER,
    bucket_for_country,
    compute_calibration_metrics,
    compute_feature_importance,
    compute_segment_metrics,
    render_bar_plot,
    render_beeswarm_plot,
    render_calibration_plot,
)
from ml.loading import DEFAULT_SYNTHETIC_CSV, DataRequest, load_run_data
from ml.paths import FEATURE_CACHE_DIR, RUNS_ROOT
from ml.reporting import SEGMENT_TEST_ROC_CURVE_SOURCE
from ml.run_verification import (
    RunVerificationError,
    VerifiedRun,
    check_recorded_run,
    read_recorded_run,
    verify_run,
)

CALIBRATION_FILENAME = "calibration_metrics.json"
FEATURE_IMPORTANCE_FILENAME = "feature_importance.json"
SEGMENT_FILENAME = "segment_metrics.json"
CALIBRATION_PLOT_FILENAME = "calibration_curve.png"
SHAP_BEESWARM_FILENAME = "global_shap_beeswarm.png"
SHAP_BAR_FILENAME = "global_shap_bar.png"

# Replaces the calibration payload's generic note, which interprets results
# that a benchmark run has not measured yet.
RUN_CALIBRATION_NOTE = "Measured on the test fold; no calibrator fitted."

# TreeExplainer attributes a binary:logistic model's margin, not its probability.
SHAP_UNITS = "log-odds"

SEGMENT_NOTE = (
    "Country segments measure performance stability across geographic groups, "
    "not demographic fairness."
)


@dataclass(frozen=True)
class RunAnalysis:
    """What analysing one verified run produced.

    `shap_values` is the test fold's SHAP matrix, in `verified.splits.test`
    row order; only its ranking is written.
    """

    verified: VerifiedRun
    calibration: dict[str, Any]
    feature_importance: dict[str, Any]
    segments: dict[str, Any]
    shap_values: np.ndarray
    written: tuple[Path, ...]


def analyze_run(
    run_name: str,
    *,
    runs_root: Path = RUNS_ROOT,
    root: Path | None = None,
    cache_root: Path = FEATURE_CACHE_DIR,
    csv_path: Path = DEFAULT_SYNTHETIC_CSV,
    limit: int | None = None,
    full_corpus: bool = False,
) -> RunAnalysis:
    """Verify `run_name`, then analyse its test fold into its run directory.

    `root`, `cache_root` and `full_corpus` locate a registered dataset;
    `csv_path` and `limit` the synthetic one. Everything else is read from the
    run's own records. Raises `RunVerificationError` before writing anything
    if the run does not verify.
    """
    recorded = read_recorded_run(run_name, runs_root=runs_root)
    # Refuse on the record alone before loading what may be a whole corpus.
    check_recorded_run(recorded)
    try:
        request = DataRequest.from_run_record(
            recorded.record,
            root=root,
            cache_root=cache_root,
            csv_path=csv_path,
            limit=limit,
            full_corpus=full_corpus,
        )
    except ValueError as exc:
        raise RunVerificationError(f"Run {run_name!r} cannot be reloaded: {exc}") from exc

    data = load_run_data(request, with_provenance=True, with_countries=True)
    explainer = load_explainer(recorded.directory)
    verified = verify_run(recorded, data, explainer)
    if data.countries is None:  # pragma: no cover - requested above
        raise RunVerificationError(f"Run {run_name!r} was loaded without countries.")
    return _analyze_verified(verified, explainer, data.countries)


def describe_calibration(test_labels: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    """Calibration of the test fold's scores, measured and never fitted."""
    payload = compute_calibration_metrics(test_labels, test_scores)
    payload["note"] = RUN_CALIBRATION_NOTE
    return payload


def describe_feature_importance(
    shap_values: np.ndarray, feature_names: Sequence[str]
) -> dict[str, Any]:
    """The mean-absolute SHAP ranking, with the units the values are in."""
    payload = compute_feature_importance(shap_values, list(feature_names))
    payload["units"] = SHAP_UNITS
    return payload


def describe_segments(
    test_labels: np.ndarray, test_scores: np.ndarray, countries: Sequence[str]
) -> dict[str, Any]:
    """Country segments of the test fold, or why there are none.

    Decided from the rows: when every test row falls in one country group, a
    breakdown would only repeat the overall result, so none is reported.
    """
    counts = Counter(bucket_for_country(country) for country in countries)
    groups = {name: counts[name] for name in BUCKET_ORDER if counts[name]}
    if len(groups) < 2:
        only = ", ".join(groups) or "no"
        return {
            "reported": False,
            "reason": (
                f"Every test-fold row falls in the {only} country group, so a segment "
                "breakdown would only repeat the overall result."
            ),
            "country_groups": groups,
        }

    payload = compute_segment_metrics(test_labels, test_scores, countries)
    payload["notes"] = SEGMENT_NOTE
    payload["reported"] = True
    payload["country_groups"] = groups
    # Each block's recall, the global one included, is read at a threshold
    # found on that block's own test rows.
    payload["threshold_source"] = {"recall_at_1pct_fpr": SEGMENT_TEST_ROC_CURVE_SOURCE}
    return payload


def _analyze_verified(
    verified: VerifiedRun, explainer: FraudExplainer, countries: Sequence[str]
) -> RunAnalysis:
    test = verified.splits.test
    test_features = verified.ds.X[test]
    test_labels = verified.ds.y[test]
    test_scores = verified.test_scores
    feature_names = verified.ds.feature_names

    calibration = describe_calibration(test_labels, test_scores)
    shap_values = explainer.compute_global_shap(test_features)
    feature_importance = describe_feature_importance(shap_values, feature_names)
    segments = describe_segments(test_labels, test_scores, [countries[row] for row in test])

    directory = verified.recorded.directory
    written = (
        _write_json(directory / CALIBRATION_FILENAME, calibration),
        _write_json(directory / FEATURE_IMPORTANCE_FILENAME, feature_importance),
        _write_json(directory / SEGMENT_FILENAME, segments),
    )
    plots = (
        directory / CALIBRATION_PLOT_FILENAME,
        directory / SHAP_BEESWARM_FILENAME,
        directory / SHAP_BAR_FILENAME,
    )
    render_calibration_plot(calibration, plots[0])
    render_beeswarm_plot(shap_values, test_features, feature_names, plots[1])
    render_bar_plot(shap_values, test_features, feature_names, plots[2])

    return RunAnalysis(
        verified=verified,
        calibration=calibration,
        feature_importance=feature_importance,
        segments=segments,
        shap_values=shap_values,
        written=written + plots,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
