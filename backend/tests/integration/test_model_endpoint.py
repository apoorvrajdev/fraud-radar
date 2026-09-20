"""Phase 5G — GET /api/v1/model.

The endpoint's job is not just to report the serving model but to keep
the production/benchmark boundary legible, so most of these tests are
about what it must *not* say. A badge that implied the ULB numbers
described the served model would undo the separation the whole of
Phase 5 was built to maintain.

Artifacts are written into a tmp directory per test, so nothing here
depends on a trained model being on disk.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_info as model_info_service

METRICS = {
    "test_pr_auc": 0.9326733114784815,
    "test_roc_auc": 0.9988999139378583,
    "recall_at_1pct_fpr": 0.978494623655914,
}
METADATA = {
    "trained_at_utc": "2026-05-21T12:47:25+00:00",
    "test_size": 7502,
    "test_fraud_rate": 0.012396694214876033,
}
THRESHOLD = {"value": 0.7431091070175171, "target_fpr": 0.01}
FEATURES = {"features": [f"f{i}" for i in range(17)]}


def _write_artifacts(root: Path, **overrides: Any) -> Path:
    files: dict[str, Any] = {
        "metrics.json": METRICS,
        "training_metadata.json": METADATA,
        "threshold.json": THRESHOLD,
        "feature_list.json": FEATURES,
    }
    files.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        if payload is None:
            continue  # explicitly absent
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No lifespan: this endpoint reads files, never the loaded booster.
    yield TestClient(app)
    app.dependency_overrides.clear()


# The artifact-reading tests call the service directly with a tmp
# directory; the endpoint tests assert on the parts of the envelope
# that hold whether or not a model has been trained in this checkout.


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_endpoint_returns_the_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/model")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"serving", "benchmarks", "reporting_currency"}


def test_serving_block_names_the_featureset_and_feature_count(
    client: TestClient,
) -> None:
    body = client.get("/api/v1/model").json()["serving"]

    assert body["featureset_version"] == "v1"
    assert body["feature_count"] == 17


def test_reporting_currency_matches_the_fx_setting(client: TestClient) -> None:
    """The dashboard labels its aggregates from this — it must not drift."""
    from app.config import get_settings

    body = client.get("/api/v1/model").json()

    assert body["reporting_currency"] == get_settings().fx_base_currency


# ---------------------------------------------------------------------------
# The production / benchmark boundary
# ---------------------------------------------------------------------------


def test_exactly_one_track_is_served(client: TestClient) -> None:
    benchmarks = client.get("/api/v1/model").json()["benchmarks"]

    served = [t for t in benchmarks if t["served"]]
    assert len(served) == 1
    assert served[0]["name"] == "synthetic_v1"


def test_ulb_is_present_and_explicitly_not_served(client: TestClient) -> None:
    """The load-bearing assertion of this file.

    ULB is real card data on an isolated track. A badge that let it
    read as the production model would be the single most misleading
    thing this dashboard could say.
    """
    benchmarks = client.get("/api/v1/model").json()["benchmarks"]

    ulb = next(t for t in benchmarks if t["name"] == "ulb_pca_v1")
    assert ulb["served"] is False
    assert ulb["kind"] == "real-anonymised"


def test_sparkov_is_labelled_as_simulated_and_not_served(
    client: TestClient,
) -> None:
    benchmarks = client.get("/api/v1/model").json()["benchmarks"]

    sparkov = next(t for t in benchmarks if t["name"] == "sparkov_v1")
    assert sparkov["served"] is False
    assert sparkov["kind"] == "synthetic-external"
    assert "not real card transactions" in sparkov["description"].lower()


def test_the_served_model_is_never_described_as_trained_on_real_data(
    client: TestClient,
) -> None:
    serving = client.get("/api/v1/model").json()["serving"]

    assert serving["dataset_kind"] == "synthetic"
    assert serving["dataset_kind"] != "real-anonymised"


def test_every_benchmark_names_the_card_holding_its_results(
    client: TestClient,
) -> None:
    """A number on a badge should be one click from its full context."""
    benchmarks = client.get("/api/v1/model").json()["benchmarks"]

    assert benchmarks
    for track in benchmarks:
        assert track["card_path"], f"{track['name']} has no card path"
        assert track["card_path"].endswith(".md")


def test_metrics_carry_a_caveat_about_what_they_measure(
    client: TestClient,
) -> None:
    """No bare accuracy claim: the caveat ships with the numbers."""
    serving = client.get("/api/v1/model").json()["serving"]

    caveat = serving["metrics_caveat"].lower()
    assert "generator" in caveat
    assert "not how detectable real fraud is" in caveat


# ---------------------------------------------------------------------------
# Reading the artifacts, and surviving their absence
# ---------------------------------------------------------------------------


def test_values_are_read_from_the_artifact_files(tmp_path: Path) -> None:
    root = _write_artifacts(tmp_path / "a")

    info = model_info_service.get_model_info(root)

    assert info.serving.trained_at_utc == "2026-05-21T12:47:25+00:00"
    assert info.serving.threshold == pytest.approx(0.7431091070175171)
    assert info.serving.metrics is not None
    assert info.serving.metrics.pr_auc == pytest.approx(0.9326733114784815)
    assert info.serving.metrics.test_size == 7502


def test_a_recorded_featureset_version_wins_over_the_registry_default(
    tmp_path: Path,
) -> None:
    root = _write_artifacts(
        tmp_path / "a",
        **{"feature_list.json": {"featureset_version": "v9", "features": ["a", "b"]}},
    )

    info = model_info_service.get_model_info(root)

    assert info.serving.featureset_version == "v9"
    assert info.serving.feature_count == 2


def test_missing_artifacts_degrade_rather_than_raise(tmp_path: Path) -> None:
    """An un-trained checkout should show "unknown", not a 500."""
    empty = tmp_path / "empty"
    empty.mkdir()

    info = model_info_service.get_model_info(empty)

    assert info.serving.trained_at_utc is None
    assert info.serving.threshold is None
    assert info.serving.metrics is None
    # The parts that do not depend on artifacts are still correct.
    assert info.serving.featureset_version == "v1"
    assert info.serving.feature_count == 17
    assert len(info.benchmarks) == 3


def test_malformed_metrics_drop_the_block_without_failing(
    tmp_path: Path,
) -> None:
    root = _write_artifacts(
        tmp_path / "a", **{"metrics.json": {"unexpected": "shape"}}
    )

    info = model_info_service.get_model_info(root)

    assert info.serving.metrics is None
    assert info.serving.trained_at_utc is not None


def test_unparseable_json_is_treated_as_absent(tmp_path: Path) -> None:
    root = _write_artifacts(tmp_path / "a")
    (root / "threshold.json").write_text("{not json", encoding="utf-8")

    info = model_info_service.get_model_info(root)

    assert info.serving.threshold is None
