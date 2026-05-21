"""Trained-model loading and SHAP explainer.

The explainer is constructed once at process startup (via the FastAPI
lifespan hook) and cached as a module-level singleton. Loading SHAP's
TreeExplainer is the expensive part — doing it once amortises the cost
across every inference request.

The inference path never imports from `backend/ml/` — that is training
code only. All artifacts written by the trainer are consumed here as
opaque data files.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import shap
import xgboost as xgb

from app.fraud.feature_spec import FEATURE_NAMES, N_FEATURES

log = logging.getLogger(__name__)

Decision = Literal["APPROVE", "REVIEW", "DECLINE"]

_MODEL_FILE = "model.json"
_THRESHOLD_FILE = "threshold.json"
_FEATURE_LIST_FILE = "feature_list.json"


@dataclass(frozen=True)
class LocalExplanation:
    """Result of explaining one transaction.

    - fraud_score: P(fraud=1) from the booster
    - shap_values: signed contribution per feature, in FEATURE_NAMES order
    - base_value: explainer.expected_value (mean raw model output)
    """

    fraud_score: float
    shap_values: np.ndarray
    base_value: float


class FraudExplainer:
    """Trained XGBoost booster + a cached SHAP TreeExplainer.

    Constructed once at startup. All methods are read-only after init.
    """

    def __init__(
        self,
        booster: xgb.Booster,
        explainer: shap.TreeExplainer,
        threshold: float,
        feature_names: list[str],
    ) -> None:
        if len(feature_names) != N_FEATURES:
            raise ValueError(
                f"feature_list.json has {len(feature_names)} features but "
                f"FEATURE_NAMES declares {N_FEATURES}. "
                "Retraining is required when the feature schema changes."
            )
        if feature_names != FEATURE_NAMES:
            raise ValueError(
                "feature_list.json order does not match FEATURE_NAMES. "
                "XGBoost is positional — refusing to serve with a drifted schema."
            )
        self._booster = booster
        self._explainer = explainer
        self._threshold = float(threshold)
        self._feature_names = list(feature_names)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def base_value(self) -> float:
        """Mean raw model output (explainer.expected_value)."""
        ev = self._explainer.expected_value
        # SHAP returns scalar for binary classifiers in recent versions, but
        # older callers see a length-1 array. Coerce defensively.
        if hasattr(ev, "__len__"):
            return float(np.asarray(ev).item())
        return float(ev)

    def predict_proba(self, x_row: np.ndarray) -> float:
        """Score a single feature vector. Returns P(fraud=1)."""
        x_row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        dmat = xgb.DMatrix(x_row, feature_names=self._feature_names)
        return float(self._booster.predict(dmat)[0])

    def explain_local(self, x_row: np.ndarray) -> LocalExplanation:
        """Compute SHAP values for one feature vector.

        The vector must be in FEATURE_NAMES order — the caller is
        responsible for that. We re-assert the shape defensively.
        """
        x_row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        if x_row.shape[1] != N_FEATURES:
            raise ValueError(
                f"Feature vector has {x_row.shape[1]} columns; expected {N_FEATURES}"
            )
        shap_values = np.asarray(self._explainer.shap_values(x_row))
        # TreeExplainer returns (1, n_features) for a single row. Flatten.
        shap_values = shap_values.reshape(-1)
        fraud_score = self.predict_proba(x_row)
        return LocalExplanation(
            fraud_score=fraud_score,
            shap_values=shap_values,
            base_value=self.base_value,
        )

    def classify(self, fraud_score: float) -> Decision:
        """Map a score to APPROVE / REVIEW / DECLINE.

        Boundaries are placeholders: APPROVE under half the operating
        threshold, DECLINE at-or-above the threshold, REVIEW in between.
        Business policy will tune these; the shape stays the same.
        """
        if fraud_score >= self._threshold:
            return "DECLINE"
        if fraud_score >= self._threshold * 0.5:
            return "REVIEW"
        return "APPROVE"

    def compute_global_shap(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for a batch of feature vectors.

        Reserved for the model card (Phase 2I) — not used by the inference
        endpoint. Plain numpy in, plain numpy out, shape (n_rows, n_features).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected shape (n_rows, {N_FEATURES}); got {X.shape}"
            )
        return np.asarray(self._explainer.shap_values(X))


def top_contributors(
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_values: np.ndarray,
    *,
    k: int = 5,
) -> list[dict[str, object]]:
    """Return the top-k features by absolute SHAP value, sorted descending.

    Pure function — pulled out of `FraudExplainer` so it is trivial to unit-test
    without instantiating an explainer.
    """
    if len(feature_names) != len(shap_values):
        raise ValueError(
            f"Length mismatch: {len(feature_names)} names vs {len(shap_values)} shap values"
        )
    order = np.argsort(-np.abs(shap_values))
    rows: list[dict[str, object]] = []
    for idx in order[:k]:
        s = float(shap_values[idx])
        rows.append({
            "feature": feature_names[idx],
            "shap_value": s,
            "feature_value": float(feature_values[idx]),
            "direction": "fraud" if s > 0 else "legit",
        })
    return rows


# ---------------------------------------------------------------------------
# Module-level singleton — populated by `initialize_explainer()` at startup.
# ---------------------------------------------------------------------------

_singleton: FraudExplainer | None = None


def initialize_explainer(artifacts_dir: Path | str) -> FraudExplainer:
    """Load model + threshold + feature list and construct the singleton.

    Idempotent: calling twice returns the same instance. The lifespan hook
    in app.main is the one caller in production; tests can call it with a
    fixture artifact directory.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    artifacts_dir = Path(artifacts_dir)
    model_path = artifacts_dir / _MODEL_FILE
    threshold_path = artifacts_dir / _THRESHOLD_FILE
    feature_list_path = artifacts_dir / _FEATURE_LIST_FILE

    missing = [p for p in (model_path, threshold_path, feature_list_path) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing model artifacts in {artifacts_dir}: {names}. "
            f"Run `uv run python -m ml.train` from the backend/ directory first."
        )

    log.info("Loading booster from %s", model_path)
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with threshold_path.open(encoding="utf-8") as f:
        threshold_payload = json.load(f)
    threshold = float(threshold_payload["value"])

    with feature_list_path.open(encoding="utf-8") as f:
        feature_payload = json.load(f)
    feature_names = list(feature_payload["features"])

    log.info("Constructing TreeExplainer (one-time)...")
    explainer = shap.TreeExplainer(booster)

    _singleton = FraudExplainer(
        booster=booster,
        explainer=explainer,
        threshold=threshold,
        feature_names=feature_names,
    )
    log.info("FraudExplainer ready — threshold=%.4f, n_features=%d",
             threshold, len(feature_names))
    return _singleton


def get_explainer() -> FraudExplainer:
    """FastAPI dependency: return the loaded explainer or 503 the request.

    Importers should depend on this rather than touching `_singleton` directly.
    """
    if _singleton is None:
        raise RuntimeError(
            "Explainer not initialized — startup hook did not run. "
            "Check that app.main calls initialize_explainer in lifespan."
        )
    return _singleton


def reset_explainer_for_tests() -> None:
    """Test helper — clears the cached singleton so each test starts fresh."""
    global _singleton
    _singleton = None
