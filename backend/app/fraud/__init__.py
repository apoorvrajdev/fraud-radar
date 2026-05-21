"""Fraud detection package: features, scoring, explainability."""
from app.fraud.explainer import (
    FraudExplainer,
    LocalExplanation,
    get_explainer,
    initialize_explainer,
    top_contributors,
)
from app.fraud.feature_spec import FEATURE_NAMES
from app.fraud.features import FeatureExtractor, extract_features

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractor",
    "FraudExplainer",
    "LocalExplanation",
    "extract_features",
    "get_explainer",
    "initialize_explainer",
    "top_contributors",
]
