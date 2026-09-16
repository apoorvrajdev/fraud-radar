"""Canonical feature names, ordering, and the featureset registry.

XGBoost is positional — column order at training time must match the order
at inference time. This module is the single source of truth.

Featuresets are versioned so a model artifact can name the feature contract
it was trained against. `v1` is the production featureset and is frozen: a
model in `ml/artifacts/` was trained on exactly these 17 columns in exactly
this order. A future `v2` is added as a new entry, never by editing `v1`.
"""
from __future__ import annotations

# Feature ordering: keep stable across training and inference. Adding new
# features means appending here AND retraining the model.
_V1_FEATURE_NAMES: list[str] = [
    # Transaction-level
    "log_amount",
    "hour_of_day",
    "is_weekend",
    "is_off_hours",
    "is_card_present",
    # Geographic
    "country_mismatch_customer",
    "country_mismatch_merchant",
    # Velocity (recent activity)
    "tx_count_1h",
    "tx_count_24h",
    "log_amount_sum_24h",
    # Customer history
    "customer_account_age_days",
    "customer_risk_tier_encoded",
    "avg_amount_30d",
    "amount_zscore_30d",
    "days_since_last_tx",
    # Merchant context
    "merchant_risk_encoded",
    "is_high_risk_category",
]

# Registry of every featureset version this codebase knows how to build.
# Keyed by version string; the value is the canonical column order.
FEATURESETS: dict[str, list[str]] = {
    "v1": _V1_FEATURE_NAMES,
}

# The featureset the production extractor and the served model use.
DEFAULT_FEATURESET: str = "v1"


def feature_names(version: str = DEFAULT_FEATURESET) -> list[str]:
    """Return the ordered feature names for `version`.

    Returns a copy so a caller cannot mutate the registry by accident —
    a reordered list would silently mismatch the trained model's columns.
    """
    try:
        names = FEATURESETS[version]
    except KeyError:
        known = ", ".join(sorted(FEATURESETS))
        raise ValueError(
            f"Unknown featureset version {version!r}. Known versions: {known}."
        ) from None
    return list(names)


# Backward-compatible alias: the production featureset's column order.
FEATURE_NAMES: list[str] = FEATURESETS[DEFAULT_FEATURESET]

# Number of features (used for shape validation)
N_FEATURES: int = len(FEATURE_NAMES)
