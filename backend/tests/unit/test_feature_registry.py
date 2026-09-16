"""Phase 5A — the featureset registry is a contract, not a convenience.

A model artifact records the featureset version it was trained against. If
the registry's `v1` list ever changes order or membership, every artifact
trained before that change becomes silently wrong: XGBoost is positional, so
column 3 meaning something new produces predictions, not errors.

The literal pin below is therefore duplicated on purpose. It is not DRY and
it is not meant to be: editing `feature_spec.py` alone must fail CI.
"""
from __future__ import annotations

import pytest

from app.fraud.feature_spec import (
    DEFAULT_FEATURESET,
    FEATURE_NAMES,
    FEATURESETS,
    N_FEATURES,
    feature_names,
)

# The production featureset, written out by hand. Do not generate this list.
V1_PINNED: list[str] = [
    "log_amount",
    "hour_of_day",
    "is_weekend",
    "is_off_hours",
    "is_card_present",
    "country_mismatch_customer",
    "country_mismatch_merchant",
    "tx_count_1h",
    "tx_count_24h",
    "log_amount_sum_24h",
    "customer_account_age_days",
    "customer_risk_tier_encoded",
    "avg_amount_30d",
    "amount_zscore_30d",
    "days_since_last_tx",
    "merchant_risk_encoded",
    "is_high_risk_category",
]


def test_v1_matches_the_pinned_order_exactly() -> None:
    assert feature_names("v1") == V1_PINNED


def test_v1_has_seventeen_features() -> None:
    assert len(V1_PINNED) == 17
    assert N_FEATURES == 17


def test_default_featureset_is_v1() -> None:
    assert DEFAULT_FEATURESET == "v1"


def test_feature_names_defaults_to_the_production_featureset() -> None:
    assert feature_names() == feature_names(DEFAULT_FEATURESET)


def test_legacy_alias_still_points_at_the_default_featureset() -> None:
    """`FEATURE_NAMES` predates the registry and is imported all over the app."""
    assert FEATURESETS[DEFAULT_FEATURESET] == FEATURE_NAMES
    assert FEATURE_NAMES == V1_PINNED


def test_unknown_version_raises_with_the_known_versions_listed() -> None:
    with pytest.raises(ValueError, match="Unknown featureset version 'v99'"):
        feature_names("v99")
    with pytest.raises(ValueError, match="v1"):
        feature_names("v99")


def test_caller_cannot_mutate_the_registry_through_the_accessor() -> None:
    names = feature_names("v1")
    names.append("injected_feature")
    assert feature_names("v1") == V1_PINNED


def test_every_registered_featureset_is_ordered_and_unique() -> None:
    for version, names in FEATURESETS.items():
        assert names, f"featureset {version} is empty"
        assert len(set(names)) == len(names), f"featureset {version} has duplicates"
