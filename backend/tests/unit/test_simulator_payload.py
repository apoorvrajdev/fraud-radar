"""Unit tests for the Phase 3D simulator payload generators.

Pure-function tests: no DB session, no HTTP client, no mocks. Every test
seeds Python's `random` module so payload generation is deterministic.
"""
from __future__ import annotations

import random
from decimal import Decimal

import pytest

from app.schemas.transaction import TransactionCreate
from app.simulator.main import (
    _FRAUD_PATTERNS,
    _HIGH_RISK_COUNTRIES,
    _MIXABLE_CURRENCIES,
    _build_payload,
    _generate_clean_payload,
    _generate_fraud_payload,
    parse_currency_mix,
)

# Sample IDs matching the 36-char Pydantic constraint on the schema.
_CUSTOMER_IDS = [
    "10000000-0000-4000-8000-00000000000a",
    "10000000-0000-4000-8000-00000000000b",
    "10000000-0000-4000-8000-00000000000c",
]
_MERCHANT_IDS = [
    "20000000-0000-4000-8000-00000000000a",
    "20000000-0000-4000-8000-00000000000b",
]


def test_clean_payload_passes_pydantic_validation() -> None:
    """The clean generator must produce a body the TransactionCreate
    schema accepts — otherwise every clean POST would 422 in real
    production use."""
    random.seed(42)
    payload = _generate_clean_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0])
    parsed = TransactionCreate.model_validate(payload)
    assert parsed.customer_id == _CUSTOMER_IDS[0]
    assert parsed.merchant_id == _MERCHANT_IDS[0]


def test_clean_payload_amount_stays_under_5000() -> None:
    """The amount_ceiling rule fires at $5,000. A 'clean' payload that
    crosses that line is a generator bug — it'd mark a clean transaction
    as REVIEW in the log and confuse the operator."""
    random.seed(42)
    for _ in range(100):
        payload = _generate_clean_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0])
        assert Decimal(payload["amount"]) < Decimal("5000.00"), payload


def test_clean_payload_country_is_not_high_risk() -> None:
    """The high_risk_country rule fires when the country is in
    {RU, CN, NG, RO, VE, ID} and the amount is over $500. The clean
    generator must avoid every member of that set to keep its label honest."""
    random.seed(42)
    for _ in range(100):
        payload = _generate_clean_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0])
        assert payload["country"] not in _HIGH_RISK_COUNTRIES, payload


@pytest.mark.parametrize("pattern", list(_FRAUD_PATTERNS))
def test_fraud_payload_passes_pydantic_validation_for_each_pattern(
    pattern: str,
) -> None:
    """Every fraud pattern must also produce a schema-valid body —
    otherwise that pattern would never reach the scoring service."""
    random.seed(42)
    payload = _generate_fraud_payload(_CUSTOMER_IDS[0], _MERCHANT_IDS[0], pattern)
    TransactionCreate.model_validate(payload)  # raises on failure


def test_build_payload_respects_fraud_rate_over_large_sample() -> None:
    """At fraud_rate=0.30, ~30% of 1000 samples should be non-clean.

    Loose binomial bound: σ ≈ 14.5 → 95% CI is ~[271, 329]. The asserted
    range [200, 400] gives ~7σ slack so the test never flakes.
    """
    random.seed(42)
    non_clean = 0
    for _ in range(1000):
        _payload, label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.30,
        )
        if label != "clean":
            non_clean += 1
    assert 200 <= non_clean <= 400, non_clean


def test_build_payload_uses_pool_ids_only() -> None:
    """The simulator must never reference an ID outside the pool it
    loaded at startup — every request would otherwise FK-violate."""
    random.seed(42)
    customer_pool = set(_CUSTOMER_IDS)
    merchant_pool = set(_MERCHANT_IDS)
    for _ in range(50):
        payload, _label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.25,
        )
        assert payload["customer_id"] in customer_pool
        assert payload["merchant_id"] in merchant_pool


# ---------------------------------------------------------------------------
# Phase 5G — currency mix
# ---------------------------------------------------------------------------


def test_parse_currency_mix_normalises_weights() -> None:
    """Weights need not sum to 1 — "USD:8,EUR:2" means 80/20."""
    mix = parse_currency_mix("USD:8,EUR:2")

    assert mix == {"USD": 0.8, "EUR": 0.2}


def test_parse_currency_mix_accepts_fractions_and_whitespace() -> None:
    mix = parse_currency_mix(" usd:0.85 , eur:0.10 , gbp:0.05 ")

    assert set(mix) == {"USD", "EUR", "GBP"}
    assert sum(mix.values()) == pytest.approx(1.0)


def test_parse_currency_mix_sums_a_repeated_code() -> None:
    mix = parse_currency_mix("USD:0.5,USD:0.5")

    assert mix == {"USD": 1.0}


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param("USD", id="no-weight"),
        pytest.param("USD:abc", id="non-numeric-weight"),
        pytest.param("USD:0", id="zero-weight"),
        pytest.param("USD:-1", id="negative-weight"),
        pytest.param("", id="empty"),
        pytest.param(",,", id="only-separators"),
    ],
)
def test_parse_currency_mix_rejects_malformed_input(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_currency_mix(spec)


def test_parse_currency_mix_rejects_an_unsupported_currency() -> None:
    """A typo must stop the run, not quietly emit unasked-for traffic.

    JPY is rejected deliberately, not by oversight: the rules engine's
    thresholds are denominated in the raw amount, so a currency two
    orders of magnitude from USD would change which rules fire.
    """
    with pytest.raises(ValueError, match="not available"):
        parse_currency_mix("USD:0.9,JPY:0.1")


def test_every_mixable_currency_is_a_valid_iso_code() -> None:
    """The backend's FX path refuses anything that is not three letters."""
    for code in _MIXABLE_CURRENCIES:
        assert len(code) == 3
        assert code.isalpha()
        assert code.isupper()


def test_build_payload_defaults_to_usd_without_a_mix() -> None:
    """Backward compatibility: the simulator's existing behaviour."""
    random.seed(42)
    for _ in range(50):
        payload, _label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.3,
        )
        assert payload["currency"] == "USD"


def test_build_payload_emits_the_requested_currencies() -> None:
    random.seed(42)
    mix = parse_currency_mix("USD:0.6,EUR:0.4")

    seen = set()
    for _ in range(200):
        payload, _label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.1, currency_mix=mix,
        )
        seen.add(payload["currency"])

    assert seen == {"USD", "EUR"}


def test_currency_mix_proportions_hold_over_many_samples() -> None:
    """20% EUR over 2000 draws: σ ≈ 18, so [300, 500] is ~5σ of slack."""
    random.seed(7)
    mix = parse_currency_mix("USD:0.8,EUR:0.2")

    eur = 0
    for _ in range(2000):
        payload, _label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.0, currency_mix=mix,
        )
        if payload["currency"] == "EUR":
            eur += 1

    assert 300 <= eur <= 500, eur


def test_currency_mix_changes_only_the_currency() -> None:
    """The fraud signal a pattern encodes must survive the mix.

    Amount, country and card-present are what each pattern uses to
    trigger a specific rule; if the mix perturbed them, non-USD traffic
    would quietly carry different fraud behaviour than USD traffic.
    """
    random.seed(99)
    baseline, baseline_label = _build_payload(
        _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.5,
    )
    random.seed(99)
    mixed, mixed_label = _build_payload(
        _CUSTOMER_IDS, _MERCHANT_IDS,
        fraud_rate=0.5,
        currency_mix={"EUR": 1.0},
    )

    assert mixed_label == baseline_label
    assert mixed["currency"] == "EUR"
    assert baseline["currency"] == "USD"
    for field in ("amount", "country", "is_card_present", "customer_id"):
        assert mixed[field] == baseline[field]


def test_mixed_currency_payloads_still_validate() -> None:
    """Non-USD payloads must satisfy the ingestion schema unchanged."""
    random.seed(11)
    mix = parse_currency_mix("USD:0.5,EUR:0.3,GBP:0.2")

    for _ in range(50):
        payload, _label = _build_payload(
            _CUSTOMER_IDS, _MERCHANT_IDS, fraud_rate=0.3, currency_mix=mix,
        )
        model = TransactionCreate(**payload)
        assert model.amount > Decimal("0")
        assert len(model.currency) == 3
