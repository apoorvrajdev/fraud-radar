"""Synthetic customer generator using Faker."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from faker import Faker


@dataclass
class CustomerRecord:
    """In-memory customer record before persistence."""

    id: str
    email: str
    full_name: str
    country: str
    risk_tier: str
    account_age_days: int


# Country distribution: heavy US weighting reflects realistic fintech traffic
COUNTRY_WEIGHTS: dict[str, float] = {
    "US": 0.55,
    "GB": 0.10,
    "CA": 0.08,
    "DE": 0.06,
    "FR": 0.05,
    "AU": 0.04,
    "IN": 0.04,
    "BR": 0.03,
    "JP": 0.03,
    "MX": 0.02,
}

# Risk tier distribution: most customers are low-risk
RISK_TIER_WEIGHTS: dict[str, float] = {
    "LOW": 0.80,
    "MEDIUM": 0.15,
    "HIGH": 0.05,
}


def generate_customers(
    n: int = 500,
    *,
    seed: int = 42,
) -> list[CustomerRecord]:
    """Generate n synthetic customers with realistic distributions.

    Customer attributes drive downstream fraud features (geo mismatches,
    risk-tier thresholds, account-age signals).
    """
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    countries = list(COUNTRY_WEIGHTS.keys())
    country_probs = list(COUNTRY_WEIGHTS.values())
    risk_tiers = list(RISK_TIER_WEIGHTS.keys())
    risk_probs = list(RISK_TIER_WEIGHTS.values())

    customers: list[CustomerRecord] = []
    for _ in range(n):
        country = rng.choice(countries, p=country_probs)
        risk_tier = rng.choice(risk_tiers, p=risk_probs)
        # Account age: most customers are 6mo to 3y old, some new, some very old
        account_age_days = int(rng.gamma(shape=2.0, scale=180.0))
        account_age_days = max(1, min(account_age_days, 365 * 5))

        customers.append(
            CustomerRecord(
                id=str(uuid.uuid4()),
                email=faker.unique.email(),
                full_name=faker.name(),
                country=str(country),
                risk_tier=str(risk_tier),
                account_age_days=account_age_days,
            )
        )
    return customers
