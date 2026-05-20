"""Synthetic merchant generator using Faker."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from faker import Faker


@dataclass
class MerchantRecord:
    """In-memory merchant record before persistence."""

    id: str
    name: str
    category: str
    mcc: str
    country: str
    risk_rating: str


# Merchant categories with realistic MCC codes (subset of ISO 18245)
# https://en.wikipedia.org/wiki/Merchant_category_code
CATEGORIES: dict[str, dict[str, str | float]] = {
    "GROCERY":         {"mcc": "5411", "weight": 0.20, "risk": "LOW"},
    "RESTAURANT":      {"mcc": "5812", "weight": 0.18, "risk": "LOW"},
    "RETAIL":          {"mcc": "5311", "weight": 0.15, "risk": "LOW"},
    "GAS_STATION":     {"mcc": "5541", "weight": 0.10, "risk": "LOW"},
    "ELECTRONICS":     {"mcc": "5732", "weight": 0.08, "risk": "MEDIUM"},
    "TRAVEL":          {"mcc": "4511", "weight": 0.07, "risk": "MEDIUM"},
    "ENTERTAINMENT":   {"mcc": "7832", "weight": 0.06, "risk": "MEDIUM"},
    "ONLINE_SERVICE":  {"mcc": "5968", "weight": 0.06, "risk": "MEDIUM"},
    "JEWELRY":         {"mcc": "5944", "weight": 0.03, "risk": "HIGH"},
    "MONEY_TRANSFER":  {"mcc": "4829", "weight": 0.04, "risk": "HIGH"},
    "GAMBLING":        {"mcc": "7995", "weight": 0.02, "risk": "HIGH"},
    "CRYPTO":          {"mcc": "6051", "weight": 0.01, "risk": "HIGH"},
}


def generate_merchants(
    n: int = 200,
    *,
    seed: int = 42,
) -> list[MerchantRecord]:
    """Generate n synthetic merchants distributed across realistic categories."""
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    categories = list(CATEGORIES.keys())
    category_probs = [float(CATEGORIES[c]["weight"]) for c in categories]
    countries = ["US", "GB", "CA", "DE", "FR", "AU", "IN", "BR", "JP", "MX"]

    merchants: list[MerchantRecord] = []
    for _ in range(n):
        category = str(rng.choice(categories, p=category_probs))
        category_info = CATEGORIES[category]

        merchants.append(
            MerchantRecord(
                id=str(uuid.uuid4()),
                name=faker.company(),
                category=category,
                mcc=str(category_info["mcc"]),
                country=str(rng.choice(countries)),
                risk_rating=str(category_info["risk"]),
            )
        )
    return merchants
