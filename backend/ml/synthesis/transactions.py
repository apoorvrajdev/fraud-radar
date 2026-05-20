"""Generator for the legitimate (non-fraud) transaction base."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
from faker import Faker

from ml.synthesis.customers import CustomerRecord
from ml.synthesis.merchants import MerchantRecord


@dataclass
class TransactionRecord:
    """In-memory transaction record before persistence."""

    id: str
    idempotency_key: str
    customer_id: str
    merchant_id: str
    amount: Decimal
    currency: str
    status: str
    payment_method: str
    card_last4: str | None
    ip_address: str | None
    device_id: str | None
    country: str
    is_card_present: bool
    is_fraud: bool  # ground-truth label, used for training
    fraud_pattern: str | None  # which injection pattern, if any
    created_at: datetime


def _sample_amount(rng: np.random.Generator, category: str) -> Decimal:
    """Sample a transaction amount with category-aware log-normal distribution."""
    # Different categories have different typical amounts
    category_means: dict[str, float] = {
        "GROCERY": 50.0,
        "RESTAURANT": 35.0,
        "RETAIL": 80.0,
        "GAS_STATION": 45.0,
        "ELECTRONICS": 250.0,
        "TRAVEL": 400.0,
        "ENTERTAINMENT": 60.0,
        "ONLINE_SERVICE": 25.0,
        "JEWELRY": 600.0,
        "MONEY_TRANSFER": 300.0,
        "GAMBLING": 100.0,
        "CRYPTO": 500.0,
    }
    mean = category_means.get(category, 50.0)
    # Log-normal with mu chosen so median is roughly the category mean
    mu = float(np.log(mean))
    sigma = 0.8
    raw = float(rng.lognormal(mean=mu, sigma=sigma))
    # Clip to a reasonable range; round to cents
    bounded = max(1.0, min(raw, 10_000.0))
    return Decimal(f"{bounded:.2f}")


def _sample_hour(rng: np.random.Generator) -> int:
    """Sample hour-of-day with realistic bimodal distribution (lunch + evening)."""
    # 60% from a normal centered at 13:00, 40% from one centered at 19:00
    if rng.random() < 0.6:
        h = int(rng.normal(loc=13, scale=2.5))
    else:
        h = int(rng.normal(loc=19, scale=2.0))
    return max(0, min(23, h))


def generate_legitimate_transactions(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n: int,
    *,
    seed: int = 42,
    days_back: int = 30,
) -> list[TransactionRecord]:
    """Generate n legitimate (non-fraud) transactions over the past `days_back` days.

    Each customer gets transactions roughly proportional to a power-law
    activity distribution. Merchants are sampled with category-weighted draws.
    """
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    # Activity skew: a few power users do many transactions
    activity_weights = rng.dirichlet(alpha=np.ones(len(customers)) * 0.5)

    # Build a lookup for merchant info
    now = datetime.utcnow()
    earliest = now - timedelta(days=days_back)

    transactions: list[TransactionRecord] = []
    for _ in range(n):
        customer = customers[int(rng.choice(len(customers), p=activity_weights))]
        merchant = merchants[int(rng.choice(len(merchants)))]

        # Time: uniformly distributed over the window, then nudge hour-of-day
        days_offset = float(rng.uniform(0, days_back))
        ts = earliest + timedelta(days=days_offset)
        hour = _sample_hour(rng)
        ts = ts.replace(hour=hour, minute=int(rng.integers(0, 60)))

        amount = _sample_amount(rng, merchant.category)

        # Country: 90% match customer home, 10% travel
        if rng.random() < 0.9:
            country = customer.country
        else:
            country = merchant.country

        is_card_present = bool(rng.random() < 0.6)
        payment_method = str(
            rng.choice(["CARD", "WALLET", "ACH"], p=[0.85, 0.12, 0.03])
        )

        transactions.append(
            TransactionRecord(
                id=str(uuid.uuid4()),
                idempotency_key=str(uuid.uuid4()),
                customer_id=customer.id,
                merchant_id=merchant.id,
                amount=amount,
                currency="USD",
                status="APPROVED",
                payment_method=payment_method,
                card_last4=f"{int(rng.integers(1000, 9999))}",
                ip_address=faker.ipv4_public(),
                device_id=str(uuid.uuid4())[:16],
                country=country,
                is_card_present=is_card_present,
                is_fraud=False,
                fraud_pattern=None,
                created_at=ts,
            )
        )

    return transactions
