"""Fraud pattern injectors.

Each function takes the existing transaction list and customer/merchant pools,
generates additional fraudulent transactions following a specific real-world
pattern, and returns them. The patterns are documented inline because they
form the training signal the ML model learns.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
from faker import Faker

from ml.synthesis.customers import CustomerRecord
from ml.synthesis.merchants import MerchantRecord
from ml.synthesis.transactions import TransactionRecord


def _make_fraud_tx(
    customer: CustomerRecord,
    merchant: MerchantRecord,
    amount: Decimal,
    ts: datetime,
    pattern: str,
    *,
    country: str | None = None,
    card_last4: str | None = None,
    ip: str | None = None,
    device_id: str | None = None,
    is_card_present: bool = False,
) -> TransactionRecord:
    """Helper: build a fraudulent transaction with consistent defaults."""
    faker = Faker()
    return TransactionRecord(
        id=str(uuid.uuid4()),
        idempotency_key=str(uuid.uuid4()),
        customer_id=customer.id,
        merchant_id=merchant.id,
        amount=amount,
        currency="USD",
        status="APPROVED",
        payment_method="CARD",
        card_last4=card_last4 or f"{np.random.default_rng().integers(1000, 9999)}",
        ip_address=ip or faker.ipv4_public(),
        device_id=device_id or str(uuid.uuid4())[:16],
        country=country or customer.country,
        is_card_present=is_card_present,
        is_fraud=True,
        fraud_pattern=pattern,
        created_at=ts,
    )


def _inject_card_testing(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n_clusters: int,
    rng: np.random.Generator,
) -> list[TransactionRecord]:
    """Card testing: 3-8 small transactions in <60 seconds.

    Pattern: a fraudster with a stolen card verifies it works by running
    small charges in rapid succession before attempting a large purchase.
    """
    fraud_txs: list[TransactionRecord] = []
    for _ in range(n_clusters):
        customer = customers[int(rng.integers(0, len(customers)))]
        cluster_size = int(rng.integers(3, 9))
        base_ts = datetime.utcnow() - timedelta(days=float(rng.uniform(0, 30)))
        # Same compromised card across the cluster
        card_last4 = f"{int(rng.integers(1000, 9999))}"
        device_id = str(uuid.uuid4())[:16]

        for i in range(cluster_size):
            merchant = merchants[int(rng.integers(0, len(merchants)))]
            ts = base_ts + timedelta(seconds=int(rng.integers(2, 15)) * i)
            amount = Decimal(f"{float(rng.uniform(0.5, 5.0)):.2f}")
            fraud_txs.append(
                _make_fraud_tx(
                    customer, merchant, amount, ts,
                    pattern="card_testing",
                    card_last4=card_last4,
                    device_id=device_id,
                )
            )
    return fraud_txs


def _inject_geo_velocity(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n: int,
    rng: np.random.Generator,
) -> list[TransactionRecord]:
    """Geo-velocity: same card in two countries within 1 hour.

    Pattern: card-not-present fraud where a stolen card is used by the
    legitimate holder and a fraudster in different countries simultaneously.
    """
    foreign_countries = ["RU", "CN", "NG", "RO", "VE", "ID"]
    fraud_txs: list[TransactionRecord] = []
    for _ in range(n):
        customer = customers[int(rng.integers(0, len(customers)))]
        merchant = merchants[int(rng.integers(0, len(merchants)))]
        ts = datetime.utcnow() - timedelta(days=float(rng.uniform(0, 30)))
        foreign_country = str(rng.choice(foreign_countries))
        amount = Decimal(f"{float(rng.uniform(100, 800)):.2f}")
        fraud_txs.append(
            _make_fraud_tx(
                customer, merchant, amount, ts,
                pattern="geo_velocity",
                country=foreign_country,
                is_card_present=False,
            )
        )
    return fraud_txs


def _inject_account_takeover(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n: int,
    rng: np.random.Generator,
) -> list[TransactionRecord]:
    """Account takeover: high-value transaction after dormancy.

    Pattern: credentials stolen via phishing or data breach. The account
    suddenly makes a large purchase after being inactive.
    """
    fraud_txs: list[TransactionRecord] = []
    # Pick customers from the high-risk-tier pool more often
    high_tier = [c for c in customers if c.risk_tier in ("MEDIUM", "HIGH")]
    pool = high_tier if high_tier else customers

    high_value_categories = [
        m for m in merchants
        if m.category in ("ELECTRONICS", "JEWELRY", "TRAVEL", "CRYPTO")
    ]
    if not high_value_categories:
        high_value_categories = merchants

    for _ in range(n):
        customer = pool[int(rng.integers(0, len(pool)))]
        merchant = high_value_categories[
            int(rng.integers(0, len(high_value_categories)))
        ]
        ts = datetime.utcnow() - timedelta(days=float(rng.uniform(0, 30)))
        amount = Decimal(f"{float(rng.uniform(800, 5000)):.2f}")
        fraud_txs.append(
            _make_fraud_tx(
                customer, merchant, amount, ts,
                pattern="account_takeover",
                is_card_present=False,
            )
        )
    return fraud_txs


def _inject_amount_anomaly(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n: int,
    rng: np.random.Generator,
) -> list[TransactionRecord]:
    """Amount anomaly: transaction 5-15x typical customer amount.

    Pattern: fraudster liquidating a stolen card before detection.
    """
    fraud_txs: list[TransactionRecord] = []
    for _ in range(n):
        customer = customers[int(rng.integers(0, len(customers)))]
        merchant = merchants[int(rng.integers(0, len(merchants)))]
        ts = datetime.utcnow() - timedelta(days=float(rng.uniform(0, 30)))
        # Very large amounts, unusual for typical traffic
        amount = Decimal(f"{float(rng.uniform(2000, 8000)):.2f}")
        fraud_txs.append(
            _make_fraud_tx(
                customer, merchant, amount, ts,
                pattern="amount_anomaly",
            )
        )
    return fraud_txs


def _inject_off_hours(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n: int,
    rng: np.random.Generator,
) -> list[TransactionRecord]:
    """Off-hours pattern: 2am-5am transactions, online-only.

    Pattern: bot-driven fraud running while the cardholder sleeps.
    """
    fraud_txs: list[TransactionRecord] = []
    for _ in range(n):
        customer = customers[int(rng.integers(0, len(customers)))]
        merchant = merchants[int(rng.integers(0, len(merchants)))]
        base = datetime.utcnow() - timedelta(days=float(rng.uniform(0, 30)))
        hour = int(rng.integers(2, 5))
        ts = base.replace(hour=hour, minute=int(rng.integers(0, 60)))
        amount = Decimal(f"{float(rng.uniform(50, 800)):.2f}")
        fraud_txs.append(
            _make_fraud_tx(
                customer, merchant, amount, ts,
                pattern="off_hours",
                is_card_present=False,
            )
        )
    return fraud_txs


def _inject_merchant_concentration(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    n: int,
    rng: np.random.Generator,
) -> list[TransactionRecord]:
    """Merchant concentration: cluster of fraud at one risky merchant.

    Pattern: compromised merchant terminal or money laundering ring.
    """
    fraud_txs: list[TransactionRecord] = []
    high_risk_merchants = [m for m in merchants if m.risk_rating == "HIGH"]
    if not high_risk_merchants:
        high_risk_merchants = merchants

    # Pick one compromised merchant per cluster
    n_clusters = max(1, n // 5)
    for _ in range(n_clusters):
        compromised = high_risk_merchants[
            int(rng.integers(0, len(high_risk_merchants)))
        ]
        cluster_size = int(rng.integers(3, 8))
        base_ts = datetime.utcnow() - timedelta(days=float(rng.uniform(0, 30)))
        for i in range(cluster_size):
            customer = customers[int(rng.integers(0, len(customers)))]
            ts = base_ts + timedelta(minutes=int(rng.integers(1, 60)) * i)
            amount = Decimal(f"{float(rng.uniform(200, 1500)):.2f}")
            fraud_txs.append(
                _make_fraud_tx(
                    customer, compromised, amount, ts,
                    pattern="merchant_concentration",
                )
            )
    return fraud_txs


def inject_fraud_patterns(
    customers: list[CustomerRecord],
    merchants: list[MerchantRecord],
    target_fraud_count: int,
    *,
    seed: int = 42,
) -> list[TransactionRecord]:
    """Generate fraud transactions distributed across the 6 patterns.

    Returns approximately target_fraud_count fraudulent transactions.
    Exact count may vary slightly due to cluster-based patterns.
    """
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    # Roughly equal allocation across patterns; cluster patterns produce
    # multiple transactions per call so we scale accordingly
    per_pattern = target_fraud_count // 6

    fraud_txs: list[TransactionRecord] = []
    # Card testing: ~5 txs per cluster, so request fewer clusters
    fraud_txs.extend(_inject_card_testing(customers, merchants, per_pattern // 5 + 1, rng))
    fraud_txs.extend(_inject_geo_velocity(customers, merchants, per_pattern, rng))
    fraud_txs.extend(_inject_account_takeover(customers, merchants, per_pattern, rng))
    fraud_txs.extend(_inject_amount_anomaly(customers, merchants, per_pattern, rng))
    fraud_txs.extend(_inject_off_hours(customers, merchants, per_pattern, rng))
    # Merchant concentration: ~5 txs per cluster
    fraud_txs.extend(
        _inject_merchant_concentration(customers, merchants, per_pattern, rng)
    )

    return fraud_txs
