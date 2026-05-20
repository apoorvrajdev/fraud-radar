"""CLI to generate the synthetic fraud dataset and persist to DB + CSV.

Usage:
    cd backend
    uv run python -m ml.generate_dataset
    uv run python -m ml.generate_dataset --n-transactions 50000 --fraud-rate 0.015
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import delete

from app.db import SessionLocal
from app.models import Customer, Merchant, Transaction
from ml.synthesis import (
    generate_customers,
    generate_legitimate_transactions,
    generate_merchants,
    inject_fraud_patterns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("generate_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--n-customers", type=int, default=500)
    parser.add_argument("--n-merchants", type=int, default=200)
    parser.add_argument("--n-transactions", type=int, default=50_000)
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.015,
        help="Approximate fraction of fraudulent transactions",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ml/data/synthetic_transactions.csv"),
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip DB persistence; write CSV only",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV; write DB only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    log.info("Generating %d customers...", args.n_customers)
    customers = generate_customers(args.n_customers, seed=args.seed)

    log.info("Generating %d merchants...", args.n_merchants)
    merchants = generate_merchants(args.n_merchants, seed=args.seed)

    n_fraud_target = int(args.n_transactions * args.fraud_rate)
    n_legit = args.n_transactions - n_fraud_target
    log.info(
        "Generating %d legitimate transactions (target fraud rate: %.2f%%)...",
        n_legit,
        args.fraud_rate * 100,
    )
    legit_txs = generate_legitimate_transactions(
        customers, merchants, n_legit, seed=args.seed
    )

    log.info("Injecting fraud patterns (~%d fraud transactions)...", n_fraud_target)
    fraud_txs = inject_fraud_patterns(
        customers, merchants, n_fraud_target, seed=args.seed
    )

    all_txs = legit_txs + fraud_txs
    random.shuffle(all_txs)

    # Summary of fraud pattern distribution
    pattern_counts = Counter(tx.fraud_pattern for tx in fraud_txs)
    log.info("Fraud pattern breakdown:")
    for pattern, count in sorted(pattern_counts.items()):
        log.info("    %-25s %d", pattern, count)
    log.info(
        "TOTAL: %d transactions  |  %d legit  |  %d fraud  |  fraud rate %.2f%%",
        len(all_txs),
        len(legit_txs),
        len(fraud_txs),
        len(fraud_txs) / len(all_txs) * 100,
    )

    if not args.no_db:
        persist_to_db(customers, merchants, all_txs)

    if not args.no_csv:
        persist_to_csv(args.csv_path, all_txs)

    log.info("Done.")


def persist_to_db(customers, merchants, transactions) -> None:
    """Write all generated records to the SQLite database via SQLAlchemy."""
    log.info("Persisting to database...")
    with SessionLocal() as db:
        # Clear existing data to make the generator idempotent
        log.info("    clearing existing transactions, customers, merchants...")
        db.execute(delete(Transaction))
        db.execute(delete(Customer))
        db.execute(delete(Merchant))
        db.flush()

        # Bulk insert customers
        db.bulk_save_objects(
            [
                Customer(
                    id=c.id,
                    email=c.email,
                    full_name=c.full_name,
                    country=c.country,
                    risk_tier=c.risk_tier,
                    account_age_days=c.account_age_days,
                )
                for c in customers
            ]
        )

        # Bulk insert merchants
        db.bulk_save_objects(
            [
                Merchant(
                    id=m.id,
                    name=m.name,
                    category=m.category,
                    mcc=m.mcc,
                    country=m.country,
                    risk_rating=m.risk_rating,
                )
                for m in merchants
            ]
        )

        # Bulk insert transactions in chunks for memory efficiency
        log.info("    inserting %d transactions in chunks...", len(transactions))
        chunk_size = 5000
        for i in range(0, len(transactions), chunk_size):
            chunk = transactions[i : i + chunk_size]
            db.bulk_save_objects(
                [
                    Transaction(
                        id=t.id,
                        idempotency_key=t.idempotency_key,
                        customer_id=t.customer_id,
                        merchant_id=t.merchant_id,
                        amount=t.amount,
                        currency=t.currency,
                        status=t.status,
                        payment_method=t.payment_method,
                        card_last4=t.card_last4,
                        ip_address=t.ip_address,
                        device_id=t.device_id,
                        country=t.country,
                        is_card_present=t.is_card_present,
                        created_at=t.created_at,
                    )
                    for t in chunk
                ]
            )
        db.commit()
    log.info("    persisted to DB successfully.")


def persist_to_csv(path: Path, transactions) -> None:
    """Write all transactions to CSV including the fraud labels for ML training."""
    log.info("Writing CSV to %s...", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(transactions[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tx in transactions:
            row = asdict(tx)
            # Stringify Decimal and datetime for CSV
            row["amount"] = str(row["amount"])
            row["created_at"] = row["created_at"].isoformat()
            writer.writerow(row)
    log.info("    wrote %d rows to %s", len(transactions), path)


if __name__ == "__main__":
    main()
