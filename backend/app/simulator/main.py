"""Background transaction simulator — Phase 3D.

Generates a continuous stream of synthetic transactions hitting the local
API at a dashboard-friendly rate (default 1 tx/sec). Used to populate the
dashboard and demonstrate the pipeline end to end.

This is NOT a load tester — see `ml/benchmark_latency.py` for that. The
simulator's job is to make the API look alive for the upcoming frontend
phases (3E–3H). It runs forever by default; Ctrl+C stops it cleanly.

The simulator is a normal HTTP client to its own service. It only
queries the DB once at startup (to learn the pool of customers and
merchants it can reference); all writes go through POST /transactions
so the rules engine, ML scorer, SHAP attribution, audit log, and
idempotency cache all run exactly as they would for any other client.

Usage:
    cd backend
    # in one terminal:
    uv run uvicorn app.main:app --port 8000
    # in another:
    uv run python -m app.simulator.main --rate 1 --fraud-rate 0.10
"""
from __future__ import annotations

import argparse
import logging
import random
import time
import uuid
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models.customer import Customer
from app.models.merchant import Merchant


# Skip benchmark seed entities — they live under a fixed UUID prefix
# (see ml/benchmark_latency.py) and would dominate the random sample
# if not filtered out.
_BENCH_PREFIX = "00000000-0000-4000-8000-"

_DEFAULT_RATE = 1.0
_DEFAULT_FRAUD_RATE = 0.10
_DEFAULT_SERVER_URL = "http://localhost:8000"

# Match the high-risk set in app/fraud/rules.py — used to trigger
# rule_high_risk_country.
_HIGH_RISK_COUNTRIES: tuple[str, ...] = ("RU", "CN", "NG", "RO", "VE", "ID")

# Foreign countries NOT in the high-risk set — used by the "stealth"
# pattern to lean on the ML model (country_mismatch features) without
# tripping any rule.
_FOREIGN_COUNTRIES: tuple[str, ...] = ("GB", "DE", "FR", "JP", "AU", "CA")

# Available fraud patterns. velocity_burst is intentionally absent: it
# requires controlling the customer's recent history, which the
# simulator (as an HTTP client) cannot do directly. dormant_account
# is absent for the same reason.
_FRAUD_PATTERNS: tuple[str, ...] = ("high_amount", "high_risk_country", "stealth")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # the simulator owns its own line format
)
log = logging.getLogger("simulator")


# ---------------------------------------------------------------------------
# Pool loader — one DB query at startup
# ---------------------------------------------------------------------------


def _load_customer_merchant_pool(db: Session) -> tuple[list[str], list[str]]:
    """Read the IDs of every non-benchmark customer and merchant.

    Excludes IDs starting with `_BENCH_PREFIX` so the benchmark seed
    entities (from `ml/benchmark_latency.py`) don't dominate the sample.

    Raises RuntimeError if either pool is empty — would indicate the
    dev DB hasn't been seeded with the Phase 2E dataset.
    """
    customers = [
        cid for (cid,) in db.execute(select(Customer.id)).all()
        if not cid.startswith(_BENCH_PREFIX)
    ]
    merchants = [
        mid for (mid,) in db.execute(select(Merchant.id)).all()
        if not mid.startswith(_BENCH_PREFIX)
    ]
    if not customers:
        raise RuntimeError(
            "No non-benchmark customers in the DB. "
            "Run `uv run python -m ml.generate_dataset` to seed."
        )
    if not merchants:
        raise RuntimeError(
            "No non-benchmark merchants in the DB. "
            "Run `uv run python -m ml.generate_dataset` to seed."
        )
    return customers, merchants


# ---------------------------------------------------------------------------
# Payload generators — pure functions over (customer_id, merchant_id)
# ---------------------------------------------------------------------------


def _generate_clean_payload(customer_id: str, merchant_id: str) -> dict[str, Any]:
    """Build a TransactionCreate payload for a 'clean' transaction.

    Conservative defaults — small amount, US country, card-present. The
    rules engine will not trigger; the ML model will almost always
    return APPROVE.
    """
    amount = round(random.uniform(5.0, 500.0), 2)
    return {
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "amount": f"{amount:.2f}",
        "currency": "USD",
        "payment_method": "CARD",
        "country": "US",
        "is_card_present": True,
    }


def _generate_fraud_payload(
    customer_id: str,
    merchant_id: str,
    pattern: str,
) -> dict[str, Any]:
    """Build a TransactionCreate payload that triggers a specific rule or
    pushes the ML model.

    `pattern` is one of the entries in `_FRAUD_PATTERNS`. Caller decides
    which pattern; this function fills in the parameters. Each pattern
    only sets fields the rules engine actually inspects — everything
    else stays conservative so the test signal is isolated.
    """
    if pattern == "high_amount":
        # > $5,000 → rule_amount_ceiling triggers (REVIEW)
        amount = round(random.uniform(5500.0, 8000.0), 2)
        return {
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "amount": f"{amount:.2f}",
            "currency": "USD",
            "payment_method": "CARD",
            "country": "US",
            "is_card_present": True,
        }
    if pattern == "high_risk_country":
        # high-risk country + > $500 → rule_high_risk_country triggers (REVIEW)
        amount = round(random.uniform(600.0, 4500.0), 2)
        return {
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "amount": f"{amount:.2f}",
            "currency": "USD",
            "payment_method": "CARD",
            "country": random.choice(_HIGH_RISK_COUNTRIES),
            "is_card_present": False,
        }
    if pattern == "stealth":
        # Small amount + foreign (non-high-risk) country + card-not-present.
        # Triggers no rule; depends on the ML model's country-mismatch and
        # CNP features to surface anything.
        amount = round(random.uniform(50.0, 200.0), 2)
        return {
            "customer_id": customer_id,
            "merchant_id": merchant_id,
            "amount": f"{amount:.2f}",
            "currency": "USD",
            "payment_method": "CARD",
            "country": random.choice(_FOREIGN_COUNTRIES),
            "is_card_present": False,
        }
    raise ValueError(f"Unknown fraud pattern: {pattern!r}")


def _build_payload(
    customer_ids: list[str],
    merchant_ids: list[str],
    *,
    fraud_rate: float,
) -> tuple[dict[str, Any], str]:
    """Sample one customer + merchant, decide clean vs fraud, return
    `(payload, pattern_label)`.

    `pattern_label` is `"clean"` for clean transactions, otherwise the
    name of the fraud pattern selected (`"high_amount"`, etc.).
    """
    customer_id = random.choice(customer_ids)
    merchant_id = random.choice(merchant_ids)
    if random.random() < fraud_rate:
        pattern = random.choice(_FRAUD_PATTERNS)
        return _generate_fraud_payload(customer_id, merchant_id, pattern), pattern
    return _generate_clean_payload(customer_id, merchant_id), "clean"


# ---------------------------------------------------------------------------
# HTTP client wrapper
# ---------------------------------------------------------------------------


def _post_one(
    client: httpx.Client,
    payload: dict[str, Any],
    *,
    server_url: str,
) -> tuple[int, dict[str, Any] | None, float]:
    """POST one transaction. Return `(status_code, response_json, latency_ms)`.

    Generates a fresh UUID for the Idempotency-Key header so every call
    creates a new row (the replay path is integration-tested separately).
    On connection or network error, returns `(-1, None, 0.0)` so the
    caller can log and continue — a single failed POST must not crash
    the simulator.
    """
    headers = {"Idempotency-Key": str(uuid.uuid4())}
    t0 = time.perf_counter()
    try:
        resp = client.post(
            f"{server_url}/api/v1/transactions",
            json=payload,
            headers=headers,
        )
    except httpx.RequestError:
        return (-1, None, 0.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    try:
        body = resp.json()
    except ValueError:
        body = None
    return (resp.status_code, body, latency_ms)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log_response(
    idx: int,
    pattern: str,
    status: int,
    body: dict[str, Any] | None,
    latency_ms: float,
) -> None:
    """Print one structured line per POST.

    Examples:
      [#0042] clean             status=201  decision=APPROVE  score=0.082  latency=18ms
      [#0043] high_amount       status=201  decision=REVIEW   score=0.241  latency=22ms  rules=[amount_ceiling]
      [#0044] high_risk_country status=201  decision=DECLINE  score=0.711  latency=19ms  rules=[high_risk_country]
      [#0045] stealth           status=-1   ERROR: connection refused
    """
    label = f"[#{idx:04d}] {pattern:<18}"
    if status == -1:
        log.info("%s status=-1   ERROR: connection refused", label)
        return
    if body is None:
        log.info("%s status=%d   (no JSON body)  latency=%dms", label, status, int(latency_ms))
        return
    decision = body.get("decision", "?")
    fraud_score = body.get("fraud_score")
    score_str = f"{fraud_score:.3f}" if isinstance(fraud_score, (int, float)) else "—"
    rules = body.get("rules_triggered") or []
    rules_clause = f"  rules=[{','.join(rules)}]" if rules else ""
    log.info(
        "%s status=%d  decision=%-7s  score=%s  latency=%dms%s",
        label, status, decision, score_str, int(latency_ms), rules_clause,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_simulator(
    *,
    rate: float,
    fraud_rate: float,
    server_url: str,
    n: int | None = None,
) -> None:
    """Main loop. Loads the customer/merchant pool, opens an httpx.Client,
    POSTs at the requested rate, logs each response.

    `n=None` runs forever (until Ctrl+C). `n=int` runs that many
    transactions then exits cleanly.
    """
    with SessionLocal() as db:
        customers, merchants = _load_customer_merchant_pool(db)
    log.info(
        "Simulator starting — rate=%.2f tx/s  fraud_rate=%.0f%%  "
        "pool: %d customers, %d merchants  target=%s",
        rate, fraud_rate * 100, len(customers), len(merchants), server_url,
    )

    interval = 1.0 / rate if rate > 0 else 1.0
    idx = 0
    try:
        with httpx.Client(timeout=10.0) as client:
            while n is None or idx < n:
                payload, pattern = _build_payload(
                    customers, merchants, fraud_rate=fraud_rate,
                )
                status, body, latency = _post_one(
                    client, payload, server_url=server_url,
                )
                _log_response(idx, pattern, status, body, latency)
                idx += 1
                if n is not None and idx >= n:
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        log.info("Stopped after %d transactions. Goodbye.", idx)
        return
    log.info("Completed %d transactions. Exit.", idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Background transaction simulator (Phase 3D)",
    )
    parser.add_argument(
        "--rate", type=float, default=_DEFAULT_RATE,
        help="Transactions per second (default: %(default)s)",
    )
    parser.add_argument(
        "--fraud-rate", type=float, default=_DEFAULT_FRAUD_RATE,
        help="Fraction of transactions using a fraud pattern (default: %(default)s)",
    )
    parser.add_argument(
        "--server-url", type=str, default=_DEFAULT_SERVER_URL,
        help="API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Stop after N transactions (default: run forever)",
    )
    args = parser.parse_args()
    run_simulator(
        rate=args.rate,
        fraud_rate=args.fraud_rate,
        server_url=args.server_url,
        n=args.n,
    )


if __name__ == "__main__":
    main()
