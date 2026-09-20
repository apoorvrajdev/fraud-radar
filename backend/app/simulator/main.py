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

# Currencies `--currency-mix` will emit (Phase 5G).
#
# The list is restricted on purpose. The rules engine thresholds — the
# $5,000 amount ceiling above all — are denominated in the raw amount,
# not a converted one (see docs/FX_CONTRACT.md, "Deliberately not
# done"). Emitting a currency of a very different magnitude would
# therefore silently change which rules fire: ¥60,000 is about $400 but
# would trip a ceiling meant for large payments. Every currency here is
# within roughly a factor of two of USD, so an amount drawn in USD
# range stays plausible in it and rule behaviour is unchanged.
#
# This is a simulator-realism constraint, not an FX rate. No conversion
# happens here; the backend does that, from real reference rates.
_MIXABLE_CURRENCIES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "CHF", "CAD", "AUD", "SGD", "NZD"}
)

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


def parse_currency_mix(spec: str) -> dict[str, float]:
    """Parse ``"USD:0.85,EUR:0.1,GBP:0.05"`` into normalised weights.

    Weights are normalised rather than required to sum to 1, so a
    caller can write ``"USD:8,EUR:2"`` and mean 80/20.

    Raises ValueError on an unparseable entry, a non-positive weight, a
    code outside `_MIXABLE_CURRENCIES`, or an empty mix — a typo in a
    currency code should stop the run, not quietly emit traffic in a
    currency nobody asked for.
    """
    weights: dict[str, float] = {}
    for chunk in spec.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        code, sep, raw_weight = entry.partition(":")
        code = code.strip().upper()
        if not sep:
            raise ValueError(f"Expected CODE:WEIGHT, got {entry!r}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"Weight for {code} is not a number: {raw_weight!r}") from exc
        if weight <= 0:
            raise ValueError(f"Weight for {code} must be > 0, got {weight}")
        if code not in _MIXABLE_CURRENCIES:
            supported = ", ".join(sorted(_MIXABLE_CURRENCIES))
            raise ValueError(
                f"{code} is not available for --currency-mix. Supported: {supported}. "
                "See the note on _MIXABLE_CURRENCIES for why the set is limited."
            )
        weights[code] = weights.get(code, 0.0) + weight

    if not weights:
        raise ValueError("Currency mix is empty")

    total = sum(weights.values())
    return {code: weight / total for code, weight in weights.items()}


def _pick_currency(mix: dict[str, float]) -> str:
    """Draw one currency from a normalised mix."""
    codes = sorted(mix)
    return random.choices(codes, weights=[mix[c] for c in codes], k=1)[0]


def _build_payload(
    customer_ids: list[str],
    merchant_ids: list[str],
    *,
    fraud_rate: float,
    currency_mix: dict[str, float] | None = None,
) -> tuple[dict[str, Any], str]:
    """Sample one customer + merchant, decide clean vs fraud, return
    `(payload, pattern_label)`.

    `pattern_label` is `"clean"` for clean transactions, otherwise the
    name of the fraud pattern selected (`"high_amount"`, etc.).

    `currency_mix`, when given, replaces the payload's currency with one
    drawn from the mix. Only the currency changes — the amount, country
    and card-present flags each pattern sets are untouched, so the fraud
    signal a pattern encodes is exactly what it was before.
    """
    customer_id = random.choice(customer_ids)
    merchant_id = random.choice(merchant_ids)
    if random.random() < fraud_rate:
        pattern = random.choice(_FRAUD_PATTERNS)
        payload = _generate_fraud_payload(customer_id, merchant_id, pattern)
        label = pattern
    else:
        payload = _generate_clean_payload(customer_id, merchant_id)
        label = "clean"

    if currency_mix:
        payload["currency"] = _pick_currency(currency_mix)
    return payload, label


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
    currency_mix: dict[str, float] | None = None,
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
    if currency_mix:
        log.info(
            "Currency mix — %s",
            "  ".join(
                f"{code} {share:.0%}" for code, share in sorted(currency_mix.items())
            ),
        )

    interval = 1.0 / rate if rate > 0 else 1.0
    idx = 0
    try:
        with httpx.Client(timeout=10.0) as client:
            while n is None or idx < n:
                payload, pattern = _build_payload(
                    customers, merchants,
                    fraud_rate=fraud_rate,
                    currency_mix=currency_mix,
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
    parser.add_argument(
        "--currency-mix", type=str, default=None,
        metavar="CODE:WEIGHT,...",
        help=(
            "Emit transactions across several currencies, e.g. "
            "'USD:0.85,EUR:0.1,GBP:0.05'. Weights are normalised. "
            "Exercises the FX enrichment path; default is USD only."
        ),
    )
    args = parser.parse_args()
    try:
        currency_mix = (
            parse_currency_mix(args.currency_mix) if args.currency_mix else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    run_simulator(
        rate=args.rate,
        fraud_rate=args.fraud_rate,
        server_url=args.server_url,
        n=args.n,
        currency_mix=currency_mix,
    )


if __name__ == "__main__":
    main()
