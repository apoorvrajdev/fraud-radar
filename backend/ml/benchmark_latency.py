"""Latency benchmark for the Phase 3C scoring pipeline.

Two phases per ADR decision #3:

  * service-only  →  `score_transaction` timing, no HTTP. Answers
    "how fast is the scoring math?"
  * endpoint     →  full POST round-trip via httpx against a running
    uvicorn. Answers "how fast is the production code path?"

Output goes to `backend/ml/artifacts/latency_metrics.json`. The README
inference-latency row reads from this file.

Usage:

    cd backend
    # service-layer only (no server needed)
    uv run python -m ml.benchmark_latency --service-only --n 100

    # full benchmark (start uvicorn first in another terminal)
    uv run uvicorn app.main:app --port 8000
    uv run python -m ml.benchmark_latency --n 100
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from app.db import SessionLocal
from app.fraud.explainer import initialize_explainer
from app.models.customer import Customer
from app.models.merchant import Merchant
from app.models.transaction import Transaction
from app.services.scoring import _MODEL_VERSION, score_transaction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("benchmark_latency")

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_OUTPUT_PATH = _ARTIFACTS_DIR / "latency_metrics.json"


# ---------------------------------------------------------------------------
# Synthetic test customer + merchant (reused across all sampled transactions)
# ---------------------------------------------------------------------------

# Deterministic UUIDs (valid 36-char v4 format) — fixed so successive
# benchmark runs reuse the same seeded customer/merchant rather than
# polluting the DB with new ones every run. The old "bench-customer-..."
# literals were 43 chars and tripped the 36-char Pydantic cap on POST,
# silently producing 100×422 errors and bogus endpoint latencies.
_BENCH_CUSTOMER_ID = "00000000-0000-4000-8000-000000000001"
_BENCH_MERCHANT_ID = "00000000-0000-4000-8000-000000000002"


def _ensure_seed_entities(db) -> None:  # type: ignore[no-untyped-def]
    """Insert the benchmark customer and merchant if they don't exist.

    Idempotent — safe to call repeatedly across benchmark runs. The dev
    DB is shared across runs (we don't tear it down between
    invocations), so this function must tolerate a pre-existing row.
    Uses `db.get(Model, pk)` for the existence check — the canonical
    SQLAlchemy 2.0 primary-key lookup, which is identity-map-aware.
    """
    existing_customer = db.get(Customer, _BENCH_CUSTOMER_ID)
    if existing_customer is None:
        customer = Customer(
            id=_BENCH_CUSTOMER_ID,
            email="bench@example.com",
            full_name="Benchmark Customer",
            country="US",
            risk_tier="LOW",
            account_age_days=365,
        )
        db.add(customer)

    existing_merchant = db.get(Merchant, _BENCH_MERCHANT_ID)
    if existing_merchant is None:
        merchant = Merchant(
            id=_BENCH_MERCHANT_ID,
            name="Benchmark Merchant",
            category="RETAIL",
            mcc="5311",
            country="US",
            risk_rating="LOW",
        )
        db.add(merchant)

    db.flush()  # surface any unexpected FK / UNIQUE violation before scoring


def _make_bench_transaction(idx: int) -> Transaction:
    """Build a fresh Transaction at NOW for benchmarking."""
    return Transaction(
        id=str(uuid.uuid4()),
        idempotency_key=f"bench-{idx}-{uuid.uuid4()}",
        customer_id=_BENCH_CUSTOMER_ID,
        merchant_id=_BENCH_MERCHANT_ID,
        amount=Decimal("100.00"),
        currency="USD",
        status="PENDING_REVIEW",
        payment_method="CARD",
        country="US",
        is_card_present=True,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """p50 / p95 / p99 of a list of milliseconds."""

    p50: float
    p95: float
    p99: float
    n: int

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "LatencyStats":
        if not samples_ms:
            return cls(0.0, 0.0, 0.0, 0)
        sorted_samples = sorted(samples_ms)
        return cls(
            p50=_percentile(sorted_samples, 50),
            p95=_percentile(sorted_samples, 95),
            p99=_percentile(sorted_samples, 99),
            n=len(sorted_samples),
        )


def _percentile(sorted_samples: list[float], pct: int) -> float:
    """Nearest-rank percentile — fine for n>=100."""
    if not sorted_samples:
        return 0.0
    idx = max(0, min(len(sorted_samples) - 1, (pct * len(sorted_samples)) // 100))
    return float(sorted_samples[idx])


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def run_service_phase(n: int) -> LatencyStats:
    """Time `score_transaction` directly. Rolls back at the end."""
    log.info("Service-only phase: %d transactions", n)
    initialize_explainer(_ARTIFACTS_DIR)
    samples_ms: list[float] = []
    with SessionLocal() as db:
        try:
            _ensure_seed_entities(db)
            db.commit()
            for i in range(n):
                tx = _make_bench_transaction(i)
                db.add(tx)
                db.flush()
                t0 = time.perf_counter()
                score_transaction(db, tx, write_audit=False)
                t1 = time.perf_counter()
                samples_ms.append((t1 - t0) * 1000.0)
        finally:
            db.rollback()
    stats = LatencyStats.from_samples(samples_ms)
    log.info(
        "  service p50=%.2fms p95=%.2fms p99=%.2fms",
        stats.p50, stats.p95, stats.p99,
    )
    return stats


def run_endpoint_phase(n: int, server_url: str) -> LatencyStats:
    """Time full POST round-trips. Requires uvicorn on `server_url`.

    Raises `httpx.ConnectError` if no server responds.
    """
    log.info("Endpoint phase: %d POSTs to %s", n, server_url)
    # We need real customer + merchant in the DB the server is talking to,
    # so seed them via a direct session before hammering the endpoint.
    with SessionLocal() as db:
        _ensure_seed_entities(db)
        db.commit()

    samples_ms: list[float] = []
    payload: dict[str, Any] = {
        "customer_id": _BENCH_CUSTOMER_ID,
        "merchant_id": _BENCH_MERCHANT_ID,
        "amount": "100.00",
        "currency": "USD",
        "payment_method": "CARD",
        "country": "US",
        "is_card_present": True,
    }
    with httpx.Client(base_url=server_url, timeout=10.0) as client:
        # Single warm-up call so connection pool / model lazy-loads don't
        # skew the first sample.
        client.post(
            "/api/v1/transactions",
            json=payload,
            headers={"Idempotency-Key": f"bench-warmup-{uuid.uuid4()}"},
        )
        for _ in range(n):
            key = f"bench-{uuid.uuid4()}"
            t0 = time.perf_counter()
            resp = client.post(
                "/api/v1/transactions",
                json=payload,
                headers={"Idempotency-Key": key},
            )
            t1 = time.perf_counter()
            if resp.status_code != 201:
                log.warning("Non-201 response: %d %s", resp.status_code, resp.text)
            samples_ms.append((t1 - t0) * 1000.0)
    stats = LatencyStats.from_samples(samples_ms)
    log.info(
        "  endpoint p50=%.2fms p95=%.2fms p99=%.2fms",
        stats.p50, stats.p95, stats.p99,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3C latency benchmark")
    p.add_argument("--n", type=int, default=100, help="Samples per phase")
    p.add_argument("--service-only", action="store_true")
    p.add_argument("--endpoint-only", action="store_true")
    p.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Where to find uvicorn for the endpoint phase",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out: dict[str, Any] = {
        "measured_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0).isoformat(),
        "model_version": _MODEL_VERSION,
    }

    if not args.endpoint_only:
        service_stats = run_service_phase(args.n)
        out["service_layer"] = asdict(service_stats)

    if not args.service_only:
        try:
            endpoint_stats = run_endpoint_phase(args.n, args.server_url)
            out["endpoint"] = asdict(endpoint_stats)
        except httpx.ConnectError as exc:
            log.error("Endpoint phase failed — is uvicorn running on %s?", args.server_url)
            out["endpoint_error"] = f"connection failed: {exc}"

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    log.info("Wrote %s", _OUTPUT_PATH)

    # Compact summary line for CI logs / Slack pastes.
    parts: list[str] = []
    if "service_layer" in out:
        s = out["service_layer"]
        parts.append(f"service p99={s['p99']:.1f}ms")
    if "endpoint" in out:
        e = out["endpoint"]
        parts.append(f"endpoint p99={e['p99']:.1f}ms")
    if parts:
        log.info("SUMMARY  %s", "  |  ".join(parts))


if __name__ == "__main__":
    main()
