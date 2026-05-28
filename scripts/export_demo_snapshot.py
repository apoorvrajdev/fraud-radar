"""Export a snapshot of the live Fraud Radar API to static JSON.

Run this against a local backend (with the simulator having run long
enough to populate realistic data). The output is written to
``frontend/public/demo-data/`` and bundled into the Vite build at
``npm run build`` time, then served from Vercel's CDN as the public
demo.

Contract is locked in ``docs/adr/PHASE_4A_DEMO_SCOPE.md``. Do not
change filenames or field shapes without updating that ADR and the
Phase 4C demo-mode client in lockstep.

Usage::

    # 1. Start the backend in another terminal:
    #    cd backend && uv run uvicorn app.main:app --port 8000
    # 2. (Optional) Let the simulator run for ~30 minutes to build
    #    up a realistic mix of decisions and alerts.
    # 3. Run this script (from anywhere; paths are repo-relative):
    #    uv run --project backend python scripts/export_demo_snapshot.py

    # Custom backend URL:
    #    uv run --project backend python scripts/export_demo_snapshot.py \\
    #        --base-url http://localhost:8000/api/v1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Locked by PHASE_4A_DEMO_SCOPE.md.
SCHEMA_VERSION = "1"
TRANSACTIONS_LIMIT = 300
ALERTS_LIMIT = 100

# Backend caps both list endpoints at 200 rows per request. To honour
# the 300-row contract from the ADR we walk the cursor.
PAGE_SIZE = 200

# Curated detail-page coverage targets (see ADR).
DETAIL_TARGETS_BLOCK = 10
DETAIL_TARGETS_REVIEW = 10
DETAIL_TARGETS_ALLOW = 8
DETAIL_TARGETS_OVERRIDDEN = 2

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "frontend" / "public" / "demo-data"
DEFAULT_BASE_URL = "http://localhost:8000/api/v1"


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` to ``path`` as pretty JSON for diff-ability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _get(client: httpx.Client, url: str, **params: Any) -> Any:
    """GET ``url`` and return parsed JSON, raising on non-2xx."""
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def _select_detail_ids(transactions: list[dict[str, Any]]) -> list[str]:
    """Pick ~30 transaction IDs covering each decision class.

    See PHASE_4A_DEMO_SCOPE.md for the selection contract. Picks rows
    in list order, so the first N matching rows per bucket win — the
    list endpoint already returns newest-first, which is what we want.
    """
    by_decision: dict[str, list[dict[str, Any]]] = {
        "BLOCK": [],
        "REVIEW": [],
        "ALLOW": [],
    }
    overridden: list[dict[str, Any]] = []

    for tx in transactions:
        decision = tx.get("fraud_decision")
        if tx.get("analyst_label") is not None:
            overridden.append(tx)
        if decision in by_decision:
            by_decision[decision].append(tx)

    picked: list[str] = []
    picked.extend(tx["id"] for tx in by_decision["BLOCK"][:DETAIL_TARGETS_BLOCK])
    picked.extend(tx["id"] for tx in by_decision["REVIEW"][:DETAIL_TARGETS_REVIEW])
    picked.extend(tx["id"] for tx in by_decision["ALLOW"][:DETAIL_TARGETS_ALLOW])
    picked.extend(tx["id"] for tx in overridden[:DETAIL_TARGETS_OVERRIDDEN])

    # De-dupe while preserving order (an overridden row may also be in
    # one of the decision buckets).
    seen: set[str] = set()
    unique: list[str] = []
    for tx_id in picked:
        if tx_id not in seen:
            seen.add(tx_id)
            unique.append(tx_id)
    return unique


def export(base_url: str, output_dir: Path) -> dict[str, int]:
    """Fetch every snapshot file and write it under ``output_dir``.

    Returns a dict of row counts suitable for ``manifest.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transactions").mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # Stats endpoints.
        print("→ fetching /stats/overview")
        overview = _get(client, "/stats/overview")
        _write_json(output_dir / "stats-overview.json", overview)

        print("→ fetching /stats/timeseries")
        timeseries = _get(client, "/stats/timeseries")
        _write_json(output_dir / "stats-timeseries.json", timeseries)

        print("→ fetching /stats/breakdown")
        breakdown = _get(client, "/stats/breakdown")
        _write_json(output_dir / "stats-breakdown.json", breakdown)

        # Transactions list — walk the cursor to honour the 300-row
        # contract (backend caps `limit` at 200 per request). Strip
        # `next_cursor` on the final envelope so the demo client never
        # tries to walk past the snapshot.
        print(f"→ fetching /transactions (paged, target={TRANSACTIONS_LIMIT})")
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        first_envelope: dict[str, Any] | None = None
        while len(items) < TRANSACTIONS_LIMIT:
            params: dict[str, Any] = {
                "limit": min(PAGE_SIZE, TRANSACTIONS_LIMIT - len(items)),
            }
            if cursor:
                params["cursor"] = cursor
            page = _get(client, "/transactions", **params)
            if first_envelope is None:
                first_envelope = page
            page_items = page.get("items", [])
            if not page_items:
                break
            items.extend(page_items)
            cursor = page.get("next_cursor")
            if not cursor:
                break
        assert first_envelope is not None
        transactions = {**first_envelope, "items": items, "next_cursor": None}
        _write_json(output_dir / "transactions.json", transactions)
        counts["transactions"] = len(items)

        # Alerts queue.
        print(f"→ fetching /alerts?limit={ALERTS_LIMIT}")
        alerts = _get(client, "/alerts", limit=ALERTS_LIMIT)
        alerts["next_cursor"] = None
        _write_json(output_dir / "alerts.json", alerts)
        counts["alerts"] = len(alerts.get("items", []))

        # Curated detail pages.
        detail_ids = _select_detail_ids(transactions.get("items", []))
        # Pull every alert row into the detail set too, so clicking any
        # row on /alerts lands on a working detail page.
        for alert in alerts.get("items", []):
            tx_id = alert.get("id")
            if tx_id and tx_id not in detail_ids:
                detail_ids.append(tx_id)

        print(f"→ fetching {len(detail_ids)} curated /transactions/{{id}} pages")
        for tx_id in detail_ids:
            detail = _get(client, f"/transactions/{tx_id}")
            _write_json(output_dir / "transactions" / f"{tx_id}.json", detail)
        counts["transaction_details"] = len(detail_ids)

    # Manifest last, so partial failures don't leave a stale date.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_url": base_url,
        "row_counts": dict(counts),
    }
    _write_json(output_dir / "manifest.json", manifest)

    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Fraud Radar API snapshot for the public Vercel demo.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Backend API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args(argv)

    try:
        counts = export(args.base_url, args.output_dir)
    except httpx.HTTPError as exc:
        print(f"error: backend request failed: {exc}", file=sys.stderr)
        print(
            "hint: is the backend running at "
            f"{args.base_url}? Start it with: "
            "cd backend && uv run uvicorn app.main:app --port 8000",
            file=sys.stderr,
        )
        return 1

    print()
    print(f"✓ snapshot written to {args.output_dir.relative_to(REPO_ROOT)}")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
