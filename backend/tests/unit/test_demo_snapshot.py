"""The committed demo snapshot is internally consistent.

These tests exist because of a real production bug. `vercel.json`
rewrites `/(.*)` to `/` so deep links survive a refresh, and that
rewrite catches `/demo-data/...` too: a snapshot file that is not there
comes back as **index.html with a 200**, not a 404. So a listed row
whose detail page is missing did not degrade to the "not in snapshot"
notice the demo was designed to show — it reached `response.json()` as
HTML and surfaced `Unexpected token '<'` to the user.

The client now detects that (see the content-type check in
`frontend/src/lib/demoApi.ts`), but the durable fix is that the
situation should not arise: every row the demo can display should have
a page to open. That is what this file pins. It is the same pattern the
benchmark-card tests use — assert a committed artifact against the
thing it is supposed to describe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_SNAPSHOT = (
    Path(__file__).resolve().parents[3] / "frontend" / "public" / "demo-data"
)

pytestmark = pytest.mark.skipif(
    not (_SNAPSHOT / "manifest.json").exists(),
    reason=f"No demo snapshot at {_SNAPSHOT}. Run scripts/export_demo_snapshot.py.",
)


def _load(name: str) -> Any:
    with (_SNAPSHOT / name).open(encoding="utf-8") as f:
        return json.load(f)


def _detail_ids() -> set[str]:
    return {p.stem for p in (_SNAPSHOT / "transactions").glob("*.json")}


# ---------------------------------------------------------------------------
# The invariant the bug violated
# ---------------------------------------------------------------------------


def test_every_listed_transaction_has_a_detail_page() -> None:
    """Clicking a row is the primary interaction — none may dead-end.

    This is the assertion that would have caught the reported bug: 272
    of 300 listed rows had no detail page, so nine clicks in ten hit the
    missing-file path.
    """
    rows = _load("transactions.json")["items"]
    have = _detail_ids()

    missing = [row["id"] for row in rows if row["id"] not in have]

    assert not missing, (
        f"{len(missing)} of {len(rows)} listed transactions have no detail "
        f"page; clicking those rows in the demo cannot open anything. "
        f"First few: {missing[:3]}"
    )


def test_every_alert_has_a_detail_page() -> None:
    """The alerts queue links to the same detail route.

    This path was already correct — the export has always pulled every
    alert row into the detail set — which is exactly why Alerts worked
    while Transactions did not. Pinning it keeps that true.
    """
    alerts = _load("alerts.json")["items"]
    have = _detail_ids()

    missing = [row["id"] for row in alerts if row["id"] not in have]

    assert not missing, f"Alert rows with no detail page: {missing[:3]}"


def test_no_orphan_detail_pages() -> None:
    """Every detail page is reachable from the list or the queue.

    An orphan is dead weight shipped to the CDN and makes the directory
    a misleading record of what the snapshot contains.
    """
    reachable = {row["id"] for row in _load("transactions.json")["items"]}
    reachable |= {row["id"] for row in _load("alerts.json")["items"]}

    orphans = _detail_ids() - reachable

    assert not orphans, f"Detail pages nothing links to: {sorted(orphans)[:3]}"


# ---------------------------------------------------------------------------
# The snapshot describes itself honestly
# ---------------------------------------------------------------------------


def test_manifest_counts_match_the_files_on_disk() -> None:
    counts = _load("manifest.json")["row_counts"]

    assert counts["transactions"] == len(_load("transactions.json")["items"])
    assert counts["alerts"] == len(_load("alerts.json")["items"])
    assert counts["transaction_details"] == len(_detail_ids())


def test_every_route_the_demo_adapter_serves_has_a_file() -> None:
    """The adapter's route table and the snapshot must not drift apart.

    A route with no file would hit the same rewrite-shaped failure as
    the bug this file documents.
    """
    for name in (
        "stats-overview.json",
        "stats-timeseries.json",
        "stats-breakdown.json",
        "transactions.json",
        "alerts.json",
        "model.json",
    ):
        assert (_SNAPSHOT / name).exists(), f"demo adapter serves {name}, which is absent"


# ---------------------------------------------------------------------------
# Phase 5G: the snapshot can actually demonstrate the currency work
# ---------------------------------------------------------------------------


def test_snapshot_contains_a_reporting_currency_transaction() -> None:
    rows = _load("transactions.json")["items"]

    identity = [r for r in rows if r.get("fx_source") == "identity"]

    assert identity, "no same-currency row — the 1:1 path is undemonstrated"


def test_snapshot_contains_a_converted_non_usd_transaction() -> None:
    """Without one of these the demo cannot show what 5F built."""
    rows = _load("transactions.json")["items"]

    converted = [
        r
        for r in rows
        if r.get("currency") != "USD" and r.get("amount_base") is not None
    ]

    assert converted, "no converted non-USD row in the snapshot"
    row = converted[0]
    assert row["fx_rate"] is not None
    assert row["fx_rate_date"] is not None
    assert row["fx_source"] in {"live", "cache", "stale"}


def test_a_converted_row_keeps_its_original_amount_distinct() -> None:
    """The derived figure must not have overwritten the charged one."""
    rows = _load("transactions.json")["items"]

    for row in rows:
        if row.get("currency") != "USD" and row.get("amount_base") is not None:
            assert row["amount"] != row["amount_base"], (
                f"{row['id']}: converted amount equals the original, which "
                f"means the original was overwritten or the rate was 1"
            )


def test_unconverted_rows_have_all_four_fx_fields_null() -> None:
    """A half-converted row would let the UI render a figure with no rate."""
    rows = _load("transactions.json")["items"]

    for row in rows:
        if row.get("fx_source") in {"unsupported", "unavailable"}:
            assert row["amount_base"] is None
            assert row["fx_rate"] is None
            assert row["fx_rate_date"] is None
