# Phase 4A Design — Public Demo Scope

> Companion to [PHASE_3_DESIGN.md](PHASE_3_DESIGN.md),
> [PHASE_3C_INTEGRATION.md](PHASE_3C_INTEGRATION.md),
> [PHASE_3E_DESIGN.md](PHASE_3E_DESIGN.md),
> [PHASE_3F_DESIGN.md](PHASE_3F_DESIGN.md),
> [PHASE_3G_DESIGN.md](PHASE_3G_DESIGN.md), and
> [PHASE_3H_DESIGN.md](PHASE_3H_DESIGN.md). Locks the contract for
> the static, zero-cost public demo before any Phase 4B–4F code is
> written.

> Status: design. Author: apoorvrajdev. Locked before implementation.

## Context

Phase 3 closed the analyst loop end-to-end on `localhost`:
simulator → scoring → dashboard → transactions → detail → alerts.
That is the full product. It is also entirely invisible to anyone
who has not cloned the repo and run two `uv` / `npm` commands.

Phase 4's job is to make the project clickable from GitHub without
paying for hosting. The constraint that drives every decision in
this ADR:

> **Zero-cost, zero-cold-start, zero-maintenance public URL.**

Every "real" backend host that meets the first two clauses fails
the third (Render free tier spins down; Fly.io's free machines
require an active credit card; Railway's free tier no longer
exists; Supabase / Neon free Postgres has row limits that the
simulator will breach within days). The only path that satisfies
all three is **no public backend at all** — ship the frontend to
Vercel and have it read a pre-generated snapshot of the live
backend instead of a live API.

This ADR does not design the snapshot script (4B), the demo-mode
client (4C), or the Loom (4E). It locks **what the demo must do**
so the next five phases have a target to hit.

## Goal

By end of Phase 4A:

1. A single source of truth exists for which pages, routes, and
   interactions the public demo at `fraud-radar.vercel.app` must
   support, and which are explicitly out of scope (Loom-only or
   local-setup-only).
2. The snapshot contract (which API responses get frozen to JSON,
   at what cardinality, with what filenames) is fixed, so the
   Phase 4B script and the Phase 4C client can be built against
   the same contract independently.
3. The demo's honesty posture is decided — banner copy, disabled
   states, "snapshot from" date stamping — so recruiters
   understand what they are looking at without it feeling
   apologetic.

## Non-Goals

- Migrating the backend to Postgres. The snapshot is JSON; no
  database is involved in the public demo.
- A Dockerfile, a `docker-compose.yml`, or any container infra.
  Dropped from Phase 4 entirely.
- A live "Generate fraud traffic" button on the public URL.
  Pre-recorded in the Loom (Phase 4E) instead.
- Server-side rendering, ISR, or any Vercel feature beyond static
  hosting + SPA rewrites. The build output is a plain static
  bundle.
- Authentication. The demo is fully read-only and public.

## Architecture Sketch

```
Recruiter
    │
    ▼
fraud-radar.vercel.app  (Vercel CDN, free tier, static SPA)
    │
    ├─ React Router routes: /, /transactions, /transactions/:id, /alerts
    │
    ▼
src/lib/api.ts  (demo-aware fetch wrapper)
    │
    ├─ if VITE_DEMO_MODE === "true": resolve from /demo-data/*.json
    │                                 (bundled in the Vite build)
    └─ else:                          axios → VITE_API_URL (local backend)
```

Local development is unaffected: omitting `VITE_DEMO_MODE` keeps
the live-backend path. The same React tree, the same TanStack
Query hooks, the same UI — only the leaf-level fetch function
branches.

## In-Scope Pages and Interactions

Every item below must work end-to-end on the static demo. If it
can't, the page is removed from the public navigation, not shipped
half-broken.

### `/` — Dashboard Overview

| Capability | Demo behaviour |
|---|---|
| Five KPI tiles (transactions 24h, fraud rate, blocked, review queue, avg score) | Snapshot values from the moment of export, rendered identically. |
| 24h fraud-rate line chart | Fully interactive (Recharts hover tooltips work on static data). |
| Per-hour volume sparkline | Same. |
| Top-10 country breakdown table | Same. |
| Per-tile loading skeletons / error fallbacks | Not exercised — data is bundled. The states stay in the code (local mode still uses them) but never render in demo mode. |
| Polling refresh (30s / 60s) | **Disabled in demo mode.** The query hooks set `refetchInterval: false` when `VITE_DEMO_MODE === "true"` so we don't spam the CDN for unchanging files. |

### `/transactions` — Transactions List

| Capability | Demo behaviour |
|---|---|
| Filter chrome (decision, country, amount, time window, customer, merchant) | Filters operate **client-side** against the bundled snapshot. Same URL-sync behaviour, same chip UX. |
| Keyset pagination ("Load more") | Snapshot ships the first **300 rows** (≈10 pages of 30) sorted as the backend would sort them. "Load more" walks the in-memory array; when exhausted, the footer says *"End of demo snapshot — clone the repo for the full 50k feed."* |
| Per-row deep links to `/transactions/:id` | Work only for IDs included in the detail snapshot (see below). Other rows are still listed but their detail page shows a friendly "not in snapshot" notice. |
| Color-coded decision pills, sortable columns | Same as live. |
| Empty / error / partial-failure states | Same code, not exercised in demo. |

### `/transactions/:id` — Transaction Detail

| Capability | Demo behaviour |
|---|---|
| Dual-badge header (model verdict + effective decision) | Snapshot. |
| SHAP bars with top contributors | Snapshot — pre-computed, rendered identically. |
| Rules-triggered chips | Snapshot. |
| Identity / channel grid | Snapshot. |
| Vertical audit timeline | Snapshot — includes any analyst-override rows captured at export time. |
| Analyst decision form (REVIEW rows) | **Disabled** with a tooltip: *"Analyst actions are available when running locally."* The form is rendered (so recruiters see it exists) but the submit button is disabled and the analyst-id capture modal is suppressed. |
| 404 for unknown IDs | Custom demo-mode message instead of a generic 404: *"This transaction is in the live system but not in the public snapshot."* |

### `/alerts` — Review Queue

| Capability | Demo behaviour |
|---|---|
| Summary strip (pending count, oldest age, score buckets) | Snapshot values, frozen at export time. |
| Filter bar (min score, country, age band) | Client-side over the snapshot. |
| Dense queue table with score chips | Snapshot — top **100 pending rows** at export time. |
| "Queue clear" celebratory state | Not reachable in demo (snapshot always has rows). Kept in code. |
| Polling (10s stale / 15s poll) | **Disabled** in demo mode (same rationale as dashboard). |

### Global UI

| Capability | Demo behaviour |
|---|---|
| Sidebar navigation | All four links live. |
| Sidebar footer "Phase 3H · live" | Changes to "Demo · snapshot from {DATE}" in demo mode. |
| **Top demo banner** | New component, rendered only when `VITE_DEMO_MODE === "true"`. One-line, dismissible, copy: *"Demo mode — snapshot from {DATE}. [View source on GitHub] · [Watch the 2-minute walkthrough]"* |

## Out-of-Scope (Loom-Only)

Captured in the Phase 4E walkthrough, not on the public URL:

- The simulator continuously generating transactions and the
  dashboard KPIs ticking in real time.
- Submitting an analyst verdict and watching the row pop off the
  alerts queue without a refresh.
- The `X-Analyst-Id` capture modal flow.
- Idempotency replay behaviour on `POST /transactions`.
- The `/health` endpoint and CORS posture.
- The training pipeline, the model card, and the SHAP plots —
  shown by scrolling the GitHub repo, not by interacting with the
  demo.

## Out-of-Scope (Local-Setup-Only)

- All `POST` / `PATCH` endpoints. The demo is read-only.
- The simulator CLI. Not bundled.
- The Alembic migration runner. Not relevant — no DB.
- Anything that depends on the SQLite file (`fraud_radar.db`) at
  runtime.

## Snapshot Contract

Phase 4B will produce exactly the following files under
`frontend/public/demo-data/`. Filenames are stable so the Phase 4C
client can `import` them directly:

| File | Source endpoint | Cardinality |
|---|---|---|
| `stats-overview.json` | `GET /api/v1/stats/overview` | single object |
| `stats-timeseries.json` | `GET /api/v1/stats/timeseries?hours=24` | single object (24 buckets) |
| `stats-breakdown.json` | `GET /api/v1/stats/breakdown?limit=10` | single object (top-10) |
| `transactions.json` | `GET /api/v1/transactions?limit=300` (no filters) | array of 300 rows + `next_cursor: null` |
| `transactions/{id}.json` | `GET /api/v1/transactions/{id}` | one file per ID, ~30 IDs chosen for coverage (see below) |
| `alerts.json` | `GET /api/v1/alerts?limit=100` | single envelope (summary + 100 rows) |
| `manifest.json` | Generated by the export script | `{ exported_at, dataset_seed, row_counts, schema_version: "1" }` |

**Detail-page ID selection (curated for demo coverage):**

- 10 `BLOCK` rows spanning the six fraud-injection patterns.
- 10 `REVIEW` rows — all overlapping with `alerts.json` so any
  alert row click works.
- 8 `ALLOW` rows including at least 2 high-score allows (to show
  the SHAP UI on borderline calls).
- 2 rows with an analyst override already applied (to demonstrate
  the dual-badge + audit-timeline behaviour).

The total snapshot footprint is bounded under **~2 MB** gzipped.
Vercel serves it from the CDN; no lazy loading needed.

## Demo-Mode Honesty Posture

Three principles, locked here so 4C doesn't drift:

1. **State it once, clearly, at the top.** A single banner with
   the snapshot date and links to the repo and the Loom. No
   apologies, no "limited demo" language — it's a snapshot, that's
   a normal thing.
2. **Disable, don't fake.** Write actions are visibly disabled
   with a tooltip pointing to the local-setup path. Never let a
   button silently no-op or pretend to succeed.
3. **No fake real-time.** Polling is off. Tiles don't tick.
   Trying to fake live updates from static data is the fastest way
   to look amateur. The Loom shows the live behaviour; the demo
   shows the UI.

## Routing / Hosting Notes

- `vercel.json` must contain an SPA rewrite so refreshing
  `/transactions/abc123` doesn't 404:
  ```json
  { "rewrites": [{ "source": "/(.*)", "destination": "/" }] }
  ```
- The Vite build receives `VITE_DEMO_MODE=true` and
  `VITE_DEMO_SNAPSHOT_DATE=YYYY-MM-DD` at build time via Vercel
  project env vars. No runtime config fetch.
- The snapshot files live under `frontend/public/demo-data/`,
  which Vite copies verbatim into the build output, so they're
  reachable as `/demo-data/*.json` at runtime.

## Test Plan

Phase 4A ships no code, so its "tests" are checklists the later
phases must satisfy:

- [ ] `VITE_DEMO_MODE=true npm run build && npm run preview`
      renders all four routes without a single network request
      to `http://localhost:8000`.
- [ ] Every entry in the in-scope tables above is manually
      walked through on the preview build before Vercel deploy.
- [ ] Every entry in the out-of-scope tables is verified to be
      either disabled with a tooltip or absent from the UI —
      never silently broken.
- [ ] `manifest.json` is fetched and displayed in the footer
      tooltip so the snapshot date in the banner can be
      cross-checked.
- [ ] Refreshing `/transactions/{id}` and `/alerts` on the
      deployed Vercel URL returns the page, not a 404.

## Decision Summary

| Decision | Choice | Reason |
|---|---|---|
| Hosting model | Static frontend on Vercel, no backend | Only path that is free, zero-cold-start, and zero-maintenance for an ML-heavy stateful stack. |
| Snapshot format | Plain JSON in `frontend/public/demo-data/` | Bundled with the build, served from the CDN, no extra fetch infra. |
| Demo-mode switch | Build-time `VITE_DEMO_MODE` env var | Keeps the live-backend dev loop untouched; one branch at the fetch layer. |
| Transactions snapshot size | 300 rows | Big enough to make filters and pagination feel real, small enough to ship in the bundle. |
| Detail-page coverage | ~30 curated IDs | Enough to show every UI variant (BLOCK / REVIEW / ALLOW / overridden) without bloating the bundle. |
| Polling | Disabled in demo mode | Static data + CDN caching make polling pointless and wasteful. |
| Write actions | Visibly disabled, never faked | Honesty beats theatre. |
| Banner copy | "Demo mode — snapshot from {DATE}. [GitHub] · [Loom]" | One line, factual, links out. |
