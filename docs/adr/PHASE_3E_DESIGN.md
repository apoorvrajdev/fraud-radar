# Phase 3E Design — Dashboard Overview Endpoints + Frontend Foundation

> Companion to [PHASE_3_DESIGN.md](PHASE_3_DESIGN.md) and
> [PHASE_3C_INTEGRATION.md](PHASE_3C_INTEGRATION.md). Captures the
> decisions taken before writing any Phase 3E code so the
> implementation has a target to hit.

> Status: design. Author: apoorvrajdev. Locked before implementation.

## Context

Phases 3A–3D shipped the full backend pipeline: rules engine,
idempotent ingestion, end-to-end scoring with audit log, and the
background simulator that feeds continuous synthetic traffic at a
dashboard-friendly rate. The simulator's whole point is to keep the
database lively for Phase 3E onward, but until 3E lands there is no
UI that can actually surface that traffic.

Phase 3E delivers two things, in this order:

1. The aggregate read endpoints the dashboard needs (`/stats/*`),
   measured against the same SQLite the scoring pipeline writes to.
2. The frontend foundation — TanStack Query client, axios API client,
   router, app shell — plus the dashboard overview page itself with
   live KPI tiles, a 24-hour fraud-rate chart, and a top-country
   breakdown.

The frontend stays intentionally small: one page, five tiles, two
charts, one table. Transactions list (3F), transaction detail (3G),
and alerts queue (3H) are explicitly out of scope for this phase.

## Goal

By end of Phase 3E, `npm run dev` against a running backend +
simulator shows a dashboard that updates every 30 seconds with KPIs
drawn from real scored transactions, and `uv run pytest -v` stays
green with new coverage of the stats repository, service, and
endpoints.

---

## API surface

Three new GET endpoints under `/api/v1/stats`. All are read-only,
unauthenticated (auth is a Phase 4 nice-to-have), and return Pydantic
schemas that already exist in [`backend/app/schemas/stats.py`](../../backend/app/schemas/stats.py).

### GET /api/v1/stats/overview

Top-line KPIs over a rolling 24-hour window ending `now()`.

Response 200 — `StatsOverview`:

```json
{
  "total_transactions_24h": 86_400,
  "approved_count_24h": 82_100,
  "declined_count_24h": 3_100,
  "pending_review_count": 1_200,
  "approved_rate": 0.9502,
  "fraud_caught_amount": "412350.18",
  "avg_fraud_score": 0.0412
}
```

- `pending_review_count` counts rows currently sitting at
  `Decision.REVIEW` regardless of age — it represents the analyst
  queue depth, not a 24h slice.
- `fraud_caught_amount` is the Decimal sum of `amount` over rows
  whose decision is `DECLINE` or `REVIEW` in the 24h window.
- `avg_fraud_score` is `None` when the window has zero rows.

### GET /api/v1/stats/timeseries?window=24h&bucket=1h

Bucketed fraud-rate + volume timeseries for charting.

Response 200 — `StatsTimeseries`:

```json
{
  "window": "24h",
  "points": [
    {"timestamp": "2026-05-24T01:00:00Z", "transaction_count": 3_412, "fraud_rate": 0.018},
    {"timestamp": "2026-05-24T02:00:00Z", "transaction_count": 3_517, "fraud_rate": 0.022},
    ...
  ]
}
```

- 24 points for `window=24h`, one per hour bucket aligned to the top
  of the hour in UTC.
- Empty buckets emit a point with `transaction_count=0` and
  `fraud_rate=0.0` so the chart's x-axis is continuous.
- For Phase 3E only `window=24h` and `bucket=1h` are accepted.
  Anything else returns 422. Wider windows arrive with 3F.

### GET /api/v1/stats/breakdown?dimension=country

Top-10 dimension values by transaction count over the same 24h
window.

Response 200 — `StatsBreakdown`:

```json
{
  "dimension": "country",
  "items": [
    {"category": "US", "transaction_count": 51_200, "declined_count": 1_800, "total_amount": "9821110.55"},
    {"category": "GB", "transaction_count": 8_400, "declined_count": 410, "total_amount": "1640220.10"},
    ...
  ]
}
```

- Only `dimension=country` is accepted in 3E. Merchant-category
  breakdown is in scope for 3F. The schema already supports both, so
  no migration required.
- Sorted by `transaction_count DESC`, capped at 10 rows.
- `category` field name is reused for the dimension value to stay
  schema-compatible; the frontend renames it for display.

### CORS

A `CORSMiddleware` is added to [`backend/app/main.py`](../../backend/app/main.py)
reading allowed origins from a new `settings.cors_origins` list,
defaulting to `["http://localhost:5173"]`. Methods limited to `GET`
and `POST` (matches the existing endpoint set), headers wildcarded,
credentials disabled. Phase 4 will tighten this for any deployed
environment.

---

## Repository + service layer

Following the existing layered convention (`api → service →
repository → models`):

### `backend/app/repositories/stats.py`

Three pure SQLAlchemy aggregate methods, each accepting `db: Session`
plus an injected `now: datetime` so tests freeze time without mocking
the clock:

- `overview(db, *, now, window=timedelta(hours=24)) -> StatsOverviewRow`
- `timeseries(db, *, now, window, bucket_minutes=60) -> list[TimeseriesRow]`
- `breakdown_by_country(db, *, now, window, limit=10) -> list[CountryRow]`

Bucketing uses `func.strftime('%Y-%m-%d %H:00:00', Transaction.created_at)`
which is SQLite-native; the equivalent Postgres expression is a
`date_trunc('hour', ...)` swap captured as a TODO in the file. The
README's "Why SQLite" decision already calls out that this is the
kind of seam we accept for the demo.

### `backend/app/services/stats.py`

Thin orchestration: converts repository rows into the Pydantic
response schemas, handles the `avg_fraud_score=None` empty case,
quantises `Decimal` to 2 dp, fills empty time buckets to keep the
chart continuous.

No business logic worth testing lives in the router. The router is
three one-liners that call the service and return the schema.

---

## Frontend architecture

```
frontend/src/
├── App.tsx                 # <Routes> — currently splash, becomes router root
├── main.tsx                # QueryClientProvider + BrowserRouter wrapping
├── lib/
│   ├── api.ts              # axios instance, baseURL from VITE_API_URL
│   ├── queryClient.ts      # TanStack QueryClient defaults
│   └── cn.ts               # clsx + tailwind-merge helper
├── types/
│   └── api.ts              # mirror of backend Pydantic schemas
├── hooks/
│   ├── useStatsOverview.ts
│   ├── useStatsTimeseries.ts
│   └── useStatsBreakdown.ts
├── components/
│   ├── layout/
│   │   ├── AppShell.tsx    # sidebar + topbar wrapper
│   │   └── Sidebar.tsx     # nav links (3F/3H show as disabled)
│   ├── ui/
│   │   ├── Card.tsx
│   │   └── Stat.tsx
│   └── dashboard/
│       ├── KpiTiles.tsx
│       ├── FraudRateChart.tsx
│       ├── VolumeSparkline.tsx
│       └── CountryBreakdownTable.tsx
└── pages/
    └── DashboardPage.tsx   # composes the above
```

### Why TanStack Query polling instead of WebSocket push

The simulator emits 1 tx/sec by default. A 30-second
`refetchInterval` shows the dashboard ticking visibly without the
operational surface area of a websocket route, connection-lifecycle
handling, reconnect logic, or message-schema versioning. WebSocket
push lives in "What I'd build next" in the README and is the natural
Phase 4 upgrade. Polling for 3E keeps the slice small and the
demo-ability identical for a reviewer watching the dashboard for a
minute.

### Why `axios` over `fetch`

`axios` is already a dependency. Centralised baseURL config, request
interceptor stubs for future auth headers, and consistent error
shape. No measurable bundle cost at this size.

### Type sharing

Types are hand-mirrored in `frontend/src/types/api.ts`. No codegen
pipeline (e.g. openapi-typescript) for Phase 3E because the schema
surface is small and stable; introducing codegen would expand the
toolchain for one page's worth of types. Phase 4 polish is the right
time to revisit if the surface grows.

### `Decimal` handling

Money values cross the wire as strings (Pydantic v2 default for
`Decimal`). The frontend treats them as `string` in the type layer
and formats with `Intl.NumberFormat` — never parsed to `Number` to
avoid losing precision. This mirrors the backend's "no float for
money" discipline.

---

## Decisions locked before implementation

### 1. Time window is fixed at 24h for Phase 3E

The overview and timeseries endpoints both hardcode the 24h window.
A `window` query param exists on `/timeseries` for forward
compatibility, but only `24h` is accepted. Wider windows arrive with
3F when the transactions list needs them for date-range filters.

**Not chosen:** parametric window now. Premature — no concrete UI
consumer for 7d/30d until 3F.

### 2. Country-only breakdown for Phase 3E

`/stats/breakdown` accepts `dimension=country` only. The schema
already supports `category`, but enabling it now would mean
designing a second visual treatment with no concrete UI need yet.
3F's transactions list will introduce the merchant-category
breakdown once that filter exists.

### 3. Polling, not WebSockets

Justified above. The 30-second `refetchInterval` is set in each hook
explicitly so it can be tuned per-tile (e.g. KPI tiles at 30s,
timeseries at 60s) without a global default surprising anyone.

### 4. No new database migration

All three endpoints are pure reads over existing tables
(`transactions`). No schema change, no new index for Phase 3E. If
the rolling-24h aggregates show up in latency profiling later, the
fix is a covering index on `(created_at, decision)`, captured as a
TODO comment on the repository methods rather than shipped
speculatively.

### 5. CORS is environment-driven

`settings.cors_origins` is a list with a dev-only default. Phase 4
deployment will set this via environment variable. No wildcard
origins, ever, even in dev — explicit list keeps the security
posture honest from the start.

### 6. The dashboard page tolerates partial failures

Each tile / chart fetches independently. A 500 from `/stats/breakdown`
must not blank the KPI tiles or the chart. Every component renders
its own loading skeleton + error fallback. This mirrors how a real
fraud console behaves: one broken aggregate never blacks out the
whole screen.

---

## Slicing + commit plan

The phase ships in three slices, one commit each, with a final docs
commit to mark the roadmap:

1. **3E-1** — `feat(api): add /stats endpoints for dashboard KPIs`
   - repository + service + router + tests + CORS middleware
   - target: pytest count rises from 178 to ~190+
2. **3E-2** — `feat(frontend): add API client, router shell, and layout`
   - `lib/`, `types/`, `components/layout`, `components/ui`,
     `QueryClientProvider`, `BrowserRouter`, `.env.example`
   - target: `npm run build` succeeds; visiting `/` shows the shell
     with empty content area
3. **3E-3** — `feat(frontend): add dashboard overview with live KPIs`
   - `hooks/`, `components/dashboard/`, `pages/DashboardPage.tsx`
   - target: with backend + simulator running, KPIs visibly update
     every 30s
4. **3E-4** — `docs(readme): mark Phase 3E complete`

Per the repo's commit policy (see [`CLAUDE.md`](../../CLAUDE.md)), all
commits are authored by apoorvrajdev and the assistant never stages,
commits, or pushes — only suggests messages.

---

## Out of scope

- Authentication, RBAC, session management — Phase 4 nice-to-have.
- WebSocket push for live updates — captured in the README's "What
  I'd build next" section.
- Dark/light theme toggle — frontend ships dark-only for 3E,
  matching the splash screen.
- Per-merchant breakdown — needs the merchant detail UI from 3F to
  link out to.
- Storybook / component catalogue — out of scope for a four-day
  build.
- Codegen for TypeScript types — see "Type sharing" above.

---

## Verification checklist

Before opening any commit for review:

1. `cd backend && uv run pytest -v` — green; new tests under
   `tests/unit/test_stats_service.py` and
   `tests/integration/test_stats_endpoint.py`.
2. `cd backend && uv run ruff check . && uv run mypy app` — clean.
3. `cd frontend && npm run build` — succeeds with zero TS errors.
4. With `uvicorn app.main:app` + `python -m app.simulator.main`
   running, visiting `http://localhost:5173` shows KPI tiles whose
   numbers move within a minute of observation.
5. Killing the backend mid-session shows error fallbacks on each
   tile, not a blank page.
