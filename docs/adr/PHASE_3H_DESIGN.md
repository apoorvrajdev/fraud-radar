# Phase 3H Design — Alerts Queue (Analyst Worklist)

> Companion to [PHASE_3_DESIGN.md](PHASE_3_DESIGN.md),
> [PHASE_3C_INTEGRATION.md](PHASE_3C_INTEGRATION.md),
> [PHASE_3E_DESIGN.md](PHASE_3E_DESIGN.md),
> [PHASE_3F_DESIGN.md](PHASE_3F_DESIGN.md), and
> [PHASE_3G_DESIGN.md](PHASE_3G_DESIGN.md). Captures the decisions
> taken before writing any Phase 3H code so the implementation has
> a target to hit.

> Status: design. Author: apoorvrajdev. Locked before implementation.

## Context

Phase 3G closed the analyst-review loop on a single transaction:
the detail page renders the model's verdict, the rules that fired,
the top SHAP contributors, the audit trail, and — for `REVIEW`
rows — an analyst decision form that writes `analyst_label`,
`analyst_notes`, and `reviewed_at`. It does not, however, tell an
analyst **which** transactions are waiting on them. Today the only
way to find a pending review is to open the Transactions page,
filter by `decision = REVIEW`, and scroll.

Phase 3H replaces that ad-hoc workflow with a purpose-built
worklist. It is the page an analyst lives on for a shift: queue
depth at the top, the riskiest unhandled row at the top of the
table, one click into the detail view, and — thanks to 3G-4 — a
review form that pops the row off the queue when submitted. The
sidebar's `Alerts` link, which has rendered as a disabled `3H`
placeholder since Phase 3E, becomes live.

This is the last frontend slice of Phase 3. After it, the demo
loop is closed end-to-end: simulator emits transactions → scoring
pipeline labels them → the Overview shows aggregate health → the
Transactions list explores history → the Alerts queue routes
analysts to the rows that need a human → the Transaction detail
page captures the human verdict.

## Goal

By end of Phase 3H:

1. `GET /api/v1/alerts` returns a paginated, filterable list of
   pending-review transactions (rows where
   `fraud_decision = 'REVIEW' AND analyst_label IS NULL`), sorted
   by `fraud_score DESC, created_at ASC` by default — riskiest
   first, with age as the tiebreaker so old rows do not get
   stranded behind a flood of fresh high-score arrivals. The
   predicate intentionally excludes `PENDING` (idempotency-conflict
   rows that have not been scored yet — they do not belong in a
   human queue) and `NULL` `fraud_decision` (seed rows that never
   ran through scoring); both are handled by exact equality.
2. The same envelope carries a `summary` block with queue health:
   total pending count, oldest pending age in seconds, score
   distribution buckets. The frontend renders that block as a
   small header strip above the queue.
3. Visiting `http://localhost:5173/alerts` against a running
   backend shows the queue page: header strip with the four
   summary stats, a filter bar (score floor, country, age band),
   and a dense table where each row is one click away from
   `/transactions/:id`.
4. Submitting an analyst decision from the detail page (the
   `useSubmitAnalystDecision` mutation already invalidates
   `["alerts"]`) makes the row disappear from the queue on the
   next refetch without manual refresh.
5. `uv run pytest -v` stays green with new coverage of the alerts
   endpoint (queue filter, summary math, sort order, pagination,
   filter validation).
6. `npm run build` and `npm run lint` succeed with zero TS errors.

---

## API surface

One endpoint. Phase 3G already covers the write surface via
`POST /transactions/{id}/decision`; the queue page is read-only
plus the existing override flow.

### GET /api/v1/alerts — paginated pending-review queue

**Path:** `GET /api/v1/alerts`

**Why a dedicated endpoint and not `GET /transactions?decision=REVIEW`:**

- The two views answer different questions. `/transactions` is a
  historical browser sorted by recency. `/alerts` is a worklist
  sorted by risk and constrained to *un-reviewed* REVIEW rows.
- Reusing `/transactions` would force the frontend to combine
  `decision=REVIEW` with an `analyst_label_is_null=true` flag
  that does not exist on the 3F query schema. Adding that flag
  would muddy the 3F semantics ("any REVIEW row, reviewed or
  not") to serve a single caller.
- `/alerts` carries a `summary` block (queue depth, oldest age,
  score buckets). That math has no business living on the
  transactions list endpoint where the same payload would be
  computed and discarded by every other caller.
- A separate URL gives the frontend a clean query-key namespace
  (`["alerts", filters]`) and lets the mutation invalidate the
  queue surgically without forcing a refetch of `/transactions`.

**Query parameters:**

| Param | Type | Default | Notes |
| --- | --- | --- | --- |
| `limit` | int (1–200) | 50 | Page size. |
| `cursor` | str | — | Opaque keyset cursor from the prior page; reuses the same `urlsafe_b64(score_str + "|" + created_at_iso + "|" + id)` format as 3F, adapted to the alerts sort. |
| `min_score` | Decimal (0–1) | — | Lower bound on `fraud_score` (inclusive). |
| `country` | str (2 chars) | — | ISO 3166-1 alpha-2. |
| `min_age_seconds` | int (≥0) | — | Only include rows where `now() - created_at >= min_age_seconds`. Lets analysts surface stale rows. |
| `max_age_seconds` | int (≥0) | — | Only include rows where `now() - created_at <= max_age_seconds`. Mostly useful for "in the last hour" views. |

Cross-field validation (`min_age_seconds <= max_age_seconds` when
both are set) is enforced in the Pydantic model with a
`model_validator` so the router stays a one-liner.

**Sort.** Fixed to `fraud_score DESC, created_at ASC, id ASC`.
This is a worklist, not a configurable view. Score-desc surfaces
risk; created_at-asc tiebreaks so old rows are not buried by a
storm of fresh high-score arrivals; id-asc tiebreaks the (rare)
identical-score+timestamp case to keep keyset pagination stable.

**Response (`AlertsResponse`):**

```jsonc
{
  "summary": {
    "pending_count": 128,            // total rows matching the queue predicate (not the filters)
    "oldest_pending_seconds": 18342, // age of the single oldest pending row, or null if queue is empty
    "score_buckets": {               // distribution of pending rows by score band
      "0.40_0.60": 47,
      "0.60_0.80": 58,
      "0.80_1.00": 23
    }
  },
  "items": [
    {
      "id": "70f98b5c-…",
      "created_at": "2026-05-27T14:54:11Z",
      "age_seconds": 412,            // computed at response time
      "amount": "7770.39",
      "currency": "USD",
      "country": "US",
      "customer_id": "ad6af8f0-…",
      "merchant_id": "13a707ea-…",
      "fraud_score": "0.7142",
      "fraud_decision": "REVIEW",
      "rules_triggered": ["high_amount", "off_hours"]
    }
  ],
  "next_cursor": "…",
  "has_more": true
}
```

**Status codes:**

| Status | Meaning |
| --- | --- |
| `200 OK` | Page returned. Empty `items` is a valid response, not an error. |
| `422 Unprocessable Entity` | Malformed cursor, bad filter combination, or out-of-range value. |

**Why the `summary` block ignores the filters.** Queue health is a
property of the *queue*, not of the analyst's current view. If an
analyst filters to `min_score=0.9` and the count drops to 4, the
header should still show that 124 other rows are waiting — that is
the entire point of having a queue-health indicator. The filters
only constrain the visible page.

**Why a denormalized `age_seconds` per row.** The frontend already
has `created_at`. Computing age client-side is one line. But
shipping the server-rendered age makes "stale rows" visually
obvious without a clock dependency on the browser and matches the
SLA framing the summary block exposes. It is computed once per
response, not stored.

---

## Schemas

All new schemas live in a new `app/schemas/alerts.py` module so
the alerts surface is discoverable as a unit and the transactions
schema file does not grow another vocabulary it does not own.

### AlertsQuery

```python
class AlertsQuery(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    limit: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None
    min_score: Decimal | None = Field(default=None, ge=Decimal("0"), le=Decimal("1"))
    country: str | None = Field(default=None, min_length=2, max_length=2)
    min_age_seconds: int | None = Field(default=None, ge=0)
    max_age_seconds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_age_range(self) -> Self:
        if (
            self.min_age_seconds is not None
            and self.max_age_seconds is not None
            and self.min_age_seconds > self.max_age_seconds
        ):
            raise ValueError("min_age_seconds must be <= max_age_seconds")
        return self
```

### AlertItem

```python
class AlertItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    age_seconds: int
    amount: Decimal
    currency: str
    country: str
    customer_id: str
    merchant_id: str
    fraud_score: Decimal
    fraud_decision: Literal["REVIEW"]
    rules_triggered: list[str] = Field(default_factory=list)
```

### AlertsSummary

```python
class AlertsSummary(BaseModel):
    pending_count: int
    oldest_pending_seconds: int | None
    score_buckets: dict[str, int]  # keys: "low", "mid", "high"
```

Score buckets use stringly-typed keys (not nested objects) so the
JSON stays trivial to consume in TypeScript without a generated
union type per bucket.

Bucket boundaries, fixed in code and documented here:

| Key | Range | Meaning |
| --- | --- | --- |
| `low` | `fraud_score < 0.20` | Rules-flagged but the ML model thinks the row is safe — leans false-positive. |
| `mid` | `0.20 <= fraud_score < 0.50` | Moderate ML signal plus a rule trip — the bulk of an analyst's day. |
| `high` | `fraud_score >= 0.50` | Strong ML signal that nonetheless landed under the auto-decline threshold. Surface first. |

> **Pre-implementation review (2026-05-27):** the original ADR
> proposed `0.40_0.60 / 0.60_0.80 / 0.80_1.00` buckets aligned
> visually to the auto-decline threshold (`0.7431` per
> `ml/artifacts/threshold.json`). A live-DB sanity check showed
> the actual REVIEW-row score distribution is `min=0.0002,
> max=0.6839, avg=0.0353` across 181 unlabeled rows — REVIEW is
> driven by the rules engine, not by a score band near threshold,
> so anything `>= 0.7431` is auto-declined and never enters the
> queue. The original buckets would have rendered the `0.80_1.00`
> bucket as a perma-zero and silently dropped most rows below
> `0.40`. The revised `low / mid / high` boundaries above are
> chosen against the empirical distribution and sum exactly to
> `pending_count` (no overflow, no silent drop).

### AlertsResponse

```python
class AlertsResponse(BaseModel):
    summary: AlertsSummary
    items: list[AlertItem]
    next_cursor: str | None = None
    has_more: bool = False
```

---

## Cursor format

Follows the same urlsafe-b64 JSON envelope **pattern** as 3F, but
the 3F encoder (`encode_cursor` in `app/services/transactions.py`)
is hardcoded to a `(ts, id)` keyspace with no parameterization, so
3H ships its own `encode_alert_cursor(score, ts, id)` /
`decode_alert_cursor` pair in `app/services/alerts.py` rather than
refactoring 3F's encoder for a single new caller. The encoded
tuple is `(fraud_score, created_at, id)` because the sort key is
score-first. Decode failures map to 422 with a "malformed cursor"
detail string. The cursor is signed only by construction (no
HMAC) — same posture as 3F, where the cursor is not
security-sensitive because there is no authorization to bypass.

> **Pre-implementation review (2026-05-27):** confirmed against
> the live SQLite DB. The 3F encoder is genuinely single-purpose;
> the ADR previously implied free reuse and has been corrected.

---

## Repository changes

`app/repositories/transaction.py` gets two new functions. Keeping
them on the existing transaction repo (rather than a new
`alerts.py` repo) reflects the underlying truth: alerts are a
filtered projection of the `transactions` table, not their own
entity. There is no `alerts` table and there will not be.

1. `list_pending_review(db, *, query: AlertsQuery, now: datetime) -> tuple[list[Transaction], str | None]`
   — applies the queue predicate (`fraud_decision = 'REVIEW' AND
   analyst_label IS NULL`), then the filters, then the keyset
   pagination on `(fraud_score DESC, created_at ASC, id ASC)`.
   Returns `(rows, next_cursor)`.

2. `pending_review_summary(db, *, now: datetime) -> AlertsSummary`
   — runs a single aggregate query: `COUNT(*)` for
   `pending_count`, `MIN(created_at)` for `oldest_pending_seconds`,
   and a `CASE`-grouped `COUNT(*)` for the score buckets. One round
   trip, no Python loops. SQLite-compatible (`CASE WHEN` works on
   both SQLite and Postgres).

The `score_buckets` query uses three explicit `CASE WHEN`
branches rather than `GROUP BY width_bucket(...)` so the same SQL
runs on SQLite. The bucket boundaries (`0.20`, `0.50`) are
presentation choices documented in the schema (and in the table
above), not policy. They are chosen against the empirical REVIEW
score distribution observed on the live DB so the `high` bucket
is a meaningful triage signal rather than a perma-zero, and so
the three buckets sum exactly to `pending_count` with no overflow
category needed.

## Service layer

A new `app/services/alerts.py` module owns the read flow:

```python
def list_alerts(db: Session, query: AlertsQuery, *, now: datetime | None = None) -> AlertsResponse:
    now = now or datetime.now(timezone.utc)
    summary = pending_review_summary(db, now=now)
    rows, next_cursor = list_pending_review(db, query=query, now=now)
    return AlertsResponse(
        summary=summary,
        items=[_to_item(row, now=now) for row in rows],
        next_cursor=next_cursor,
        has_more=next_cursor is not None,
    )
```

`_to_item` hydrates `rules_triggered` from the persisted JSON text
column (same helper the detail service already uses, lifted to a
shared spot if needed) and computes `age_seconds = int((now -
row.created_at).total_seconds())`.

The injectable `now` parameter is the same testability pattern
3G's review service uses — lets the tests pin a deterministic
clock without monkey-patching `datetime.now`.

---

## Performance and indexing

The queue predicate is `fraud_decision = 'REVIEW' AND
analyst_label IS NULL`. The existing
`ix_transactions_fraud_decision` index on `fraud_decision` covers
the first half of the predicate. SQLite will then filter the
matched rows on `analyst_label IS NULL` in memory, which is fine
at the dataset's scale (≤ a few thousand pending rows even in a
busy synthetic run).

A partial index (`WHERE fraud_decision = 'REVIEW' AND
analyst_label IS NULL`) would be the right call on Postgres for a
production-scale dataset. It is **out of scope here** — adding a
partial index would force an Alembic migration whose only payoff
is performance under load this demo will never see, and SQLite's
partial-index support has gotchas (the dialect emits it correctly
but the planner does not always prefer it on small tables). If
load testing in Phase 4 shows the predicate scanning a meaningful
fraction of the table, we add the partial index then.

Sort key `(fraud_score DESC, created_at ASC, id ASC)` does not
have a compound index. Adding one would require another migration
and again is out of scope. The page-size cap of 200 makes the
in-memory sort cost trivial at this dataset scale.

---

## Frontend

### Routing

A new route `/alerts` is added to the existing router:

```tsx
<Route path="/alerts" element={<AlertsPage />} />
```

The sidebar's `Alerts` entry (currently `disabled: true,
comingIn: "3H"`) becomes a live `NavLink` pointing at `/alerts`.

### Data hook

A new `useAlerts(query)` hook in `frontend/src/hooks/`:

- TanStack Query, key `["alerts", query]`.
- `staleTime: 10_000` so the queue does not refetch on every
  filter focus event but stays fresh enough to feel live during
  a triage session.
- `refetchInterval: 15_000` while the page is visible, off when
  the tab is hidden (TanStack handles the visibility check
  automatically).
- The 3G-4 `useSubmitAnalystDecision` mutation already calls
  `queryClient.invalidateQueries({queryKey: ["transactions"]})`.
  A one-line addition will also invalidate `["alerts"]` so the
  queue self-corrects without a refetch interval needing to fire.

A second hook `useAlertsSummary` is **not** added — the summary
block is part of the `useAlerts` response, so a separate hook
would just split one fetch into two and create a render race.

### Layout

`frontend/src/pages/AlertsPage.tsx`:

1. **Page header** — title "Alerts" + subtitle "Transactions
   waiting on an analyst verdict. Highest risk first; oldest
   ties first."
2. **Summary strip** — four stat cards built from the existing
   `KpiCard` component used on the Overview page:
   - "Pending" → `summary.pending_count`
   - "Oldest" → `formatAge(summary.oldest_pending_seconds)` (e.g.
     "5h 12m") with an empty-state dash if the queue is clear
   - "Strong ML signal" → `summary.score_buckets["high"]` (renamed
     from a previous "High risk" draft so the label is anchored to
     what the data can actually produce — REVIEW rows never reach
     the `>= 0.7431` auto-decline band, so `high` here means
     `>= 0.50`, just under threshold)
   - "Avg score" — derived client-side from the visible page as a
     gentle "you are looking at" indicator; the summary block
     itself does not need to carry it because the `high` bucket
     already conveys the queue-wide picture
3. **Filter bar** — three controls reusing the existing
   `FilterShell` pattern from 3F: min score (number input,
   0.00–1.00), country (text input, 2 chars), age band (a small
   `<select>` with options "Any age", "Older than 1h", "Older
   than 24h", "In the last hour" — wired to `min_age_seconds` and
   `max_age_seconds` underneath).
4. **Queue table** — a dense table:
   - Columns: Age · Score · Amount · Country · Rules · Customer
   - Score is rendered as a colored chip aligned to the same
     buckets the summary uses: rose at `>= 0.50` (high), amber
     at `0.20–0.50` (mid), neutral at `< 0.20` (low). Aligning
     the per-row chip to the header-strip bucketing keeps the
     visual vocabulary consistent across the page.
   - Rules cell shows the first two rule tags + a "+N more"
     pill if needed.
   - Each row is a `<Link to={`/transactions/${row.id}`}>` so
     clicking lands on the existing detail page where 3G-4's
     form takes over.
5. **Footer** — same "Load more" button + "N loaded · more
   available" caption pattern as the Transactions list. No
   infinite-scroll; explicit affordance is friendlier for a demo.
6. **Empty / error states** — when `pending_count === 0`, the
   table area renders a celebratory empty state ("Queue clear.
   No transactions are waiting on a review."). When the query
   errors, an inline retry card. When the page is loading for
   the first time, a skeleton matching the summary strip + 6
   skeleton rows.

### Keyboard shortcuts

Out of scope for 3H. The Phase 4 polish pass adds `j/k` to move
focus between rows and `Enter` to open the focused alert. Adding
it here would burn the slice's budget on accessibility plumbing
(focus rings, screen-reader announcements, route announcer
integration) without measurably improving the demo. The shortcut
hint sits on the Phase 4 backlog instead.

### URL state

The filter values are mirrored into the URL query string via the
same `useFiltersFromURL` helper Phase 3F introduced. That helper
already takes a schema-driven map; we extend it with the alerts
filter shape rather than forking. The page is shareable: paste a
URL with `?min_score=0.8&min_age_seconds=3600` and the queue
opens pre-filtered.

---

## Sidebar activation

The `Sidebar.tsx` `NAV_ITEMS` list currently has:

```tsx
{ to: "/alerts", label: "Alerts", icon: ShieldAlert, disabled: true, comingIn: "3H" },
```

Slice 3H-3 flips that to:

```tsx
{ to: "/alerts", label: "Alerts", icon: ShieldAlert },
```

The sidebar's `Phase 3G · live` footer becomes `Phase 3H · live`.
No further sidebar refactor needed — the disabled-item branch in
the existing render code stays in place for future "coming in
Phase X" entries.

---

## Tests

### Backend

`backend/tests/integration/test_alerts.py` (new):

- `GET /alerts` returns the envelope shape (summary + items +
  pagination) on a populated DB.
- Queue predicate is correct: rows where `fraud_decision != REVIEW`
  are excluded, and rows where `fraud_decision = REVIEW AND
  analyst_label IS NOT NULL` (reviewed REVIEW rows) are excluded.
- Sort order: `fraud_score DESC, created_at ASC, id ASC` across a
  hand-built fixture with overlapping scores.
- Pagination: page 1 + page 2 via cursor returns the full set
  with no duplicates and no gaps; bad cursor → 422.
- Filters: `min_score`, `country`, `min_age_seconds`,
  `max_age_seconds` each constrain `items` but leave `summary`
  unchanged.
- `age_seconds` is non-negative and monotonic with `created_at`.
- `summary.pending_count` matches a direct `COUNT(*)` query on
  the queue predicate.
- `summary.oldest_pending_seconds` matches the age of the
  oldest pending row.
- `summary.score_buckets` math: a fixture with a known
  distribution returns the expected bucket counts.
- Empty queue: `pending_count = 0`, `oldest_pending_seconds = None`,
  buckets all zero, `items = []`, `has_more = false`.

`backend/tests/unit/test_alerts_service.py` (new):

- `_to_item` produces the right `age_seconds` for an injected
  `now`.
- Score-bucket boundary cases (`0.40`, `0.60`, `0.80`, `1.00`)
  land in the documented bucket.

### Frontend

No new unit tests — same posture as 3G. Phase 4 introduces the
Cypress / Playwright pass. Phase 3H gets a manual-checklist entry
in the README "How to demo this" section covering: queue loads,
clicking a row lands on the detail page, submitting a decision
removes the row from the queue, the summary count drops by one.

---

## Slicing

- **3H-1** — ADR (this file). Status: locked once committed.
- **3H-2** — Backend: `app/schemas/alerts.py`, repository
  additions on `app/repositories/transaction.py`,
  `app/services/alerts.py`, `app/api/v1/alerts.py` router, wiring
  into `app/api/v1/__init__.py`, full backend test coverage.
- **3H-3** — Frontend: `useAlerts` hook, `AlertsPage` with header
  strip + filter bar + queue table + empty/error states, sidebar
  activation, alerts-cache invalidation hook added to
  `useSubmitAnalystDecision`.
- **3H-4** — Polish: queue empty state copy + score-chip color
  ramp tuning + sidebar footer bump + README sync (Phase 3H
  complete, Phase 3 closed, Phase 4 surfaced as next).

---

## Out of scope

- Real-time push (SSE / WebSocket) for queue updates. The
  10-second stale time + 15-second poll is the right tradeoff
  for a demo. Phase 5 hardening.
- Multi-analyst claim / lock semantics ("this row is being
  worked on by X"). Phase 5.
- Bulk-action affordances (multi-select on the queue → bulk
  approve / decline). Phase 4 UX pass.
- An SLA-violation alert (red flag on rows older than N hours).
  The `oldest_pending_seconds` summary stat covers the demo
  story; per-row visual SLA flags land with bulk actions.
- Keyboard navigation (`j/k`, `Enter`). Phase 4 polish.
- A "queue trend" sparkline showing pending count over time.
  Phase 4 — needs a small time-series store.
- A "similar pending rows" suggestion surface on the detail page
  driven by the queue. Phase 4 at the earliest.

---

## Commit policy

Same as every prior phase. Conventional Commits, one slice per
commit, attributed to `apoorvrajdev <apoorvrajmgr@gmail.com>`.
The agent suggests the commit script and never runs `git add`,
`git commit`, or `git push` on its own.
