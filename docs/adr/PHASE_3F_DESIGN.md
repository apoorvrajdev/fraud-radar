# Phase 3F Design — Transactions List with Filters

> Companion to [PHASE_3_DESIGN.md](PHASE_3_DESIGN.md),
> [PHASE_3C_INTEGRATION.md](PHASE_3C_INTEGRATION.md), and
> [PHASE_3E_DESIGN.md](PHASE_3E_DESIGN.md). Captures the decisions
> taken before writing any Phase 3F code so the implementation has a
> target to hit.

> Status: design. Author: apoorvrajdev. Locked before implementation.

## Context

Phase 3E shipped the dashboard overview — five KPI tiles, a 24h
fraud-rate line chart, a per-hour volume sparkline, and a top-10
country breakdown — driven by three read-only aggregate endpoints
and a TanStack Query polling loop. The Overview answers *"is
anything weird happening right now?"*

Phase 3F answers the next question an analyst would actually ask:
*"show me the rows."* It introduces the first paginated list endpoint
in the codebase and the first frontend page with non-trivial
interactive state (filters, URL-synced query parameters, infinite
scroll). The sidebar's Transactions link, which Phase 3E rendered as
a disabled `3F` placeholder, becomes live.

Transaction detail (3G) and alerts queue (3H) are explicitly out of
scope for this phase. The list page links *to* a detail route stub
but does not implement the detail page itself.

## Goal

By end of Phase 3F:

1. `GET /api/v1/transactions` returns paginated, filterable
   transaction rows with stable keyset cursors and 422-validated
   query params.
2. Visiting `http://localhost:5173/transactions` against a running
   backend + simulator shows a virtualised table that scrolls
   smoothly through tens of thousands of rows, with five filter
   controls whose state lives in the URL.
3. `uv run pytest -v` stays green with new coverage of the
   transactions repository keyset pagination, the filter logic, and
   the endpoint shape.
4. `npm run build` succeeds with zero TS errors.

---

## API surface

One new endpoint. Reuses the existing `Transaction` table and the
`TransactionResponse` schema for row shape.

### GET /api/v1/transactions

Paginated, filterable transactions list. Sort is fixed to
`created_at DESC, id DESC` for Phase 3F — configurable sort lands
with Phase 4 polish if needed.

**Query params:**

| Param          | Type                          | Default | Notes                                                            |
| -------------- | ----------------------------- | ------- | ---------------------------------------------------------------- |
| `limit`        | `int` (1–200)                 | 50      | Page size cap mirrors industry-standard list APIs.               |
| `cursor`       | `str` (opaque)                | none    | Returned by the previous page as `next_cursor`.                  |
| `decision`     | `APPROVE\|REVIEW\|DECLINE\|PENDING` | none | Single value; multi-decision filters defer to Phase 4.    |
| `country`      | ISO-3166 alpha-2 (e.g. `US`)  | none    | Single value.                                                    |
| `min_amount`   | `Decimal` ≥ 0                 | none    | Inclusive lower bound.                                           |
| `max_amount`   | `Decimal` ≥ 0                 | none    | Inclusive upper bound. Must be ≥ `min_amount` if both set.       |
| `start_time`   | ISO-8601 UTC                  | none    | Inclusive lower bound on `created_at`.                           |
| `end_time`     | ISO-8601 UTC                  | none    | Exclusive upper bound. Must be > `start_time` if both set.       |
| `customer_id`  | UUID                          | none    | Exact match.                                                     |
| `merchant_id`  | UUID                          | none    | Exact match.                                                     |

Invalid combinations return `422` with a Pydantic error body
describing which constraint failed. Unknown query params are
silently ignored (matches FastAPI's default).

**Response 200** — `TransactionList`:

```json
{
  "items": [
    {
      "id": "01J6...",
      "customer_id": "...",
      "merchant_id": "...",
      "amount": "412.50",
      "currency": "USD",
      "status": "APPROVED",
      "payment_method": "CARD",
      "country": "US",
      "fraud_score": "0.0421",
      "fraud_decision": "APPROVE",
      "created_at": "2026-05-27T18:42:10.123Z"
    }
  ],
  "next_cursor": "MjAyNi0wNS0yN1QxODo0MDow...",
  "has_more": true
}
```

- `items` reuses the existing `TransactionResponse` schema unchanged
  — no new row schema, no migration.
- `next_cursor` is `null` when the current page is the last one;
  `has_more` mirrors `next_cursor is not None` for ergonomic
  consumption from the frontend.

---

## Pagination — keyset, not offset

`OFFSET n` reads and discards `n` rows on every request; for a
50,000-row table that means the last page costs nearly 50,000 row
reads even though only 50 are returned. Keyset pagination scans
exactly `limit` rows per page regardless of position by using a
`WHERE` clause against the last seen sort key.

### Cursor encoding

The cursor is an opaque `urlsafe_base64(json.dumps({ts, id}))` blob.
Opaqueness matters because clients must not parse or construct
cursors themselves — that lets the server change the cursor shape
without a versioning conversation.

```python
def encode_cursor(ts: datetime, id_: str) -> str:
    payload = {"ts": ts.isoformat(), "id": id_}
    return urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
```

Malformed cursors return `422`, not `400` — same posture as every
other invalid query param.

### Query shape

```sql
SELECT *
FROM transactions
WHERE
  (filters...)
  AND (created_at, id) < (:cursor_ts, :cursor_id)  -- omitted on first page
ORDER BY created_at DESC, id DESC
LIMIT :limit + 1                                    -- +1 to detect `has_more`
```

`+1` row trick: fetch `limit + 1`, return the first `limit`, use
the `(limit + 1)`th row's existence to set `has_more`.

### Why tuple comparison instead of `OR`-decomposed conditions

`(created_at, id) < (?, ?)` is the standard keyset idiom. SQLite,
Postgres, and MySQL all support row-tuple comparison and can use a
composite index on `(created_at DESC, id DESC)` to satisfy it. The
hand-decomposed equivalent —
`(created_at < ? OR (created_at = ? AND id < ?))` — generates the
same plan but reads worse.

### Index plan

The existing `ix_transactions_created_at` index on `(created_at)`
already lets SQLite reverse-scan for the unfiltered keyset query.
For Phase 3F that is sufficient — the simulator's 50k-row dev
dataset returns the first page in single-digit milliseconds.

Production indexing notes captured as TODO comments on the
repository method, not shipped speculatively:

- Covering `(created_at DESC, id DESC)` for keyset scan.
- Partial `(decision)` for the high-selectivity `REVIEW` /
  `DECLINE` filters.
- `(customer_id, created_at DESC)` is already in place from Phase
  2A.

No Alembic migration in this phase.

---

## Repository + service layer

Following the existing layered convention (`api → service →
repository → models`):

### `backend/app/repositories/transaction.py` (extend)

Add a single method to the existing repository module — keep the
list logic colocated with the rest of `Transaction` CRUD.

```python
@dataclass(frozen=True)
class TransactionListFilters:
    decision: Decision | None = None
    country: str | None = None
    min_amount: Decimal | None = None
    max_amount: Decimal | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    customer_id: str | None = None
    merchant_id: str | None = None


def list_transactions(
    db: Session,
    *,
    filters: TransactionListFilters,
    limit: int = 50,
    cursor: tuple[datetime, str] | None = None,
) -> tuple[list[Transaction], tuple[datetime, str] | None]:
    """Returns (rows, next_cursor_or_None). Caller wraps in schema."""
```

Returning the cursor as a `tuple[datetime, str] | None` keeps
encoding logic out of the repository — the service does the base64
round-trip.

### `backend/app/services/transactions.py` (new)

Thin orchestration:

1. Decode the incoming `cursor` string into `(datetime, str)` (or
   `None`).
2. Call the repository.
3. Encode the returned cursor.
4. Build the `TransactionList` Pydantic response.

No business logic worth its own test file lives in the router. The
router is one function: validate, delegate, return.

### `backend/app/api/v1/transactions.py` (extend)

Add `GET /transactions` to the existing router. The endpoint
collects query params into a `TransactionListQuery` Pydantic model
that does cross-field validation (`min_amount ≤ max_amount`,
`start_time < end_time`), then calls the service.

---

## Frontend architecture

```
frontend/src/
├── App.tsx                          # add /transactions route
├── components/
│   └── transactions/                # new directory
│       ├── TransactionsTable.tsx    # virtualised table body
│       ├── TransactionsFilters.tsx  # filter sidebar / topbar
│       ├── FilterChips.tsx          # active-filter pills, click to clear
│       └── DecisionBadge.tsx        # APPROVE / REVIEW / DECLINE pill
├── hooks/
│   ├── useTransactionsList.ts       # useInfiniteQuery wrapper
│   └── useTransactionFilters.ts     # URL ↔ state bridge via useSearchParams
├── lib/
│   └── transactions.ts              # axios call + cursor serialisation helpers
├── types/
│   └── api.ts                       # add TransactionListItem + TransactionList types
└── pages/
    └── TransactionsPage.tsx         # composes filters + table
```

### Why `useInfiniteQuery` + "load more" instead of true infinite scroll

True infinite-scroll (IntersectionObserver auto-fetch) is the right
end-state but introduces three problems in a four-day build:

1. It hides the loading state behind a viewport intersection, making
   network errors invisible until the user scrolls back up.
2. It interacts badly with deep-link sharing — refreshing a URL with
   scroll-restored state lands users in unfetched territory.
3. It needs a virtualisation library to keep the DOM tractable.

A manual "Load more" button at the table footer covers 90% of the
analyst use-case (browse the most recent rows, occasionally page
back further), keeps loading state explicit, and skips the
virtualisation dependency. If user testing finds the button
annoying, swapping to auto-fetch is a 10-line `useEffect` change in
Phase 4.

### Why URL-synced filter state

Every filter selection updates `?decision=REVIEW&country=US&...` so
analysts can share a filtered view by pasting a URL. `useSearchParams`
from `react-router-dom` provides the bidirectional bridge.

Consequence: the filter component is *controlled* by the URL, not
by `useState`. The custom `useTransactionFilters` hook hides the
URL-encoding details so components stay readable.

### Why no row virtualisation in 3F

`react-window` and `react-virtual` are both excellent but each adds
~12 kB to the bundle and complicates the row markup. With a 50-row
default page size and the user explicitly clicking "Load more" to
extend, the DOM stays small enough for native scrolling to handle.
Virtualisation lands in Phase 4 polish if the analyst page grows to
default 200-row pages.

### Decimal display still goes through `lib/format.ts`

Same discipline as Phase 3E: `amount` and `fraud_score` arrive as
strings, stay as strings in the type layer, and only become `number`
inside the `Intl.NumberFormat` call. The `formatMoney` helper from
3E is reused; a new `formatFraudScore(value: string | null)` is
added for the 4-decimal display.

---

## Decisions locked before implementation

### 1. Keyset cursors, not offset pagination

Justified above. The cost difference is invisible at 50k rows on
SQLite but is the kind of decision that quietly breaks when the
table grows. Better to use the correct idiom from the start.

**Not chosen:** offset+limit. Industry-standard for *catalog* lists
where users often jump to page 50; wrong for a *log* like
transactions where users almost always scroll from the top.

### 2. Single-value filters in 3F

`decision=REVIEW` is supported; `decision=REVIEW,DECLINE` is not.
Multi-value filtering needs a SQL `IN` rewrite and a frontend
checkbox group with three-state semantics — both worth doing, both
out of scope for 3F. Captured as a sentence in the README's "What
I'd build next."

**Not chosen:** multi-select from day one. Premature — the analyst
flow this enables (triage a single decision class) is the most
common single use-case.

### 3. URL-synced filter state, not local component state

Sharability and refresh-resilience matter more than a marginal
typing-latency improvement. The cost is one extra `useEffect` per
filter change to write back to the URL — trivial at this size.

### 4. No new index, no migration

The existing `ix_transactions_created_at` covers the unfiltered
keyset query. The Phase 2A `ix_transactions_customer_created`
covers the `customer_id` filter. Other filters are low-volume
enough on the dev dataset to need no help. Production deployment
adds indexes as a Phase 4 task; shipping speculative indexes now
would make the migrations harder to review.

### 5. List endpoint returns the same row schema as `POST /transactions`

The `TransactionResponse` schema already exists and is the natural
shape for a list row. Reusing it keeps the API surface tight: there
is one shape for "a transaction in a response," and 3G's detail
page extends it via `TransactionDetail`.

**Not chosen:** a slimmer `TransactionListRow` with only
display-critical fields. The savings (a few hundred bytes per row)
do not justify a second shape an analyst tool's hands-on test
script would have to know about.

### 6. Filter validation is per-field-Pydantic plus cross-field model validator

`min_amount ≤ max_amount` and `start_time < end_time` are model-level
constraints, not field-level. A Pydantic `model_validator(mode="after")`
on `TransactionListQuery` encodes them, returning `422` on violation.

This mirrors the project's existing posture: invalid inputs become
4xx at the boundary, never deeper.

### 7. The frontend "Load more" button never disappears mid-pagination

If a network call fails on `Load more`, the button shows an inline
error message and the next click retries — it does not collapse the
already-rendered rows. This matches the Phase 3E posture: partial
failures degrade gracefully, never blank the screen.

---

## Slicing + commit plan

The phase ships in four slices, one commit each:

1. **3F-1** — `feat(api): add /transactions list endpoint with keyset pagination and filters (Phase 3F-1)`
   - Repository `list_transactions` + filters dataclass
   - Service `list_transactions` + cursor codec
   - Router `GET /transactions` with `TransactionListQuery` model
   - New `TransactionList` schema in `app/schemas/transaction.py`
   - Unit tests for cursor codec, filter validation, keyset edge cases
   - Integration tests for endpoint + 422 paths + pagination round-trip
   - Target: pytest count rises by ~15 tests.
2. **3F-2** — `feat(frontend): add transactions page route, filter state, and skeleton table (Phase 3F-2)`
   - `TransactionsPage`, `TransactionsFilters`, `FilterChips`,
     `useTransactionFilters` URL bridge, `useTransactionsList` hook
     with empty render
   - Sidebar `Transactions` link becomes live
   - Target: `/transactions` renders the filter chrome and a skeleton
     table; `npm run build` succeeds.
3. **3F-3** — `feat(frontend): wire transactions list to live data with load-more pagination (Phase 3F-3)`
   - `TransactionsTable` rows, `DecisionBadge`, per-tile loading +
     error fallbacks, "Load more" button, empty-state copy
   - Target: with backend + simulator running, the table populates
     and pagination works end-to-end.
4. **3F-4** — `docs(readme): mark Phase 3F complete`

Per the repo's commit policy, all commits are authored by
apoorvrajdev and the assistant never stages, commits, or pushes —
only suggests messages.

---

## Out of scope

- Sortable column headers — fixed `created_at DESC` for 3F. Sort
  configuration arrives if/when 3H's review queue needs a different
  ordering.
- Row-click navigation to transaction detail — 3G builds the detail
  page; 3F leaves the row clickable target as a stub `<Link>`
  pointing at `/transactions/:id` which renders a "coming in 3G"
  placeholder.
- Multi-value filters (e.g. `decision=REVIEW,DECLINE`) — Phase 4
  polish.
- Saved filter presets — Phase 4 polish.
- Bulk operations (mark-as-reviewed across selection) — belongs in
  Phase 3H alongside the alerts queue UX.
- CSV export — would be a useful Phase 4 demo touch but is not on
  the critical path for "show the rows."
- Row virtualisation — see "Why no row virtualisation" above.

---

## Verification checklist

Before opening any commit for review:

1. `cd backend && uv run pytest -v` — green; new tests under
   `tests/unit/test_transactions_repo.py` (or extension of an
   existing file) and `tests/integration/test_transactions_list_endpoint.py`.
2. `cd backend && uv run ruff check . && uv run mypy app` — no new
   error categories beyond the existing baseline.
3. `cd frontend && npm run build && npm run lint` — clean.
4. With `uvicorn app.main:app` + `python -m app.simulator.main`
   running, visiting `http://localhost:5173/transactions` shows a
   populated table that pages cleanly via "Load more" and respects
   each filter when set.
5. Killing the backend mid-session shows an inline error in the
   table area and leaves the filter chrome interactive.
6. Hand-craft a URL like
   `/transactions?decision=REVIEW&country=US&min_amount=100` and
   refresh — the page restores the filter state from the URL.
