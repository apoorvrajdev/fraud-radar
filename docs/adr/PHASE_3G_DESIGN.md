# Phase 3G Design — Transaction Detail with Fraud-Score Breakdown

> Companion to [PHASE_3_DESIGN.md](PHASE_3_DESIGN.md),
> [PHASE_3C_INTEGRATION.md](PHASE_3C_INTEGRATION.md),
> [PHASE_3E_DESIGN.md](PHASE_3E_DESIGN.md), and
> [PHASE_3F_DESIGN.md](PHASE_3F_DESIGN.md). Captures the decisions
> taken before writing any Phase 3G code so the implementation has a
> target to hit.

> Status: design. Author: apoorvrajdev. Locked before implementation.

## Context

Phase 3F shipped the transactions list — eight composable filters,
keyset pagination, color-coded decision pills, and a "Load more"
footer. Each row links to `/transactions/:id`, but that route
currently renders a stub page that just echoes the id and says
"detail view ships in Phase 3G".

Phase 3G replaces that stub with the page an analyst actually works
on. It is the slice where the SHAP attribution work from Phase 2H
finally reaches the screen, where the rules engine output stops
being a backend-only artifact, and where the `analyst_label`,
`analyst_notes`, and `reviewed_at` columns that have been sitting
unused on the `transactions` table since Phase 1 finally get
written to. It is also the first endpoint with a write surface
that mutates an existing row rather than appending a new one.

Alerts queue (3H) remains out of scope. The detail page does not
manage the queue — it just makes a single row inspectable and
actionable.

## Goal

By end of Phase 3G:

1. `GET /api/v1/transactions/{id}` returns a composite detail
   envelope: the transaction row, its scoring artifacts, customer
   and merchant identifiers, analyst fields, and the trailing
   audit-log entries for that resource.
2. `POST /api/v1/transactions/{id}/decision` lets an authenticated
   analyst commit a `CONFIRMED_FRAUD` or `CONFIRMED_LEGIT` label on
   a `REVIEW` row, with optional notes, idempotent on resubmit, and
   append-only on the audit log.
3. Visiting `http://localhost:5173/transactions/<id>` against a
   running backend shows a one-screen detail layout with the
   decision header, the rules-triggered list, the top-5 SHAP
   contributors as directional bars, the raw 17-feature table
   (collapsed by default), the audit history, and — for `REVIEW`
   rows only — an analyst decision form.
4. `uv run pytest -v` stays green with new coverage of the detail
   envelope, the decision endpoint (success, 404, 409 on terminal
   state, idempotent re-submit, audit-log row written), and the
   `X-Analyst-Id` header validation.
5. `npm run build` and `npm run lint` succeed with zero TS errors.

---

## API surface

Two endpoints. The first replaces the current `GET
/transactions/{id}` response shape; the second is new.

### GET /api/v1/transactions/{id} — composite detail envelope

The existing route currently returns `TransactionScored`
(score/decision/rules/contributors only). It is replaced with a
richer `TransactionDetail` envelope. **This is a deliberate
breaking change** to the response shape, justified by:

- No frontend code depends on the current envelope (the Phase 3F
  list page never calls it; the detail page is a stub).
- The POST `/transactions` ingestion endpoint keeps returning
  `TransactionScored`. That is the contract the simulator and any
  external producer rely on — that one does not change.
- Adding a second `/{id}/detail` endpoint would split a single
  resource across two URLs for no clear gain.

**Path:** `GET /api/v1/transactions/{id}`

**Response (`TransactionDetail`):**

```jsonc
{
  // transaction row
  "id": "…",
  "customer_id": "…",
  "merchant_id": "…",
  "amount": "123.45",
  "currency": "USD",
  "country": "US",
  "payment_method": "CARD",
  "is_card_present": true,
  "status": "APPROVED",
  "created_at": "2026-05-27T14:32:18Z",

  // scoring artifacts (from persisted columns, no recomputation)
  "fraud_score": "0.4217",
  "fraud_decision": "REVIEW",
  "threshold": "0.7431",            // pulled from artifacts at startup
  "rules_triggered": ["high_amount", "off_hours"],
  "top_contributors": [             // top-5 SHAP from `top_features` text col
    {"feature": "amount", "value": 9421.0, "shap": 1.83,  "direction": "fraud"},
    {"feature": "geo_velocity_kmh", "value": 0.0, "shap": -0.42, "direction": "legit"},
    …
  ],

  // analyst fields (nullable)
  "analyst_label": null,            // 'CONFIRMED_FRAUD' | 'CONFIRMED_LEGIT' | null
  "analyst_notes": null,
  "reviewed_at": null,

  // effective decision = analyst_label-derived if present, else fraud_decision
  "effective_decision": "REVIEW",   // 'APPROVE' | 'REVIEW' | 'DECLINE' | 'PENDING'

  // audit trail (most recent 20 entries for this resource_id)
  "audit": [
    {
      "id": 4421,
      "actor": "scoring-pipeline",
      "action": "INITIAL_DECISION",
      "payload": {"decision": "REVIEW", "score": 0.4217},
      "created_at": "2026-05-27T14:32:18Z"
    }
  ]
}
```

**Status codes:**

| Status | Meaning |
| --- | --- |
| `200 OK` | Detail envelope returned. |
| `404 Not Found` | No transaction with that id. |

The `top_contributors` shape matches what `_scored_from_transaction`
already produces from `transactions.top_features` — we add a
computed `direction` field on the way out (`"fraud"` if `shap > 0`,
`"legit"` if `shap < 0`) so the frontend does not duplicate that
classification logic.

**Why no live SHAP recomputation here.** The `/explain` endpoint
already exists for that. The detail envelope is a read of *what
was decided at scoring time*, which is what an audit log
fundamentally requires. Recomputing SHAP on every page load would
make decisions look like moving targets and burn CPU. The detail
page can offer a "see full SHAP plot" link that hits the existing
`/explain?format=force` PNG path; that is a Phase 3G stretch goal,
not a blocking deliverable.

### POST /api/v1/transactions/{id}/decision — analyst override

**Path:** `POST /api/v1/transactions/{id}/decision`

**Headers:**

| Header | Required | Notes |
| --- | --- | --- |
| `X-Analyst-Id` | yes | 1–64 chars; identifies the analyst for audit. No auth verification — see "Auth posture" below. |

**Body:**

```jsonc
{
  "label": "CONFIRMED_FRAUD",       // or "CONFIRMED_LEGIT"
  "notes": "Manual investigation — IP geolocation matches stolen-card report 4421"  // optional, ≤2000 chars
}
```

**Mutations on success:**

1. `transactions.analyst_label` ← `label`
2. `transactions.analyst_notes` ← `notes`
3. `transactions.reviewed_at` ← `now()`
4. A new `AuditLog` row is inserted with:
   - `actor = X-Analyst-Id`
   - `action = "ANALYST_DECISION"`
   - `resource_type = "transaction"`
   - `resource_id = tx.id`
   - `payload = json({"label": …, "notes": …, "prev_label": <previous value>})`

`fraud_decision` is **not mutated**. The model's verdict is
preserved verbatim for offline evaluation and retraining — that is
the entire reason `analyst_label` is a separate column.

**Status codes:**

| Status | Condition |
| --- | --- |
| `200 OK` | Override accepted; response is the full updated `TransactionDetail` envelope. |
| `404 Not Found` | No transaction with that id. |
| `409 Conflict` | Transaction is not in `REVIEW` — `APPROVE`, `DECLINE`, and `PENDING` rows are not analyst-actionable. |
| `422 Unprocessable Entity` | Missing `X-Analyst-Id`, invalid label, notes too long. |

**Idempotency.** Re-submitting the same `(label, notes)` for an
already-reviewed transaction returns `200 OK` with the existing
state and **does not write another audit-log row**. Submitting a
*different* `(label, notes)` for an already-reviewed transaction
also returns `200 OK`, updates the columns, and **writes a new
audit-log row** with `action = "ANALYST_DECISION_REVISED"` so the
chain of human verdicts is preserved. This is the cheapest
correct behavior for a demo and matches how real review tools
handle "I changed my mind".

**Why not block revisions.** A real fraud-ops console always lets
the lead analyst override a junior analyst's call. Blocking
revisions here would model a constraint that does not exist in
production systems.

---

## Schemas

### TransactionDetail (new)

Lives in `app/schemas/transaction.py` alongside `TransactionResponse`
and `TransactionScored`. Reuses the existing `ContributorEntry`
from `app/schemas/explanation.py` plus a new `direction: Literal["fraud", "legit"]`
field — or a tiny `DetailContributorEntry` that extends it. We'll
go with **extending in place** so a single schema serves both code
paths and the frontend never needs two parallel types. The new
field is computed on the way out, so persisted rows do not need a
migration.

### AnalystDecisionRequest (new)

```python
class AnalystDecisionRequest(BaseModel):
    label: Literal["CONFIRMED_FRAUD", "CONFIRMED_LEGIT"]
    notes: str | None = Field(default=None, max_length=2000)
    model_config = ConfigDict(extra="ignore")
```

### AuditEntry (new)

```python
class AuditEntry(BaseModel):
    id: int
    actor: str
    action: str
    payload: dict[str, Any] | None  # parsed from AuditLog.payload TEXT column
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
```

---

## Repository changes

`app/repositories/transaction.py` already exposes `get_by_id`. Two
additions:

1. `apply_analyst_decision(db, *, tx, label, notes, now) -> Transaction`
   — flushes the three column updates inside the caller's session.
   Does *not* commit. The caller (the service layer) owns the
   transaction boundary so the column update and the audit-log
   insert land atomically.

2. `recent_audit_entries(db, *, resource_type, resource_id, limit=20) -> list[AuditLog]`
   — lives in `app/repositories/audit.py` (which already exists per
   the workspace tree). Orders by `created_at DESC, id DESC`,
   `limit(20)`.

## Service layer

A new `app/services/review.py` module owns the override flow:

```python
def apply_analyst_decision(
    db: Session,
    *,
    transaction_id: str,
    analyst_id: str,
    request: AnalystDecisionRequest,
    now: datetime | None = None,
) -> TransactionDetail
```

Responsibilities, in order:

1. Load the transaction; raise `LookupError` if missing (router
   maps to 404).
2. If `tx.fraud_decision != REVIEW`, raise `ReviewConflictError`
   (router maps to 409).
3. Compare against current `(analyst_label, analyst_notes)`:
   - Identical → return the existing detail envelope unchanged
     (idempotent no-op, no audit-log row).
   - Different → update columns, choose `action` based on whether
     `analyst_label IS NULL` (`ANALYST_DECISION`) or not
     (`ANALYST_DECISION_REVISED`), insert the audit-log row,
     commit.
4. Build and return the fresh `TransactionDetail`.

A second module `app/services/transaction_detail.py` owns the
**read** path: load the row, hydrate `top_contributors` from
`top_features`, compute `direction`, fetch the trailing audit
entries, compute `effective_decision`, return `TransactionDetail`.

Splitting read and write into two modules keeps the service files
single-responsibility and matches the existing `services/scoring.py`
+ `services/stats.py` pattern.

---

## Auth posture

No real authentication. The `X-Analyst-Id` header is **trusted as
declared**. This is consistent with the rest of Phase 3 (no auth
anywhere) and is the cheapest correct behavior for a portfolio
demo. The header serves one purpose: making the audit log
realistic. A future Phase 5 hardening pass would replace it with
JWT verification, but designing for that here would burn the
slice's budget on plumbing.

The frontend prompts for an analyst id on first use of the
decision form and stores it in `localStorage`. No password, no
session. The README will call this out explicitly in the security
limitations section so it is not mistaken for real auth.

---

## Frontend

### Routing

The existing `<Route path="/transactions/:id" element={<TransactionDetailPage />} />`
shipped in Phase 3F-3 is reused. The stub page is **replaced**, not
extended. The back link to `/transactions` stays.

### Data hook

New `useTransactionDetail(id: string)` hook in
`frontend/src/hooks/`. TanStack Query, key `["transaction", id]`,
no polling — detail is a deliberate read. Returns the full
`TransactionDetail` envelope. Errors surface as a card-level error
state, not a page-level blank.

A second hook `useApplyAnalystDecision(id: string)` wraps
`useMutation` for the POST. On success it invalidates both
`["transaction", id]` and `["transactions"]` so the list page also
reflects the new `analyst_label`.

### Layout

One column on small screens, two columns from `lg:` up. Sections
in render order:

1. **Header card** — back link, transaction id (full, monospace,
   copy-on-click), timestamp, amount (`formatMoneyPrecise`),
   `DecisionBadge` for the effective decision, plus a small muted
   "ML said X · analyst said Y" caption when the two disagree.
2. **Two-column body**:
   - **Left**: parties card (customer id, merchant id, country,
     payment method, card-present flag); raw features table
     (collapsible, default collapsed).
   - **Right**: rules-triggered list (each rule as a small pill);
     SHAP contributors panel (top-5 horizontal directional bars —
     hand-rolled CSS, not Recharts); audit history list.
3. **Analyst decision form** — full-width, renders only when
   `fraud_decision === "REVIEW"`. Two radio buttons
   (`Confirm fraud` / `Confirm legitimate`), a textarea for
   notes, a submit button. The mutation handles the optimistic
   state. On success the form swaps for a muted
   "Reviewed by &lt;actor&gt; at &lt;time&gt;" summary; analyst id
   gates the submit and is captured via a small modal on first
   use.

### SHAP contributor bars

A purpose-built component, not Recharts. Each row:

```
amount                                       9,421.0
                              ┌──────────────────────┐
                              │██████████████████░░░░│  +1.83
                              └──────────────────────┘
```

Bars span from a centered zero baseline; positive (`shap > 0`,
"fraud" direction) extend right in `rose-400/60`, negative extend
left in `emerald-400/60`. Width is normalized against the largest
absolute SHAP value in the displayed top-5 so the layout doesn't
rescale across transactions in a misleading way (consistent
encoding across the page is a Phase 4 polish concern; per-page
normalization is the right tradeoff here). Each bar shows the
signed SHAP magnitude to two decimal places.

The whole component is ~80 lines of TSX + Tailwind, ships in
`frontend/src/components/transactions/ShapContributors.tsx`.

### Empty/error states

- Detail query fails → page renders the header skeleton, then a
  card-level "Could not load this transaction" with a retry.
- `top_contributors` is empty (legacy rows from the Phase 3B stub
  era before scoring was wired) → the SHAP panel shows
  "No SHAP attribution recorded for this transaction" and a link
  to hit the live `/explain` endpoint.
- `audit` is empty (shouldn't happen in practice — scoring writes
  one) → muted "No audit history yet".

---

## Tests

### Backend

`backend/tests/integration/test_transaction_detail.py` (new):
- `GET /transactions/{id}` returns the full envelope shape.
- 404 on unknown id.
- `top_contributors` carries the computed `direction` field.
- `effective_decision` falls through to `fraud_decision` when
  `analyst_label IS NULL`.
- `effective_decision` flips to `DECLINE` when
  `analyst_label = CONFIRMED_FRAUD` and to `APPROVE` when
  `analyst_label = CONFIRMED_LEGIT`.
- `audit` returns at most 20 entries, newest-first.

`backend/tests/integration/test_analyst_decision.py` (new):
- 200 path: REVIEW row + `CONFIRMED_FRAUD` → columns updated,
  one new audit-log row with `action = "ANALYST_DECISION"`,
  response envelope's `effective_decision = DECLINE`.
- Idempotent resubmit: identical `(label, notes)` → no new
  audit-log row.
- Revision: different `(label, notes)` after first commit →
  new audit-log row with `action = "ANALYST_DECISION_REVISED"`.
- 404 on unknown id.
- 409 when transaction is APPROVE / DECLINE / PENDING.
- 422 when `X-Analyst-Id` missing, label invalid, notes >2000 chars.
- `fraud_decision` is unchanged after every successful override.

`backend/tests/unit/test_transaction_detail.py` (new):
- `direction` classification: `shap > 0` → fraud, `shap < 0` →
  legit, `shap == 0` → legit (tie-break, documented).
- `effective_decision` mapping table covers all four `Decision`
  values × three `analyst_label` values.

### Frontend

No new unit tests — the project's pattern is to lean on Cypress /
manual smoke for UI, which Phase 4 will formalize. The Phase 3G
mutation flow gets a manual checklist in the README's "How to
demo this" section.

---

## Slicing

- **3G-1** — ADR (this file). Status: locked.
- **3G-2** — Backend: `TransactionDetail` schema, repository
  additions, `services/transaction_detail.py` + `services/review.py`,
  endpoint shape changes on `GET /transactions/{id}`, new
  `POST /transactions/{id}/decision`, full test coverage.
- **3G-3** — Frontend: replace the stub `TransactionDetailPage`
  with the full read-only detail layout (header, rules, SHAP
  contributors, features, audit). No analyst form yet.
- **3G-4** — Frontend: analyst decision form, analyst-id capture
  modal, optimistic-mutation wiring. README sync marking Phase 3G
  complete.

---

## Out of scope

- A standalone alerts queue or worklist view (Phase 3H).
- Real authentication or session management.
- Live SHAP recomputation in the detail envelope.
- Customer-name / merchant-name display (the related tables exist
  but Phase 3G keeps the detail page id-only; names land in a
  later UX pass).
- A "see full force plot" link to `/explain?format=force` — stretch
  goal, lands only if 3G-3 finishes early.
- Bulk-action affordances (multi-select on the list page → bulk
  override). Phase 4.
- A "this decision affects N similar pending rows" suggestion
  surface. Phase 4 if at all.

---

## Commit policy

Same as every prior phase. Conventional Commits, one slice per
commit, attributed to `apoorvrajdev <apoorvrajmgr@gmail.com>`.
The agent suggests the commit script and never runs `git add`,
`git commit`, or `git push` on its own.
