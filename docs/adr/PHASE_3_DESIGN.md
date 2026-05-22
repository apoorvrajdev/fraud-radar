# Phase 3 Backend Design — Ingestion, Scoring, and Idempotency

> Status: design. Last updated 2026-05-22. Author: apoorvrajdev.
> This document covers Phases 3A through 3D. Frontend phases (3E-3H) 
> will be designed once these endpoints exist.

## Goal

Wire the components built in Phase 2 (FeatureExtractor, XGBoost model, 
FraudExplainer) into a working transaction pipeline. By end of Phase 3D, 
a client can POST a transaction and receive an APPROVE / REVIEW / DECLINE 
decision with a SHAP attribution attached, idempotently. A background 
simulator continuously feeds the endpoint with synthetic traffic.

---

## API surface

### POST /api/v1/transactions

Accepts a new transaction, runs it through the rules engine and ML 
scorer, persists the decision, returns the result.

Request:
- Header `Idempotency-Key: <UUID>` (required)
- Body: Pydantic `TransactionCreate` schema — the existing schema in 
  `app/schemas/`. Money fields as `Decimal` strings, currency as ISO 
  4217, country as ISO 3166.

Response 201 (new):

{
"transaction_id": "uuid",
"fraud_score": 0.04,
"decision": "APPROVE",
"threshold": 0.7431,
"rules_triggered": [],
"top_contributors": [...]  // top 5 SHAP, same shape as /explain
}

Response 200 (idempotent replay):
- Returns the same body that was returned on the original 201
- Header `X-Idempotency-Replay: true` to signal this was cached
- Same response body as the original even if the model has been 
  retrained since

Response 409 (idempotency conflict):
- Same `Idempotency-Key` was used with a *different* request body
- Body: `{"detail": "idempotency key reused with different payload"}`

Response 422: validation error (Pydantic default)

### GET /api/v1/transactions/{id}
- Returns the full transaction record with the decision and the SHAP 
  attribution read from the `top_features` TEXT column on the 
  transactions table — the explainer is NOT re-invoked on GET. SHAP is 
  computed and persisted on POST.
- Used by frontend transaction detail page (Phase 3G)

### GET /api/v1/transactions
- Paginated list with filters (date range, decision, country, amount range)
- Used by frontend list page (Phase 3F)
- Defer detailed filter design until 3F — for 3A-3D, a basic 
  `limit`/`offset`/`decision` filter is enough.

---

## Idempotency design

### What gets stored

A new table `idempotency_keys`:
- `key` — TEXT primary key (UUID string)
- `request_hash` — TEXT, SHA-256 of `TransactionCreate(**body).model_dump_json()`. 
  We hash the Pydantic-normalized representation, not the raw HTTP 
  bytes, so the hash is stable across whitespace differences and 
  field reordering by intermediaries.
- `transaction_id` — TEXT FK to `transactions.id`
- `response_body` — TEXT, the full JSON response that was returned
- `status_code` — INT. Distinguishes cached 201s from cached 4xx errors. 
  Stripe stores this; our Phase 3 implementation inserts only on 201 
  responses but the schema supports the more general case.
- `created_at` — TIMESTAMP
- `expires_at` — TIMESTAMP (created_at + 24 hours)

### Flow

1. Request lands with `Idempotency-Key: K` and body `B`.
2. SHA-256 hash `H(B)` over the Pydantic-normalized JSON.
3. Look up `K` in `idempotency_keys`:
   - **Miss:** proceed with normal scoring, persist the result, then 
     insert into `idempotency_keys` with key=K, request_hash=H(B), 
     response_body=serialized response. Return 201.
   - **Hit, request_hash matches:** return cached `response_body` with 
     201 + `X-Idempotency-Replay: true`. Do NOT re-score.
   - **Hit, request_hash differs:** return 409.

### Concurrency

Two requests with the same `K` arriving within milliseconds is the 
edge case. Two approaches:
- **A:** SQLite-level `INSERT OR IGNORE` race; the loser detects via 
  rowcount, sleeps briefly, retries the SELECT.
- **B:** Wrap the lookup-or-insert in a transaction with `SELECT ... 
  FOR UPDATE`. The syntax parses in SQLite but isn't honored — SQLite 
  serializes writes via the global write lock regardless.

Going with **A**. SQLite serializes writes anyway, so this is mostly 
academic for the portfolio demo. Production-Postgres would use 
advisory locks or an upsert with `RETURNING`.

### TTL

24-hour TTL on idempotency keys, enforced by `expires_at`. A cleanup 
job is out of scope for Phase 3; rows just accumulate. If this were 
real, a daily cron or a `pg_cron` job would delete expired rows. At 
simulator rate (~1 tx / 2s, full SHAP in response), the table grows 
roughly 200 MB/day. Acceptable for SQLite portfolio scope; production 
would have a TTL cleanup job.

### Model versioning

Idempotency replays return the original response computed against the 
model that was active at the time of the first POST. After `ml.analyze` 
retrains, new POSTs use the new model; replays remain frozen. This is 
correct idempotency semantics but is worth documenting explicitly 
because reviewers sometimes flinch on first read.

---

## Rules engine design

Pure functions in `app/fraud/rules.py`. Each rule:
- Takes a `TransactionContext` (the transaction + customer + merchant + 
  recent transaction history)
- Returns `RuleResult` = `(triggered: bool, reason: str | None)`

### Initial rule set (Phase 3A scope)

| Rule | Trigger condition | Effect |
|---|---|---|
| `velocity_burst` | ≥3 transactions in 120 seconds from the same customer | HARD_BLOCK |
| `geo_velocity_impossible` | Two distinct countries within 60 minutes for the same customer | HARD_BLOCK |
| `amount_ceiling` | amount > $5,000 | REVIEW |
| `high_risk_country` | country in RU/CN/NG/RO/VE/ID AND amount > $500 | REVIEW |
| `dormant_account_high_value` | account_age > 180d AND no tx in 180d AND amount > $1,000 | REVIEW |
| `off_hours_high_value` | hour ∈ [2, 5] AND amount > $500 AND NOT is_card_present | REVIEW |

Velocity rules match on `customer_id` for portfolio scope. Production 
threat model is stolen card, which would match on `card_last4`.

`HARD_BLOCK` short-circuits the pipeline → decision = DECLINE, ML 
scorer never runs.

`REVIEW` does NOT short-circuit. The transaction still goes to the 
scorer; the rule trigger is recorded alongside the model output. The 
combined decision follows this matrix:

| Hard rule | REVIEW rule | Model output | Final decision |
|---|---|---|---|
| fired | — | (not run) | DECLINE |
| — | fired | APPROVE | REVIEW |
| — | fired | REVIEW | REVIEW |
| — | fired | DECLINE | DECLINE |
| — | — | APPROVE / REVIEW / DECLINE | as model says |

Conservative wins — whichever signal (rule or model) is more 
restrictive is the final decision. `rules_triggered` in the response 
lists every rule that fired, even when its REVIEW recommendation lost 
to a more restrictive model verdict.

### Why this order

Hard rules first (compliance + obvious fraud), soft rules second 
(advisory), ML last (gradient signal). This is the defense-in-depth 
posture every production fraud system uses. The rule decisions are 
explainable in human terms; the ML decisions need SHAP. Combining 
gives you both.

---

## Scoring pipeline (Phase 3C)

The integration layer. Owns the end-to-end flow:

request → idempotency check → build TransactionContext (lookup
customer/merchant, get recent history) → run rules →
if HARD_BLOCK: persist + return DECLINE
else: extract features → score with XGBoost → SHAP →
map to APPROVE/REVIEW/DECLINE using threshold + REVIEW rules →
persist transaction + decision + audit log → cache idempotency →
return

Lives in `app/services/scoring.py`. The router in 
`app/api/v1/transactions.py` is thin orchestration — it calls the 
service, handles HTTP concerns (status codes, headers), nothing more.

`TransactionContext` is a `@dataclass` in `app/services/scoring.py` 
holding the transaction, its customer, its merchant, and the 
customer's recent transaction history (last N transactions or last 
M minutes, whichever is more useful per rule). Constructed once 
per request, passed to rules and feature extractor.

### Latency budget

Target p99 < 100ms per the README's stated claim. Components:
- Idempotency lookup: ~1 ms (SQLite indexed PK)
- Rule evaluation: 6–10 ms (three rules each running a customer-keyed 
  time-window query against `ix_transactions_customer_created`)
- Feature extraction: 30–80 ms (this is the slow part — velocity queries)
- XGBoost predict: ~1 ms
- SHAP attribution: ~10 ms (TreeExplainer on a single 17-feature row 
  against a 600-estimator booster; the earlier 30–50 ms estimate was 
  double-counting)
- Persist: ~5 ms

Realistic p99: 60–120 ms on developer hardware. We'll measure and 
report honestly, not retrofit the README.

A benchmark script `ml/benchmark_latency.py` runs 100 transactions 
through the endpoint and reports p50/p95/p99. Output goes in 
`backend/ml/artifacts/latency_metrics.json` (committed) and the README 
inference-latency row updates from "pending" to the measured number.

### Audit log

Every scoring decision writes one row to `audit_log` (table from Phase 
2A): action, actor=`system`, target=transaction_id, the resulting 
transaction state, the list of rules that fired, and the model output. 
Snapshot only — there is no 'before' state for a fresh POST. 
Append-only, no updates or deletes.

---

## Background simulator (Phase 3D)

A standalone script `ml/simulate_traffic.py`:
- Loads customer + merchant pools from DB (reuses Phase 2E factories)
- In a loop with configurable rate (default 1 tx / 2 sec)
- Picks a customer and merchant, generates a transaction with 
  realistic params, occasionally injects a fraud pattern
- POSTs to the running ingestion endpoint with a fresh idempotency key
- Logs the response decision

Runs as a separate process. Not part of the API. The frontend will 
show new transactions appearing live because the simulator is feeding 
them in.

### Configuration

```bash
uv run python -m ml.simulate_traffic \
  --rate 2 \
  --fraud-rate 0.02 \
  --duration 600 \
  --endpoint http://localhost:8000
```

---

## What's deferred to later phases

- Authentication / authorization on the endpoints — Phase 4 nice-to-have
- Rate limiting — out of scope
- Filter design for `GET /transactions` — designed in 3F
- Frontend everything — designed in 3E-3H
- Calibrating ML probabilities (Platt/isotonic) — out of portfolio scope
- Drift detection — out of portfolio scope
- Production observability (metrics, traces) — Phase 4 if time allows

---

## Open questions

1. **Should the response include SHAP attribution by default, or 
   require an explicit `?include_explanation=true`?** Pro of always-on: 
   the frontend doesn't need a second round-trip. Con: every response is 
   bigger. **Decision:** always on for the portfolio scope; flag this in 
   the README as a knob that would be off-by-default in production for 
   performance reasons. In production this would be gated behind 
   `?include_full_attribution=true` and only `top_contributors` (top-5, 
   ~500 bytes) would be returned by default. The portfolio keeps the 
   full payload for demo simplicity.

2. **What happens when the model artifact is missing at startup?** 
   Phase 2H raises a clear error. The endpoint should refuse to start 
   rather than start in degraded mode. Already handled.

3. **What about idempotency under the simulator?** The simulator 
   generates fresh UUIDs per request, so no idempotency conflicts in 
   practice. The endpoint just needs to be *correct* about idempotency, 
   not stressed by it.
