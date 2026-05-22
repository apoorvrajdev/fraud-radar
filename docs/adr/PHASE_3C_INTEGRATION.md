# Phase 3C Integration Decisions

> Quick-reference companion to [PHASE_3_DESIGN.md](PHASE_3_DESIGN.md).
> Captures the six implementation decisions made before writing the
> integration code for Phase 3C — the phase that wires the rules
> engine, FeatureExtractor, XGBoost model, FraudExplainer, and the
> idempotent ingestion endpoint into one request flow.

> Status: design. Author: apoorvrajdev. Locked before implementation.

## Context

Phase 3A shipped the rules engine (pure functions over a frozen
`TransactionContext`). Phase 3B shipped the ingestion endpoint with
Stripe-style idempotency, persisting a stub `Decision.PENDING` row.
Phase 3C replaces that stub with end-to-end scoring: load context,
run rules, extract features, score with XGBoost, attribute with SHAP,
compose the decision per the matrix in `PHASE_3_DESIGN.md`, persist
everything, write the audit log.

This document records the six decisions that determine *how* those
pieces connect, captured before any code is written so the
implementation has a target to hit.

## 1. TransactionContext is the single source of truth for recent history

The scoring service loads `recent_transactions` once and passes the
same list to both the rules engine and the FeatureExtractor.

This requires a small refactor of FeatureExtractor: today it queries
the DB internally for its velocity features; in Phase 3C it accepts a
pre-loaded list as a parameter. The refactor is in scope for 3C.

**Tradeoff:** one refactor against one source of truth. The
alternative — letting rules and features query independently — would
ship faster today but risk subtle bugs later if the two lookback
windows diverge (e.g. rules see 180 days of history but features see
only 24 hours, and someone reading the decision matrix in a year
can't reproduce why a specific transaction was flagged).

**Bonus:** the refactored FeatureExtractor becomes unit-testable
without a DB session, which improves overall test quality.

## 2. Audit log actor is "scorer:v{model_version}"

Every automated scoring decision writes one row to `audit_log` with
`actor="scorer:v1"` (or whatever version string the current model
artifact reports via `training_metadata.json`).

This costs nothing to implement and gives us a model-versioning story
for free: post-hoc analysis can filter decisions by model version,
and future human override decisions (Phase 3H — alerts/review queue)
will have a real user actor that's clearly distinguishable.

**Not chosen:** `actor="system"`. Too coarse. Doesn't survive a
retrain.

**Not chosen:** authenticated user identity. The API has no auth
yet; that's Phase 4 nice-to-have.

## 3. Latency benchmark measures two layers

`ml/benchmark_latency.py` runs 100 transactions and reports two
distinct numbers in `latency_metrics.json`:

- `service_layer_p99` — `score_transaction(context)` function timing,
  excluding HTTP. Answers: "how fast is the scoring math?"
- `endpoint_p99` — full POST round-trip via httpx against a running
  uvicorn. Answers: "how fast is the production code path?"

The README inference-latency row shows both with explicit labels.
Reporting only one number would obscure where time goes; reporting
both pre-emptively answers the obvious interview question ("is that
the model or the framework?").

## 4. PENDING stays in the Decision enum

After 3C, the happy path always produces APPROVE/REVIEW/DECLINE.
PENDING remains in the enum as a legitimate value for rows that
error mid-scoring and need re-attempting — exactly the framing
already documented in `app/fraud/decision.py`'s module docstring
from Phase 3B.

**Action required:** update Phase 3B's integration tests to assert
real decision values instead of PENDING. The
`test_post_returns_201_with_pending_decision_for_new_key` test
becomes `test_post_returns_201_with_real_decision_for_new_key` and
checks that `decision in {"APPROVE", "REVIEW", "DECLINE"}`.

**Not chosen:** removing PENDING via another migration. The constraint
expansion landed in Phase 3B; reverting it would be churn for zero
benefit. The enum value has a permanent legitimate use case.

## 5. Recent-history lookback is 180 days

The TransactionContext loader fetches the last 180 days of
transactions per customer per request. 180 days is the longest rule
lookback window (`rule_dormant_account_high_value`), so this single
fetch covers every rule.

At typical customer activity (~100 transactions per year), this
returns ~50 rows. The query uses the `ix_transactions_customer_created`
composite index from Phase 2A — measured cost ~2ms.

**Not chosen:** per-rule lookback windows with separate queries.
Three DB round-trips vs one. The latency saved by separate queries
is dwarfed by the cost of three round-trips.

## 6. README latency claim updates after measurement

The current README claim ("single-digit milliseconds" for scoring)
is an aspiration written before any measurement existed. The
benchmark produces the real numbers; the README row updates to
whatever the benchmark reports.

If the real `endpoint_p99` is 110ms instead of the implied "<10ms",
the README will say so honestly, with a one-line breakdown of where
the time goes. Honest engineering log beats marketing copy.

The currently-stated p99 target in `PHASE_3_DESIGN.md` is 60–120ms,
which is the realistic range. The README will match reality.

## Files affected by 3C

For implementation reference:

- **New:** `app/services/scoring.py` (the orchestration layer)
- **New:** `ml/benchmark_latency.py` (the benchmark script)
- **New:** `backend/ml/artifacts/latency_metrics.json` (the output)
- **New:** `tests/unit/test_scoring.py` (~10-15 tests)
- **New:** `tests/integration/test_scoring_endpoint.py` (~5-8 tests)
- **Modify:** `app/fraud/extractor.py` — accept pre-loaded
  recent_transactions instead of querying internally
- **Modify:** `app/api/v1/transactions.py` — POST endpoint calls the
  scoring service instead of writing a PENDING stub
- **Modify:** `tests/integration/test_transactions_endpoint.py` —
  update PENDING assertions to assert real decision values
- **Modify:** `README.md` — inference-latency row updated with
  measured numbers

## What's NOT in scope for 3C

- Authentication or authorization (Phase 4)
- Rate limiting (out of scope)
- Streaming / async scoring (out of scope; the synchronous flow is
  the right model for this portfolio)
- ML drift detection (out of scope)
- Model recalibration (Platt/isotonic) (out of scope)
- TTL cleanup job for idempotency_keys (out of scope per Phase 3 ADR)