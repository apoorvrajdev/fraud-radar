---
name: backend
description: Invariants, edge cases and definition of done for changes under backend/app/ — API routes, services, repositories, schemas, ORM models, Alembic migrations, the scoring pipeline (rules, features, decision matrix, explainer), idempotency, the audit log and FX enrichment. Use before changing any of these.
---

# Backend changes

## Layering

`api/v1/` routers orchestrate only → `services/` decide → `repositories/` query → `models/`.
`fraud/` is pure: no I/O, no session. `enrichment/` attaches derived fields and must be removable
without changing any decision. Every response is a Pydantic schema from `schemas/`.

## Invariants

- Money is `Decimal` end to end — columns, schemas, arithmetic, and expected values in tests.
- Scoring is deterministic for a given transaction and context: features in the registered order
  (`fraud/feature_spec.py`), the same rules, the same decision matrix. Train/serve parity is pinned by
  `test_features_parity.py`, `test_features_batch_parity.py` and `test_model_score_parity.py`.
- The score bands (DECLINE at the threshold, REVIEW from half of it) exist twice:
  `services/scoring.py::_model_decision_from_score` (the persisted decision) and
  `fraud/explainer.py::FraudExplainer.classify` (what `/explain` reports), and `ml/analyze.py` writes
  them into `MODEL_CARD.md`. Change all three together or `/explain` contradicts the stored decision.
- Rules can force REVIEW or DECLINE regardless of score — the conservative-wins matrix in
  `services/scoring.py::_compose_decision` (ADR 3). A change aimed at the review queue must say which
  of the two it moves.
- FX enrichment never raises into the scoring path. A failed conversion leaves a normal scored,
  decided, persisted and audited row with null FX columns (`docs/FX_CONTRACT.md`, "Failure behaviour").
- `POST /transactions` is idempotent on `Idempotency-Key`. An analyst decision resubmitted unchanged
  is idempotent; a changed one records `ANALYST_DECISION_REVISED`.
- The audit log is append-only: `repositories/audit.py` records and reads, nothing else.
- Migrations are additive. Never edit a file already in `alembic/versions/`.
- The ULB featureset never enters the production featureset registry; promotion, run analysis and
  `FraudExplainer` refuse it.

## Security posture — facts, not goals

- No authentication. `X-Analyst-Id` is trusted as declared: acceptable for the demo only.
- CORS is an explicit origin list (`Settings.cors_origins`), never `*`.
- The only outbound call is the FX provider (Frankfurter, 2 s timeout, cached).
- Configuration comes from `Settings` in `app/config.py`; secrets never enter the repo.

Changing any of these moves a trust boundary.

## Edge cases the tests must cover

Add to this list whenever a bug escapes.

- Decimal quantisation at the schema boundary; no float anywhere on the path
- A transaction in the reporting currency takes the FX identity path with no lookup
- FX provider timeout, error or stale cache → the transaction is still scored, never failed
- Idempotent replay returns the stored response and writes no second row
- Keyset pagination: stable order, a cursor on the boundary row, an empty last page
- Unknown ids answer 404 (`HTTPException` in the router), never 500

## Tests

- Unit: `tests/unit/test_<module>.py`. Services get a test session; `fraud/` gets plain objects.
- Integration: `tests/integration/test_<endpoint>_endpoint.py` with FastAPI's `TestClient`.
- When scoring is touched, confirm the scoring integration tests ran rather than skipped — they need
  `backend/ml/artifacts/model.json`.
- New behaviour gets a test that fails without the change.

## When the frontend consumes the change

A changed or new response shape also touches `frontend/src/types/api.ts`, a hook, the demo adapter
(`frontend/src/lib/demoApi.ts`) and the snapshot export (`scripts/export_demo_snapshot.py`). Treat it
as cross-system work and use the frontend checklist too.

## Definition of done

- [ ] Backend gate green (CI's ruff and mypy commands, then `uv run pytest`), skip count checked
- [ ] New or changed behaviour covered by a test that fails without the change
- [ ] No layering violation, no float money, no edited migration
- [ ] Docs the change makes stale updated: ARCHITECTURE, FX_CONTRACT, the governing ADR, README numbers
- [ ] High-risk items (fraud outcome, money, schema, idempotency, audit, trust boundary) approved before implementation
