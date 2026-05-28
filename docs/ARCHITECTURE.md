# Fraud Radar — Architecture

> Companion document to the [README](../README.md) and the ADRs under
> [`docs/adr/`](adr/). Diagrams render natively on GitHub via Mermaid.

This document is the one-page mental model of the system. It covers:

1. The runtime topology (live mode and public-demo mode)
2. The request path for a single scored transaction
3. The data flow for the analyst-review loop
4. The layered code structure

The detail of any particular slice lives in its own ADR. This page
exists so a reviewer can orient themselves in under two minutes.

---

## 1. Runtime topology

Two deployment shapes ship from the same codebase. Local development
runs the full live stack; the public Vercel demo serves a frozen
snapshot and never touches a backend.

### Live (local development)

```mermaid
flowchart LR
    SIM["Background simulator<br/>(CLI HTTP client,<br/>6 fraud patterns)"]
    API["FastAPI service<br/>app.main:app<br/>(port 8000)"]
    DB[("SQLite<br/>fraud_radar.db<br/>50,010 seeded rows")]
    MODEL["XGBoost booster<br/>+ SHAP TreeExplainer<br/>(loaded once at startup)"]
    FE["React 19 + Vite<br/>TanStack Query<br/>(port 5173)"]
    USER(["Analyst<br/>(browser)"])

    SIM -->|"POST /transactions<br/>(Idempotency-Key)"| API
    API <-->|"SQLAlchemy 2.0"| DB
    API <-->|"in-process call"| MODEL
    FE -->|"GET /stats, /transactions,<br/>/alerts, /transactions/:id"| API
    FE -->|"POST /transactions/:id/decision<br/>(X-Analyst-Id)"| API
    USER --> FE
```

### Public demo (Vercel, zero-cost, zero-backend)

```mermaid
flowchart LR
    RECRUITER(["Recruiter<br/>(browser)"])
    CDN["Vercel CDN<br/>fraud-radar.vercel.app"]
    BUNDLE["Static SPA bundle<br/>VITE_DEMO_MODE=true"]
    SNAPSHOT[/"public/demo-data/<br/>stats-*.json<br/>transactions.json + /:id/<br/>alerts.json<br/>manifest.json"/]

    RECRUITER --> CDN
    CDN -->|"index.html + JS/CSS"| BUNDLE
    BUNDLE -->|"axios adapter<br/>(demoApi.ts)"| SNAPSHOT
```

The same React tree runs in both modes. The branch lives at exactly
one layer — `src/lib/api.ts` swaps in `demoAdapter` when the env flag
is set, and every hook stays unchanged. See
[`docs/adr/PHASE_4A_DEMO_SCOPE.md`](adr/PHASE_4A_DEMO_SCOPE.md) for
the locked snapshot contract.

---

## 2. Request path: one transaction, end-to-end

This is what happens between a simulator `POST` and the row appearing
on the dashboard.

```mermaid
sequenceDiagram
    autonumber
    participant SIM as Simulator
    participant API as FastAPI router<br/>(api/v1/transactions.py)
    participant IDEM as Idempotency service
    participant SCORE as Scoring service
    participant CTX as TransactionContext loader
    participant FEAT as FeatureExtractor<br/>(17 features)
    participant RULES as Rules engine<br/>(6 pure rules)
    participant ML as XGBoost + SHAP
    participant DEC as Decision matrix
    participant AUD as Audit log

    SIM->>API: POST /transactions + Idempotency-Key
    API->>IDEM: lookup(key)
    alt cached
        IDEM-->>API: stored response
        API-->>SIM: 200 (replay)
    else fresh
        API->>SCORE: score_transaction(tx)
        SCORE->>CTX: load(customer, merchant,<br/>recent velocity window)
        CTX-->>SCORE: TransactionContext
        SCORE->>FEAT: extract(tx, ctx)
        FEAT-->>SCORE: 17-d vector
        SCORE->>RULES: evaluate(tx, ctx)
        RULES-->>SCORE: rules_triggered: list[str]
        SCORE->>ML: predict_proba + TreeExplainer
        ML-->>SCORE: fraud_score, top SHAP contributors
        SCORE->>DEC: decide(score, rules, threshold)
        DEC-->>SCORE: APPROVE | REVIEW | DECLINE
        SCORE->>AUD: append("SCORED", payload)
        SCORE-->>API: TransactionScored
        API->>IDEM: store(key, response)
        API-->>SIM: 201 Created
    end
```

Service-layer latency (Phase 3C measurement): **p50 = 3.7 ms · p95 =
5.8 ms** end-to-end. Full methodology in
[`backend/ml/artifacts/latency_metrics.json`](../backend/ml/artifacts/latency_metrics.json).

---

## 3. Analyst-review loop

Once a transaction lands in `REVIEW`, it surfaces on the alerts queue
and waits for a human verdict. The cache wiring below is what makes
"submit verdict → row vanishes from queue" feel instant.

```mermaid
flowchart TD
    QUEUE["GET /alerts<br/>predicate: fraud_decision=REVIEW<br/>AND analyst_label IS NULL"]
    LIST["/alerts page<br/>(summary strip + queue table)"]
    DETAIL["/transactions/:id page<br/>(SHAP bars + audit timeline)"]
    FORM["Analyst decision form<br/>(REVIEW only,<br/>X-Analyst-Id capture)"]
    POST["POST /transactions/:id/decision<br/>(idempotent on identical resubmit;<br/>ANALYST_DECISION_REVISED on change)"]
    REPO[("transactions.analyst_label<br/>+ audit_log row")]
    CACHE["TanStack Query cache<br/>invalidate ['transactions'] + ['alerts']<br/>setQueryData ['transaction-detail', id]"]

    QUEUE --> LIST
    LIST -->|"click row"| DETAIL
    DETAIL --> FORM
    FORM --> POST
    POST --> REPO
    POST --> CACHE
    CACHE -->|"effective_decision flips,<br/>row drops off queue"| LIST
    CACHE -->|"dual-badge updates<br/>without refetch"| DETAIL
```

The `effective_decision` field on the detail envelope preserves the
model's verdict verbatim while reflecting any analyst override, so
the audit trail and the operational view never disagree. See
[`docs/adr/PHASE_3G_DESIGN.md`](adr/PHASE_3G_DESIGN.md) for the
envelope shape and override semantics, and
[`docs/adr/PHASE_3H_DESIGN.md`](adr/PHASE_3H_DESIGN.md) for the queue
predicate and score-bucket boundaries.

---

## 4. Layered code structure

The backend is a layered monolith — the seams are placed so any one
layer could be pulled into its own service later without rewriting
the others.

```mermaid
flowchart TD
    subgraph Backend["backend/app/"]
        ROUTERS["api/v1/<br/>thin routers — orchestration only"]
        SCHEMAS["schemas/<br/>Pydantic v2 wire contracts"]
        SERVICES["services/<br/>business logic<br/>(scoring, alerts, idempotency, review)"]
        FRAUD["fraud/<br/>rules + features + explainer<br/>(pure, no I/O)"]
        REPOS["repositories/<br/>data access<br/>(velocity queries, keyset pagination)"]
        MODELS["models/<br/>SQLAlchemy 2.0 ORM<br/>(Decimal money, CHECK constraints)"]
    end

    subgraph ML["backend/ml/"]
        DATA["data.py + generate_dataset.py<br/>(50,010 synthetic rows, seed=42)"]
        TRAIN["train.py<br/>(calls production FeatureExtractor —<br/>zero train/inference skew)"]
        EVAL["evaluation.py + analysis/<br/>(segments, calibration, global SHAP)"]
        ART["artifacts/<br/>model.json, metrics.json,<br/>MODEL_CARD.md"]
    end

    subgraph Frontend["frontend/src/"]
        PAGES["pages/<br/>route components"]
        HOOKS["hooks/<br/>TanStack Query wrappers"]
        LIB["lib/api.ts<br/>(swaps to demoAdapter in demo mode)"]
        COMP["components/<br/>layout + dashboard + transactions + alerts + ui"]
    end

    ROUTERS --> SERVICES
    ROUTERS --> SCHEMAS
    SERVICES --> FRAUD
    SERVICES --> REPOS
    REPOS --> MODELS
    DATA --> TRAIN
    TRAIN --> FRAUD
    TRAIN --> ART
    EVAL --> ART
    PAGES --> HOOKS
    HOOKS --> LIB
    PAGES --> COMP

    ART -.->|"loaded once at startup"| FRAUD
```

The arrow from `ART` to `fraud/` is the only cross-tree dependency at
runtime: the FastAPI app boots, loads `model.json` and the SHAP
explainer once, and reuses them across every request. Training runs
out-of-band and writes new artifacts that the app picks up on its
next restart.

---

## See also

- [`docs/adr/PHASE_3_DESIGN.md`](adr/PHASE_3_DESIGN.md) — rules
  engine, ingestion endpoint, idempotency, scoring pipeline,
  simulator.
- [`docs/adr/PHASE_3E_DESIGN.md`](adr/PHASE_3E_DESIGN.md) — dashboard
  aggregate endpoints + CORS posture + frontend foundation.
- [`docs/adr/PHASE_3F_DESIGN.md`](adr/PHASE_3F_DESIGN.md) —
  transactions list + keyset pagination + filter contract.
- [`docs/adr/PHASE_3G_DESIGN.md`](adr/PHASE_3G_DESIGN.md) — detail
  envelope + analyst-override endpoint + `effective_decision`
  semantics.
- [`docs/adr/PHASE_3H_DESIGN.md`](adr/PHASE_3H_DESIGN.md) — alerts
  queue + queue predicate + score-bucket boundaries.
- [`docs/adr/PHASE_4A_DEMO_SCOPE.md`](adr/PHASE_4A_DEMO_SCOPE.md) —
  zero-cost Vercel-only architecture + snapshot contract.
- [`backend/ml/MODEL_CARD.md`](../backend/ml/MODEL_CARD.md) — segment
  metrics, calibration, global SHAP, limitations.
