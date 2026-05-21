<h1 align="center">Fraud Radar</h1>

<p align="center">
  <strong>Real-time fraud detection where machine learning meets production engineering.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20development-yellow?style=flat-square" alt="Status: In Development" />
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React 18" />
  <img src="https://img.shields.io/badge/TypeScript-strict-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/XGBoost-tabular-EB6F2D?style=flat-square" alt="XGBoost" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License: MIT" />
</p>

<p align="center">
  A deliberately scoped engineering showcase that mirrors how tier-1 financial institutions build the systems that decide, in milliseconds, whether your card transaction goes through.
</p>

---

## Status

> 🚧 **Active build.** This is a flagship portfolio project being built across four focused days as a public engineering showcase. Day 1 and Day 2 are both complete — the SQLAlchemy 2.0 models, Alembic migrations, Pydantic v2 schemas, repository layer, synthetic dataset generator, feature extractor, XGBoost training pipeline, SHAP explanation endpoint, and auto-regenerated model card (segment metrics, calibration analysis, global SHAP) are all in place. The dataset holds 50,010 transactions across 500 customers and 200 merchants at a 1.52% fraud rate with six injected fraud patterns, reproducible from `seed=42`. The feature extractor turns each transaction into a 17-dimensional vector covering amount, time-of-day, geographic mismatch, velocity, customer history, and merchant context. The trained classifier scores **0.9327 PR-AUC** and **0.9785 Recall @ 1% FPR** on a chronological held-out test fold. Day 3 picks up with the rules engine, the transaction-ingestion endpoint, and the React dashboard. The intent is to ship steadily, in public, with honest commits — not to pretend it is further along than it is.

> 📊 **Headline results (held-out test fold).** PR-AUC **0.9327** · ROC-AUC **0.9989** · Recall @ 1% FPR **0.9785** · Recall @ 5% FPR **1.0000** · at operating threshold 0.7431 → precision **0.61**, recall **0.96**. Source: [`backend/ml/artifacts/metrics.json`](backend/ml/artifacts/metrics.json). Detailed breakdown in [Test-Set Metrics](#-test-set-metrics) below.

---

## 📌 What Is This Project?

Fraud Radar is a real-time card-fraud detection dashboard that simulates the kind of monitoring platforms used inside tier-1 financial institutions — the JPMorgans, Citis, Amexes, and Stripes of the world. Transactions flow in, get scored by a hybrid rules-plus-ML pipeline in single-digit milliseconds, and surface in a dashboard where analysts can triage anything suspicious.

It is **not** a production payment processor and it is **not** a startup MVP. There is no payment rail behind it and no real money moves. Every transaction is synthetic. What this project *is* is a deliberate engineering showcase — the kind of thing you build to demonstrate that you understand how the real systems are put together, not just how to glue a model to a web framework.

I built it for hiring teams evaluating backend, ML, and fintech engineering skills, and for anyone who has ever wondered what an end-to-end fraud-monitoring platform actually looks like underneath the marketing copy. Every architectural decision in this repository is one I can defend in an interview.

---

## 🎯 Why It Matters

Card fraud costs the global payments industry **tens of billions of dollars** every year. The platforms that fight it are some of the most demanding software in the financial world: they need to score transactions in **single-digit milliseconds**, produce **explainable decisions** that hold up under regulatory scrutiny, and maintain **immutable audit trails** that survive compliance reviews years after the fact.

This project demonstrates the architectural patterns that real fraud platforms rely on, scaled down to a size one engineer can build and reason about end to end:

- **Idempotent transaction ingestion** using the Stripe-style `Idempotency-Key` pattern so retries never double-charge or double-block.
- **Hybrid rules-plus-ML scoring** that combines fast hard-coded rules with a probabilistic model.
- **Human-in-the-loop review queues** because no production fraud system runs fully automated.
- **Append-only audit logs** that record every decision, every override, every reason.

---

## 💡 What This Project Demonstrates

- Production-style **FastAPI** backend with a clean layered architecture (`api → service → repository → models`).
- **Idempotent transaction processing** using a Stripe-pattern `Idempotency-Key` header.
- **Hybrid fraud pipeline**: a deterministic rules engine layered with an **XGBoost** probabilistic classifier.
- **Model explainability** with **SHAP** values surfaced for every decision the model makes.
- **Realistic synthetic dataset** built on top of `Faker` and modeling six distinct real-world fraud patterns.
- **SQLAlchemy 2.0** typed ORM with **Alembic** migrations and Postgres-compatible schemas.
- **React 18 + TypeScript + Tailwind** dashboard with TanStack Query for server state.
- **Append-only audit log** that demonstrates compliance and observability awareness from day one.
- **Decimal-precision money handling** — no `float` arithmetic anywhere near a monetary value.
- **Clean Git workflow** with Conventional Commits and small, reviewable changesets.

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────┐
                    │     React Dashboard (Vite + TS)      │
                    │   TanStack Query · Tailwind · Recharts │
                    └──────────────────┬───────────────────┘
                                       │ HTTPS / JSON
                    ┌──────────────────▼───────────────────┐
                    │       FastAPI API Gateway            │
                    │   Pydantic v2 · Idempotency-Key       │
                    └──────────────────┬───────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
        ┌───────▼──────┐       ┌───────▼───────┐      ┌───────▼────────┐
        │  api/        │──────▶│  services/    │─────▶│ repositories/  │
        │  routers     │       │ fraud pipeline│      │  SQLAlchemy 2  │
        └──────────────┘       └───────┬───────┘      └───────┬────────┘
                                       │                      │
                              ┌────────▼─────────┐    ┌───────▼────────┐
                              │  Rules Engine    │    │   SQLite DB    │
                              │  + XGBoost Model │    │ (Postgres-ready)│
                              │  + SHAP Explainer│    └────────────────┘
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │  models/*.joblib │
                              │  (artifact store)│
                              └──────────────────┘

                    ┌──────────────────────────────────┐
                    │  Background Transaction Simulator │
                    │  (feeds the ingestion endpoint)   │
                    └──────────────────────────────────┘
```

**Why a monolith?** Splitting fraud scoring, ingestion, and persistence into separate services would burn the entire four-day budget on Kubernetes manifests, message bus wiring, and inter-service contracts instead of features the reviewer can actually see and click through. A layered monolith captures the same separation-of-concerns thinking with a tenth of the operational surface area.

**Why SQLite?** It keeps the developer experience to a single `uv sync` with zero containers required, and every model is written with SQLAlchemy types that are 1:1 compatible with PostgreSQL. Swapping engines is a single `DATABASE_URL` change and an Alembic run.

**Why synchronous fraud scoring?** Production fraud platforms typically score on a hot path fed by Kafka or Kinesis. Inside a single-process demo, an async pub/sub layer would be cargo culting — synchronous calls model the same logical flow and keep latency observable. The README's "What I'd build next" section calls out the event-driven upgrade path explicitly.

---

## 🧠 Fraud Detection Pipeline

Every incoming transaction passes through a two-stage scoring pipeline before a decision is returned:

```
Transaction
    │
    ▼
┌─────────────────────┐
│  1. Rules Engine    │  ──▶  Hard-block on obvious fraud signals
└──────────┬──────────┘       (velocity, geo-impossibility, blocklists)
           │
           ▼
┌─────────────────────┐
│  2. Feature Builder │  ──▶  Derives velocity, geo, recency, merchant
└──────────┬──────────┘       and amount-percentile features
           │
           ▼
┌─────────────────────┐
│  3. XGBoost Scorer  │  ──▶  Probability ∈ [0, 1]
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. SHAP Explainer  │  ──▶  Top contributing features per decision
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Decision Logic  │  ──▶  APPROVE  · REVIEW  · DECLINE
└─────────────────────┘
```

1. **Rules engine** runs synchronously and short-circuits on obvious fraud — impossible-travel velocity, known blocklists, hard amount ceilings.
2. **XGBoost classifier** returns a calibrated probability for everything the rules don't catch.
3. **SHAP explainer** attaches the top contributing features to every scored decision so the dashboard never shows a black-box verdict.
4. **Decision logic** maps the probability into one of three actions using configurable thresholds: `APPROVE`, `REVIEW` (queue for analyst triage), or `DECLINE`.

The synthetic dataset that trains the model deliberately injects **six real-world fraud patterns** so the classifier learns signal that resembles production data, not toy noise:

- **Card testing** — rapid bursts of small-value transactions used to validate stolen card numbers.
- **Geo-velocity** — physically impossible travel between consecutive transactions.
- **Account takeover** — a long dormancy followed by a sudden high-value transaction.
- **High-amount anomalies** — transactions far outside the cardholder's historical spend distribution.
- **Off-hours patterns** — clustering of transactions at unusual times for the cardholder.
- **Merchant concentration** — abnormal density of activity at a single merchant or merchant category.

---

## 🔍 Explainability

Every scored transaction can be inspected via `GET /api/v1/transactions/{id}/explain`. The endpoint runs the same `FeatureExtractor` the scorer uses and returns the fraud score, the decision, the top 5 SHAP contributors, and the full 17-feature attribution map:

```bash
curl http://localhost:8000/api/v1/transactions/<TX_ID>/explain
```

Two PNG formats are also supported for embedding in dashboard tiles: `?format=force` (top-8 features plus an aggregated remainder) and `?format=waterfall` (all 17 features).

SHAP is computed at inference time against a `TreeExplainer` cached at process startup — not pre-computed in batch. Every explanation is live for every scored transaction, and the same feature-extraction code path serves both training and inference, so the explainer never drifts from the scorer.

---

## 🛠️ Tech Stack

| Layer        | Technologies                                                          |
| ------------ | --------------------------------------------------------------------- |
| **Backend**  | Python 3.11+, FastAPI, SQLAlchemy 2.0, Pydantic v2, Alembic, uv       |
| **ML**       | scikit-learn, XGBoost, SHAP, pandas, numpy, Faker                     |
| **Frontend** | React 18, TypeScript (strict), Vite, Tailwind CSS, TanStack Query, Recharts |
| **Database** | SQLite (dev), Postgres-compatible schemas via SQLAlchemy              |
| **Tooling**  | ruff, mypy (strict), pytest, ESLint                                   |
| **Infra**    | Docker / Podman Compose (planned, Day 4)                              |

---

## 📁 Repository Structure

```
fraud-radar/
├── backend/                          # FastAPI service + ML pipeline
│   ├── app/                          # Web/API layer
│   │   ├── api/v1/                   # Routers — thin orchestration only
│   │   ├── services/                 # Business logic (planned, Day 3)
│   │   ├── repositories/             # SQLAlchemy data access
│   │   ├── models/                   # SQLAlchemy ORM models
│   │   ├── schemas/                  # Pydantic v2 request/response schemas
│   │   ├── fraud/                    # FeatureExtractor, FraudExplainer, plots
│   │   ├── core/                     # Reserved cross-cutting utilities
│   │   ├── db/                       # Session, engine
│   │   └── simulator/                # Background transaction simulator (planned, Day 3)
│   ├── ml/                           # Training + analysis + artifacts
│   │   ├── synthesis/                # Synthetic dataset generators
│   │   ├── analysis/                 # Segment, calibration, global SHAP
│   │   ├── data/                     # Generated CSV (gitignored)
│   │   ├── artifacts/                # Trained model + metrics JSONs
│   │   ├── notebooks/                # Exploratory notebooks
│   │   ├── train.py                  # XGBoost training pipeline
│   │   ├── analyze.py                # Post-training analysis + model card
│   │   ├── generate_dataset.py       # CLI to seed DB + write labelled CSV
│   │   ├── splits.py                 # Chronological train/val/test split
│   │   ├── tuning.py                 # RandomizedSearchCV wrapper
│   │   ├── evaluation.py             # PR-AUC, Recall@FPR, threshold selection
│   │   ├── data.py                   # Dataset loader (DB rows + CSV labels)
│   │   ├── artifacts.py              # Save/load model artifacts (JSON)
│   │   └── MODEL_CARD.md             # Auto-regenerated from analyze.py
│   ├── tests/
│   │   ├── unit/                     # 87 unit tests
│   │   └── integration/              # 5 integration tests
│   ├── alembic/                      # DB migrations
│   │   └── versions/
│   └── pyproject.toml                # uv-managed dependencies
├── frontend/                         # React + TypeScript dashboard (Day 3)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── types/
│   │   └── assets/
│   └── package.json
├── docs/
│   ├── adr/                          # Architecture decision records
│   └── screenshots/
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+**
- Node **20+**
- [`uv`](https://docs.astral.sh/uv/) for Python package management
- Git

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

The interactive API docs will be live at **http://localhost:8000/docs**.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be live at **http://localhost:5173**.

### Reproduce Training

`fraud_radar.db` and `ml/data/synthetic_transactions.csv` are gitignored — they must be regenerated locally before training:

```bash
cd backend
uv sync
uv run python -m ml.generate_dataset   # ~2 min: populates DB + writes labelled CSV
uv run python -m ml.train              # ~5–10 min: tunes, fits, evaluates, saves artifacts
uv run python -m ml.analyze            # ~1 min: regenerates MODEL_CARD.md + segment / calibration / feature-importance artifacts
cat ml/artifacts/metrics.json
```

The training run writes `model.json`, `feature_list.json`, `threshold.json`, `metrics.json`, `training_metadata.json`, and `pr_curve.png` into `backend/ml/artifacts/`. Of those, `model.json` and `pr_curve.png` are gitignored; the rest are committed so each training run is reviewable in version control.

`ml.analyze` then adds `segment_metrics.json`, `calibration_metrics.json`, `feature_importance.json`, plus `calibration_curve.png`, `global_shap_beeswarm.png`, and `global_shap_bar.png`, and rewrites [`backend/ml/MODEL_CARD.md`](backend/ml/MODEL_CARD.md) with the current numbers. JSON outputs are committed; the PNGs are gitignored.

### Tests

```bash
cd backend
uv run pytest -v
```

Runs ~92 test cases covering the chronological splitter, evaluation metrics, artifact round-trip, SHAP additivity, force / waterfall plot rendering, segment routing, calibration math (including positive-class variants), and the `/explain` endpoint via `TestClient`.

---

## 🗺️ Roadmap

### Day 1 — Foundations

- [x] **1A** — Repository scaffolding with Conventional Commits
- [x] **1B** — FastAPI backend with health endpoint
- [x] **1C** — React + TypeScript + Tailwind frontend
- [x] **1D** — Portfolio-grade README with architecture and roadmap

### Day 2 — Data Layer + ML Foundation

- [x] **2A** — SQLAlchemy 2.0 ORM models with Decimal money typing
- [x] **2B** — Alembic migrations with CHECK constraints and indexes
- [x] **2C** — Pydantic v2 schemas with ISO 4217/3166 validation
- [x] **2D** — Repository layer with velocity-query support
- [x] **2E** — Synthetic dataset generator (50,010 transactions, 6 fraud patterns)
- [x] **2F** — Feature extractor (17 features: amount, time, geo, velocity, history, merchant)
- [x] **2G** — XGBoost classifier training (PR-AUC 0.9327, Recall @ 1% FPR 0.9785; chronological split, randomised CV)
- [x] **2H** — SHAP explainability integration (inference-time `TreeExplainer`; JSON, force, waterfall response formats)
- [x] **2I** — Model card with metrics (segment analysis, calibration, global SHAP, limitations)
- [x] **2J** — README polish marking Day 2 complete

### Day 3 — API + Frontend Dashboard

- [ ] **3A** — Rules engine
- [ ] **3B** — Transaction ingestion endpoint with idempotency
- [ ] **3C** — Fraud scoring pipeline integration
- [ ] **3D** — Background transaction simulator
- [ ] **3E** — Dashboard overview with KPIs
- [ ] **3F** — Transactions list with filters
- [ ] **3G** — Transaction detail with fraud-score breakdown
- [ ] **3H** — Alerts / review queue page

### Day 4 — Polish & Documentation

- [ ] **4A** — Docker / Podman Compose setup
- [ ] **4B** — Architecture diagrams
- [ ] **4C** — Screenshots and demo GIF
- [ ] **4D** — Final README with measured metrics

---

## 🎨 Engineering Decisions

> **Why a monolith?**
> Splitting Fraud Radar into microservices would consume the entire four-day budget on Kubernetes, message buses, and service contracts rather than on features a reviewer can actually click. A layered monolith demonstrates the same separation-of-concerns discipline without the operational overhead, and the seams are placed so that pulling a service out later would be a refactor, not a rewrite.

> **Why SQLite?**
> It is a deliberate dev-environment choice. Schemas are written using SQLAlchemy 2.0 ORM with column types that map cleanly to PostgreSQL. Swapping the engine is a single `DATABASE_URL` change plus an Alembic run — there is no SQLite-specific SQL anywhere in the codebase.

> **Why a hybrid rules + ML approach?**
> Real fraud systems never rely solely on a model. Rules catch the obvious, hard, and legally-required cases instantly with explainable logic that a compliance officer can audit. ML catches the subtle, drifting, and previously-unseen patterns the rules would miss. Together they form the **defense-in-depth** posture that every production fraud platform I have studied actually uses.

> **Why XGBoost?**
> It is the industry standard for tabular fraud detection. It trains fast on modest hardware, handles the extreme class imbalance that fraud data always carries, and pairs naturally with SHAP for per-decision explanations. A deep network would be a worse fit and harder to explain.

> **Why does the trainer call the production feature extractor?**
> Labels live in the synthetic CSV companion to the database; features come straight from [`backend/app/fraud/`](backend/app/fraud/) — the same `FeatureExtractor` the scoring endpoint will call at inference time. Running `python -m ml.train` is several minutes slower than a vectorised pandas implementation would be, but it guarantees zero train/inference skew: the bytes that enter XGBoost during training are byte-identical to the bytes that will enter it when a live transaction is scored. Train/inference skew is the bug that quietly destroys fraud-model precision in production, and the only durable fix is to share one code path.

---

## 📊 Test-Set Metrics

Measured on the held-out chronological test fold — last 15% of 50,010 transactions by `created_at`, no shuffling. Hyperparameters were selected by a 25-iteration `RandomizedSearchCV` scored on PR-AUC; the final fit used `early_stopping_rounds=50` against the validation fold. Numbers come straight from [`backend/ml/artifacts/metrics.json`](backend/ml/artifacts/metrics.json).

| Metric                            | Target  | Measured   |
| --------------------------------- | ------- | ---------- |
| PR-AUC                            | > 0.75  | **0.9327** |
| ROC-AUC                           | —       | **0.9989** |
| Recall @ 1% FPR                   | > 0.60  | **0.9785** |
| Recall @ 5% FPR                   | —       | **1.0000** |
| Precision @ threshold 0.7431      | —       | **0.6138** |
| Recall @ threshold 0.7431         | —       | **0.9570** |
| F1 @ threshold 0.7431             | —       | **0.7479** |
| Inference latency (p99)           | < 100ms | _pending — scoring endpoint not yet shipped_ |

> 📝 **Note.** The full [model card](backend/ml/MODEL_CARD.md) — segment performance, calibration analysis (aggregate and positives-only), global SHAP, limitations, and ethical considerations — is rebuilt on every `ml/analyze.py` run.

---

## 🧪 What I'd Build Next

Clear extension paths beyond the four-day scope, ordered by how much I'd learn building them:

- **Event-driven architecture** — Kafka or Redpanda for true async ingestion and scoring, with a replayable transaction log.
- **Online learning pipeline** — incorporate analyst feedback from the review queue back into a retraining loop.
- **Multi-currency / multi-region** — proper currency conversion, regional rule overrides, locale-aware risk tiers.
- **Production-grade observability** — OpenTelemetry traces across the pipeline, Prometheus metrics, Grafana dashboards.
- **Authentication and role-based access control** — analyst, reviewer, and admin tiers with proper auth on every endpoint.
- **3-D Secure step-up authentication** — simulated 3DS challenge flow for transactions scored as `REVIEW`.
- **Real-time transaction stream** — WebSocket push so the dashboard updates without polling.

---

## 📚 Lessons Being Learned

> The hardest engineering skill on a project like this is not the technical work — it is the judgment of where to stop. Every section of this codebase has an unbuilt version of itself that would be more rigorous, and shipping requires deciding which of those unbuilt versions is fine to leave on the cutting-room floor for now.

> Idempotency looks trivial on paper and gets genuinely subtle the moment you have to design for retries under partial-failure semantics. Writing it once with a real test for double-submission has taught me more than reading the Stripe docs ever did.

> Synthetic data quality determines model credibility more than the choice of algorithm. A perfectly tuned XGBoost on a naive dataset will score perfectly and tell you nothing useful. The hours spent on the dataset generator are the hours that decide whether the model is interesting.

---

## 📝 License & Contact

This project is released under the [MIT License](LICENSE).

**Built by [apoorvrajdev](https://github.com/apoorvrajdev)** — reach me at [apoorvrajmgr@gmail.com](mailto:apoorvrajmgr@gmail.com).

---

<p align="center">
  <em>Built as a flagship portfolio project for fintech engineering roles.</em>
</p>
