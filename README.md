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

> 🚧 **Active build.** This is a flagship portfolio project being built across four focused days as a public engineering showcase. Day 1 is complete, and Day 2 is in progress — the SQLAlchemy 2.0 models, Alembic migrations, Pydantic v2 schemas, and repository layer are in place, and the synthetic dataset generator has shipped (50,010 transactions across 500 customers and 200 merchants, 1.52% fraud rate, six injected fraud patterns, reproducible from `seed=42`). Still ahead on Day 2: feature engineering, the XGBoost training run, SHAP explainability, and the model card. The intent is to ship steadily, in public, with honest commits — not to pretend it is further along than it is.

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

> 📊 **Metrics pending.** PR-AUC, recall at fixed FPR, and inference latency numbers will land in this README the moment Day 2 training completes. I am not publishing aspirational numbers as if they were measured.

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
├── backend/                    # FastAPI service + ML pipeline
│   ├── app/
│   │   ├── api/                # Routers — thin orchestration only
│   │   ├── services/           # Business logic (fraud pipeline, scoring)
│   │   ├── repositories/       # SQLAlchemy data access
│   │   ├── models/             # ORM + Pydantic schemas
│   │   └── ml/                 # Training, features, SHAP, artifact loading
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   └── pyproject.toml          # uv-managed dependencies
├── frontend/                   # React + TypeScript dashboard
│   ├── src/
│   │   ├── components/         # Presentation components
│   │   ├── pages/              # Route-level views
│   │   └── lib/                # API client, hooks, utilities
│   └── package.json
├── docs/                       # Architecture notes and design records
├── data/                       # Synthetic datasets (gitignored)
├── models/                     # Trained artifacts (gitignored)
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

---

## 🗺️ Roadmap

### Day 1 — Foundations

- [x] Repository scaffolding with Conventional Commits
- [x] FastAPI backend with health endpoint
- [x] React + TypeScript + Tailwind frontend
- [x] Portfolio-grade README with architecture and roadmap

### Day 2 — Data Layer + ML Foundation

- [x] SQLAlchemy 2.0 ORM models with Decimal money typing
- [x] Alembic migrations with CHECK constraints and indexes
- [x] Pydantic v2 schemas with ISO 4217/3166 validation
- [x] Repository layer with velocity-query support
- [x] Synthetic dataset generator (50,010 transactions, 6 fraud patterns)
- [ ] Feature engineering module
- [ ] XGBoost classifier training
- [ ] SHAP explainability integration
- [ ] Model card with metrics

### Day 3 — API + Frontend Dashboard

- [ ] Rules engine
- [ ] Transaction ingestion endpoint with idempotency
- [ ] Fraud scoring pipeline integration
- [ ] Background transaction simulator
- [ ] Dashboard overview with KPIs
- [ ] Transactions list with filters
- [ ] Transaction detail with fraud-score breakdown
- [ ] Alerts / review queue page

### Day 4 — Polish & Documentation

- [ ] Docker / Podman Compose setup
- [ ] Architecture diagrams
- [ ] Screenshots and demo GIF
- [ ] Final README with measured metrics

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

---

## 📊 Planned Metrics

These are **targets**, not measured results. The final numbers will land here as soon as the Day 2 training run completes, with the corresponding evaluation code committed alongside.

| Metric                       | Target  |
| ---------------------------- | ------- |
| PR-AUC                       | > 0.75  |
| Recall @ 1% FPR              | > 0.60  |
| Precision @ top 1% scores    | > 0.50  |
| Inference latency (p99)      | < 100ms |

> 📝 **Note.** The published model card on Day 4 will include the full confusion matrix, calibration plot, feature importance ranking, and a frank assessment of where the model still falls short.

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
