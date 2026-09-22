# Project Conventions for Claude Code

## CRITICAL: Commit & Attribution Rules

**Claude Code MUST follow these rules without exception:**

1. **NEVER add `Co-Authored-By: Claude` or any AI co-author trailer to commit messages.**
2. **NEVER add `🤖 Generated with Claude Code` footers or any AI attribution.**
3. **NEVER mention Claude, Anthropic, AI, or LLMs in commit messages, pull request descriptions, code comments, file headers, or documentation.**
4. **All commits must be authored solely by:**
   - Name: `apoorvrajdev`
   - Email: `apoorvrajmgr@gmail.com`
   - Never change the configured Git identity.
5. **Commit completed, verified work without asking** — see "Git Workflow & Commit Policy" below for commit boundaries, granularity, and message style.
6. **NEVER push or merge unless the user explicitly authorizes it.** Authorization covers that action only, not later ones.

## Commit Message Format

Use Conventional Commits. Examples:
- `chore: initial repo scaffolding`
- `feat(backend): add transaction submission endpoint`
- `fix(fraud): correct velocity feature calculation`
- `docs: add architecture diagram`
- `test(fraud): add rules engine unit tests`
- `refactor(api): extract service layer`
- `ci: run the test suite on every pull request`

Keep subject under 72 characters. Body optional but explains *why*, not *what*. Describe the engineering change itself, not the tooling used to make it.

## Project Stack

- **Backend:** Python 3.11+, FastAPI, SQLAlchemy 2.0, Pydantic v2, Alembic, uv for package management
- **ML (offline only, `backend/ml/`):** scikit-learn, XGBoost, SHAP, pandas
- **Frontend:** React 19, TypeScript (strict), Vite 8, Tailwind CSS 3, TanStack Query 5
- **Database:** SQLite (dev), Postgres-compatible schemas via SQLAlchemy
- **Quality:** ruff (lint only — see Traps), mypy (strict), pytest; tsc and eslint on the frontend

## Code Standards

- All Python functions have type hints
- All money values use `Decimal`, never `float`
- All API responses go through Pydantic schemas
- Layered architecture: `api/` → `services/` → `repositories/` → `models/`
- No business logic in routers — routers only orchestrate
- Tests live in `backend/tests/unit/` and `backend/tests/integration/`

## Where Things Live

`docs/ARCHITECTURE.md` is the two-minute mental model: §1 Runtime topology, §2 Request path (one transaction, end to end), §3 Analyst-review loop, §4 Layered code structure. "ADR 3F" below means `docs/adr/PHASE_3F_*.md`, and so on.

| Path | Owns | Governing doc |
|---|---|---|
| `backend/app/api/v1/` | thin routers | ARCHITECTURE §2, §4 |
| `backend/app/services/` | scoring, idempotency, review, alerts, stats | ADR 3, 3F, 3G, 3H |
| `backend/app/fraud/` | rules, features, decision matrix, explainer — pure, no I/O | ADR 3, 5C |
| `backend/app/enrichment/` | FX conversion — the only outbound call | `docs/FX_CONTRACT.md` |
| `backend/app/{repositories,models,schemas}/` | queries (keyset pagination; the alerts queue lives in `repositories/transaction.py`), ORM, wire contracts | ADR 3F, 3H |
| `backend/alembic/versions/` | migrations | — |
| `backend/ml/` | offline track: datasets, features, experiments, `tracks/ulb/`, run records, cards | ADR 5A, 5C, 5D, 5E; `docs/DATA_LICENSES.md` |
| `frontend/src/` | pages → hooks → `lib/api.ts` → components | ADR 3E, 4A |
| `frontend/public/demo-data/` | the generated public-demo snapshot | ADR 4A |
| `scripts/export_demo_snapshot.py` | writes the snapshot from a running backend | ADR 4A |

## Engineering Workflow

### 1. Classify first — by the riskiest file touched, not by diff size

| Level | Looks like | Flow |
|---|---|---|
| **0 Trivial** | typo, comment, doc wording | edit → the check covering that file (none for prose) → commit |
| **1 Small** | one module; nothing visible outside it changes | read target, callers, tests → edit → targeted tests + lint/types on touched files → self-review → commit |
| **2 Feature** | new endpoint, filter, page or component on one side | load the area skill → state a short plan: files, acceptance criteria, tests → implement with tests → full gate for that side → self-review against the skill → commit |
| **3 Cross-system** | backend contract + UI + demo snapshot; `ml/` ↔ `app/` | Level 2, plus: settle the contract first (`schemas/` ↔ `frontend/src/types/api.ts`), build one vertical slice end to end, run both gates, `/code-review` before reporting |
| **4 High-risk** | any trigger below | written plan → **wait for approval** → constrained change → full gate + the invariant tests → `/code-review` (and `/security-review` if a trust boundary moved) → report → **wait for approval** before anything irreversible |

A one-line edit to `_model_decision_from_score` in `services/scoring.py` is Level 4. Changing a shared helper's behaviour changes every caller, so grep the callers before classifying; an additive sibling that changes no existing behaviour does not. If unsure, take the higher level. At Level 2+, state the level in one line before starting so it can be corrected early.

**Level 4 triggers in this repo:**

- A fraud outcome or train/serve parity could change: `app/fraud/*`, `services/scoring.py`, feature order in `fraud/feature_spec.py`
- Money or FX semantics: Decimal columns, `app/enrichment/`, amount aggregates in `repositories/stats.py`
- A new Alembic migration (additive only; an existing version is never edited)
- Idempotency or audit-log semantics
- The served model (`backend/ml/artifacts/` top level, `ml.promote`) or any published result: run records, generated cards, the frozen 5D/5E methodology
- Running ML jobs that take minutes or write records: train, analyze, transfer, drift, rules audit
- Trust boundaries: CORS, `X-Analyst-Id` handling, a new outbound call, secrets, `app/config.py` settings
- Licensing: anything ULB-derived, `docs/DATA_LICENSES.md`
- New dependencies; `backend/ml/synthesis/` (frozen v1 baseline)

Irreversible means: applying a migration to real data, promoting a model, writing run records, pushing.

### 2. Context — narrow first, widen on evidence

1. This file → the area skill (`.claude/skills/{backend,frontend,ml-offline}/SKILL.md`) → the governing doc section (grep its headings; read only that section).
2. Grep/Glob for the symbol. Read the target, its direct callers, and its tests — the tests are the executable spec.
3. Widen only while a question is still open. For sweeps across many directories, use an Explore agent and keep only its conclusion.

- Not for orientation: README.md end to end, every ADR, `frontend/public/demo-data/` (400 generated files), `backend/ml/artifacts/runs/`.
- Docs state intent; code and tests state behaviour. When they disagree, trust the code and flag the doc.
- Don't re-read a file already read this session unless it changed.

### 3. Implement

- Smallest correct change, in the existing pattern; reuse before adding. One module at a time, with its tests.
- No unrelated edits, drive-by refactors or reformatting.
- Keep API contracts backward-compatible, or call out the break.
- Never weaken or delete an existing test to make a change pass.
- Bugs: reproduce (a failing test where possible) → root cause → smallest fix → that test passes.
- Ask one concise question only when a requirement can't be inferred from the code, the docs or a safe default.
- Update the docs the change makes stale (ARCHITECTURE, FX_CONTRACT, the governing ADR, README numbers). Don't create new doc files unless asked; session state and local quirks belong in memory, not the repo.

### 4. Verify — evidence, not claims

Run every check the change can affect, and none it can't. All of these were green on a clean checkout on 2026-09-22.

| Check | Command | Time |
|---|---|---|
| Backend, while iterating (`backend/`) | `uv run pytest tests/unit/test_<x>.py -q` · `uv run ruff check <files>` · `uv run mypy <files>` | seconds |
| Backend gate (`backend/`) | the ruff and mypy `run:` lines of the backend job in `.github/workflows/ci.yml`, path lists copied verbatim, then `uv run pytest` | ~2 min |
| Frontend gate (`frontend/`) | `npx tsc -b` · `npx eslint .` · `VITE_DEMO_MODE=true npm run build` | ~1 min |

- Before committing: the backend gate if anything under `backend/` changed, the frontend gate if anything under `frontend/` changed, neither for prose-only changes.
- Read the pytest summary: a skipped test is not a passing test (see Traps).
- The frontend has no automated tests; UI behaviour is verified by running it (frontend skill). If you didn't, say so.
- A check you didn't run is reported as not run.

### 5. Review

Before reporting, re-read `git diff` against the request, the area skill's invariants, changes you didn't intend, and whether the new tests would fail without the change. Level 3: `/code-review`. Level 4: `/code-review high`, plus `/security-review` if a trust boundary moved; the human approves last.

### 6. Sub-agents

Default to working alone: this is a layered monolith, most changes are coupled, and splitting them costs more in integration than it saves.

- **Explore agent** — read-only sweeps across many files where only the conclusion matters.
- **Parallel agents** — only for independent work on disjoint files: e.g. once an API schema is committed, the UI and the backend tests; or separate investigations. One owner per file, all launched in one message; the lead integrates and runs the gates on the merged tree.
- Sub-agents do not reliably inherit this file. Brief each one with the goal, its file boundary, "read CLAUDE.md and the area skill first", and the checks to run.
- Not for Level 0–1, not for Level 4 implementation, never two agents on one file.

### 7. Report

Level 2+: files changed · checks run (command → result) · deviations and assumptions · commits · what wasn't verified. Level 0–1: one or two lines. Don't restate the diff.

## Traps

- **Never run `ruff format`.** 146 of 158 backend files are not ruff-formatted, so it rewrites unrelated lines. `ruff check` is the lint gate; match the surrounding style.
- **CI lints and type-checks only the listed `ml/` modules.** Older training scripts are excluded on purpose, so `ruff check ml` is not the gate. New `ml/` modules get added to both lists.
- **Skips hide coverage.** Scoring integration tests skip without `backend/ml/artifacts/model.json` (gitignored; CI trains a tiny one); `test_demo_snapshot` skips without `frontend/public/demo-data/manifest.json`.
- **Demo mode breaks quietly.** It branches only in `frontend/src/lib/api.ts` → `demoApi.ts`; every route the UI calls needs a demo handler and snapshot data.
- **Generated files are regenerated, never edited:** `backend/ml/BENCHMARK_CARD.md`, `ULB_BENCHMARK_CARD.md`, `MODEL_CARD.md`, `frontend/public/demo-data/`.

## Phased Work Rules

Large work is specified in a phase plan, `docs/PHASE_<N>_PLAN.md`. Phase 5's (`docs/PHASE_5_PLAN.md`) is implemented; until a new plan says otherwise, its Section 6 non-goals stay out of scope and a change to a Section 25 file needs a stated reason. Those files, under `backend/`: `app/fraud/{rules,decision,transaction_context}.py`, `app/services/{idempotency,alerts,review,transaction_detail}.py`, `app/repositories/{transaction,audit}.py`, `app/api/v1/alerts.py`, existing Alembic versions, `ml/{splits,evaluation,tuning}.py`, `ml/analysis/`, `ml/synthesis/`; plus the frontend alerts and dashboard components and every existing test (fixtures may be added, assertions never weakened).

When a phase plan is active:

- Read it before starting.
- Implement exactly ONE milestone at a time.
- Do not begin the next milestone automatically.
- Respect its non-goals and its files-not-to-modify list.
- Tests are part of each milestone's implementation.
- Preserve existing behavior unless the milestone explicitly requires a change.
- Do not weaken or delete existing tests to make a milestone pass.
- Do not add dependencies unless the milestone requires them.
- Commit completed, verified milestone work per the Git Workflow & Commit Policy; never push or merge without explicit authorization.
- At the end of each milestone, report:
  1. files changed
  2. tests run/results
  3. git diff summary
  4. deviations from the plan
  5. commits created for the milestone
- STOP after the milestone and wait for explicit approval before starting the next milestone.

## Git Workflow & Commit Policy

The user prefers a healthy, active Git history with frequent commits. Use judgment about commit boundaries; do not ask before every normal commit.

- Commit autonomously once a meaningful unit of work is complete and verified — run whichever tests, type checks, or builds are relevant first, then commit that unit before moving on to the next.
- Prefer separate, logically scoped commits for genuinely independent units (implementation, tests, documentation, refactoring, configuration, cleanup). Never batch unrelated changes into one commit.
- Keep commits atomic: one coherent change per commit, understandable on its own.
- Never create empty, no-op, or otherwise artificial commits to raise the commit count, and never split one atomic change into micro-commits for the same reason.
- Review `git status` and `git diff` before every commit so only the intended files are staged.
- Stage files by path; never `git add .` or `git add -A`. Changes already in the tree that you didn't make are the user's: leave them unstaged and mention them.
- Never commit secrets, credentials, generated artifacts, temporary files, or unrelated user changes.
