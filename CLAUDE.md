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

- **Backend:** Python 3.11+, FastAPI, SQLAlchemy 2.0, Pydantic v2, uv for package management
- **ML:** scikit-learn, XGBoost, SHAP, pandas
- **Frontend:** React 18, TypeScript (strict), Vite, Tailwind CSS, TanStack Query
- **Database:** SQLite (dev), Postgres-compatible schemas via SQLAlchemy
- **Quality:** ruff (lint+format), mypy (strict), pytest

## Code Standards

- All Python functions have type hints
- All money values use `Decimal`, never `float`
- All API responses go through Pydantic schemas
- Layered architecture: `api/` → `services/` → `repositories/` → `models/`
- No business logic in routers — routers only orchestrate
- Tests live in `backend/tests/unit/` and `backend/tests/integration/`

## Working Style

- Plan before implementing for any non-trivial change
- One module at a time, with tests
- After making changes, summarize what you did and commit the completed unit per the Git Workflow & Commit Policy

## Phase 5 Implementation Rules

The Phase 5 implementation specification is:
`docs/PHASE_5_PLAN.md`

Before performing Phase 5 work:
- Read `docs/PHASE_5_PLAN.md`.
- Implement exactly ONE milestone (M0–M7) at a time.
- Do not begin the next milestone automatically.
- Section 6 defines explicit Phase 5 non-goals.
- Section 25 identifies files that should not be modified unless necessary.
- Tests are part of each milestone's implementation.
- Preserve existing behavior unless the milestone explicitly requires a change.
- Do not weaken or delete existing tests to make a milestone pass.
- Do not add dependencies unless the milestone requires them.
- Commit completed, verified milestone work per the Git Workflow & Commit Policy; never push or merge without explicit authorization.
- At the end of each milestone, report:
  1. files changed
  2. tests run/results
  3. git diff summary
  4. deviations from the Phase 5 plan
  5. commits created for the milestone
- STOP after the milestone and wait for explicit approval before starting the next milestone.

## Git Workflow & Commit Policy

The user prefers a healthy, active Git history with frequent commits. Use judgment about commit boundaries; do not ask before every normal commit.

- Commit autonomously once a meaningful unit of work is complete and verified — run whichever tests, type checks, or builds are relevant first, then commit that unit before moving on to the next.
- Prefer separate, logically scoped commits for genuinely independent units (implementation, tests, documentation, refactoring, configuration, cleanup). Never batch unrelated changes into one commit.
- Keep commits atomic: one coherent change per commit, understandable on its own.
- Never create empty, no-op, or otherwise artificial commits to raise the commit count, and never split one atomic change into micro-commits for the same reason.
- Review `git status` and `git diff` before every commit so only the intended files are staged.
- Never commit secrets, credentials, generated artifacts, temporary files, or unrelated user changes.
