# Project Conventions for Claude Code

## CRITICAL: Commit & Attribution Rules

**Claude Code MUST follow these rules without exception:**

1. **NEVER add `Co-Authored-By: Claude` or any AI co-author trailer to commit messages.**
2. **NEVER add `🤖 Generated with Claude Code` footers or any AI attribution.**
3. **NEVER mention Claude, Anthropic, AI, or LLMs in commit messages, code comments, file headers, or documentation.**
4. **All commits must be authored solely by:**
   - Name: `apoorvrajdev`
   - Email: `apoorvrajmgr@gmail.com`
5. **NEVER stage or commit changes on your own.** Only suggest commit messages — the user runs `git commit` themselves.
6. **NEVER push to remote.** Only the user pushes.

## Commit Message Format

Use Conventional Commits. Examples:
- `chore: initial repo scaffolding`
- `feat(backend): add transaction submission endpoint`
- `fix(fraud): correct velocity feature calculation`
- `docs: add architecture diagram`
- `test(fraud): add rules engine unit tests`
- `refactor(api): extract service layer`

Keep subject under 72 characters. Body optional but explains *why*, not *what*.

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
- After making changes, summarize what you did so the user can review and commit

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
- Do not commit changes unless explicitly instructed.
- At the end of each milestone, report:
  1. files changed
  2. tests run/results
  3. git diff summary
  4. deviations from the Phase 5 plan
  5. suggested commit message
- STOP after the milestone and wait for explicit approval before starting the next milestone.