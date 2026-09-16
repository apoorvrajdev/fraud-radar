# Project Conventions for Claude Code

## CRITICAL: Commit & Attribution Rules

**Claude Code MUST follow these rules without exception:**

1. **NEVER add `Co-Authored-By: Claude` or any AI co-author trailer to commit messages.**
2. **NEVER add `🤖 Generated with Claude Code` footers or any AI attribution.**
3. **NEVER mention Claude, Anthropic, AI, or LLMs in commit messages, code comments, file headers, or documentation.**
4. **All commits must be authored solely by:**
   - Name: `apoorvrajdev`
   - Email: `apoorvrajmgr@gmail.com`
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


# Git Workflow & Commit Policy

## Commit Frequently and Granularly

The user prefers a healthy, active Git history with frequent commits.

- You have permission to create Git commits autonomously when a meaningful unit of work is complete.
- Prefer multiple small, logically scoped commits over one large commit containing unrelated changes.
- Whenever a task contains multiple independent implementation, test, documentation, refactoring, configuration, or cleanup units, commit those units separately when practical.
- Do not unnecessarily batch several unrelated completed changes into a single commit.
- After completing a meaningful, self-contained change and verifying it, prefer committing it before moving to the next independent change.
- Keep commits atomic and easy to understand. A commit should represent one coherent change.
- Use clear conventional commit messages such as:
  - `feat: ...`
  - `fix: ...`
  - `refactor: ...`
  - `test: ...`
  - `docs: ...`
  - `chore: ...`
  - `ci: ...`
- Do not create empty commits, no-op commits, or artificial commits whose only purpose is to increase the commit count.
- Do not split a single atomic change into meaningless micro-commits merely to inflate the commit count.
- When several genuinely independent changes are completed during one working session, prefer separate commits for each change.
- Before committing, review `git diff` and `git status` to ensure only the intended files are included.
- Do not accidentally include secrets, credentials, generated junk, unrelated user changes, or temporary files in commits.

## GitHub Identity / Attribution

The commits should use the user's existing Git/GitHub identity.

- Do not add `Co-authored-by:` trailers for Claude, Anthropic, or any AI system.
- Do not add Claude/Anthropic attribution to commit messages, commit bodies, PR descriptions, source comments, README files, or other project history unless the user explicitly requests it.
- Do not change the user's configured Git author identity unless explicitly instructed.
- Do not describe a commit as "generated by Claude", "Claude-assisted", "AI-generated", or similar.
- Commit messages should describe the engineering change itself, not the tool used to implement it.

## Autonomous Commit Permission

The user has explicitly authorized Claude Code to:
- implement requested changes;
- run relevant tests and verification;
- create appropriate Git commits after meaningful completed units of work.

Do not ask for permission before every normal commit. Use judgment about commit boundaries and commit only after the relevant change has been implemented and verified.

Do not push to a remote or merge branches unless the user explicitly asks for that action or the current task explicitly authorizes it.