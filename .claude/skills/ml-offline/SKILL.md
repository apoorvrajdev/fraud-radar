---
name: ml-offline
description: Rules for work under backend/ml/ — datasets, feature builds, training, analysis, experiments (transfer, temporal drift, rules audit), the ULB track, run records, the generated benchmark and model cards, and promotion. Use before changing ML code or running any ML command.
---

# Offline ML track

## What is frozen

- Methodology: `docs/adr/PHASE_5D_BENCHMARK_METHODOLOGY.md` and
  `docs/adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md`. Their decisions govern wherever
  `docs/PHASE_5_PLAN.md` differs. Changing a decision means a new, approved methodology — not an edit.
- Recorded runs in `ml/artifacts/runs/`: `synthetic_v1`, `sparkov_v1_200cards`, `sparkov_v1_full`,
  `sparkov_v1_drift`, `ulb_pca_v1_seed42/43/44`. Records are committed; `model.json` and PNGs are
  gitignored, with their digests recorded.
- `ml/synthesis/` — the in-house generator, frozen as the v1 baseline.

## Commands and what they write

All from `backend/`. Read the module docstring for flags before running anything.

| Command | Writes | Cost | Approval |
|---|---|---|---|
| `uv run python -m ml.benchmark_card` | `ml/BENCHMARK_CARD.md` from committed records | seconds | no — run only when records changed |
| `uv run python -m ml.tracks.ulb.card` | `ml/ULB_BENCHMARK_CARD.md` | seconds | no — same |
| `uv run python -m ml.train [--dataset … --run-name …]` | a new run dir, or the served artifacts when unnamed | minutes to ~40 min | **yes** |
| `uv run python -m ml.analyze [--run-name …]` | analysis files and `MODEL_CARD.md` | minutes | **yes** |
| `uv run python -m ml.experiments.{transfer,temporal_drift,rules_audit} …` | a record inside a run dir | 20–40 min | **yes** |
| `uv run python -m ml.promote <run>` | replaces the served model | seconds | **yes** — changes live scores |

## Invariants

- Records are the source of truth. Cards are generated from records and never hand-edited;
  `test_the_committed_card_is_what_the_committed_records_produce` pins both cards.
- Record digests are taken over LF-normalised text (the working copy is CRLF). Don't touch record line endings.
- A named run directory is never overwritten: `ml.train` refuses one that already holds `run.json`.
- `ulb_pca_v1` stays out of the production featureset registry; nothing from `tracks/ulb/` reaches `app/`.
- Published ULB-derived figures carry the ODbL notice and the method-offer pointer
  (`docs/DATA_LICENSES.md`). Push track code with or before the records it produced.
- Batch features go through the production `FeatureExtractor` (`app/fraud/features.py`). A second
  implementation would reintroduce train/serve skew.
- Thresholds are selected on validation only; analysis scores the test fold only.

## Verification

- Unit tests for the module touched (`tests/unit/test_<module>.py`), then the backend gate.
- CI lints and type-checks only the `ml/` modules listed in `.github/workflows/ci.yml`; add a new module to both lists.
- Around anything that writes into a run dir: list and hash the existing files first, then report
  exactly which files are new and that every other file is byte-identical.

## Definition of done

- [ ] Unit tests and the backend gate green
- [ ] No run record, card or served artifact changed without approval
- [ ] Cards regenerated — not edited — if and only if records changed
- [ ] Any methodology deviation recorded in the governing ADR, not silently applied
