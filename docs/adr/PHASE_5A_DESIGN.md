# Phase 5A — Data & Benchmark Architecture

**Status:** accepted · **Date:** 2026-09-16 · **Scope:** `backend/ml/datasets/`, `backend/ml/paths.py`, `backend/ml/runs.py`, `backend/app/fraud/feature_spec.py`

---

## Context

The model is trained and evaluated on one dataset: the in-house synthetic generator. Every headline number in the README comes from data produced by the same code that the features were designed around. That is a closed loop — it measures whether the generator is learnable, not whether fraud is detectable — and Phase 5 exists to break it by bringing in externally generated and real-world benchmarks.

Before any external data arrives, the repository needs somewhere for it to land that does not bend the production system out of shape. Without that, the first external dataset brings a parallel schema, a second feature-extraction path, and a set of numbers nobody can trace back to specific bytes.

Phase 5A is that groundwork and nothing else: contracts, provenance, layout, versioning. No dataset is downloaded, no adapter is written, no model logic changes.

---

## Decisions

### 1. The canonical schema is the ORM model

> External datasets are adapted **into** the production schema. Nothing is adapted out of it.

A `CanonicalDataset` holds transient `Customer` / `Merchant` / `Transaction` instances — the exact types `FeatureExtractor` and the rules engine already consume.

The alternative, a dedicated offline dataclass schema, sounds cleaner and is worse. It requires a mapping layer between "what we train on" and "what we score", and that layer is precisely where train/serve skew lives: it is the code nobody updates when a field's meaning shifts. Reusing the ORM types deletes the layer, so there is nothing to keep in sync.

The instances are never added to a `Session`. `CanonicalDataset.validate()` checks that with `object_session()`, because an object that quietly joined a session would make the offline path start issuing queries — reintroducing the coupling by accident.

### 2. Labels live outside the transaction objects

`labels` is a separate `transaction_id -> 0 | 1` mapping, never an attribute on `Transaction`.

This makes target leakage structurally impossible rather than merely discouraged. A feature function receives a `Transaction`; if the target is not reachable from that object, no amount of careless feature engineering can read it. The usual defence — "we remember not to use the label column" — is a convention, and conventions fail quietly.

### 3. Adapters are a protocol, resolved by name

`DatasetAdapter` is a `Protocol` with `name` and `load(root, *, max_entities, seed)`. `ml/datasets/registry.py` resolves a name to an adapter.

Training, quality reporting and evaluation therefore never grow `if dataset == "sparkov"` branches. Adding a benchmark means adding one module that registers itself.

`max_entities` is deliberately generic where the plan said `max_cards`. Subsampling is by entity — whatever owns a history in that source: a card, a customer, an account — and the adapter records which in the `Subsample` record. Callers stay dataset-agnostic; only the adapter knows what an entity is.

Registration refuses to replace an existing name with a different adapter, because a run's recorded `dataset_name` has to keep identifying whatever actually produced it.

### 4. Invariants are enforced at load, not assumed

`validate()` rejects: transactions out of chronological order, naive timestamps, duplicate ids, missing or orphan labels, non-binary labels, unresolvable customer/merchant references, and session-attached objects.

Each of these fails silently otherwise. An unsorted stream produces velocity features computed partly from the future and a model that looks excellent. A missing label shifts the label column against the feature matrix. Both yield numbers, not errors — which is why they are checked at the boundary where a fix is still cheap.

### 5. Provenance travels with the data

`DatasetProvenance` records name, version, origin (synthetic vs real), source URL, citation, licence, the label field and what `1` means, schema version, retrieval time, per-file SHA-256, row and fraud counts, time coverage, subsample record, preprocessing steps, and notes.

A metric without provenance is an anecdote. These fields let a reader answer *which bytes, which licence, which slice, which label definition* without trusting whoever ran the job. File hashes matter most: raw data is never committed, so the hash is the only durable link between a model and its inputs.

Origin is explicit because it changes how a number may be read. A strong score on generated data says the generator is learnable; the same score on real data says something about fraud.

### 6. Four stages, four locations

```
data/raw/         source files exactly as downloaded        never committed
data/cache/       canonical datasets, feature matrices      never committed
artifacts/runs/   per-run metrics and metadata              committed (minus model.json)
artifacts/        the promoted run — what the API loads     committed (minus model.json)
```

Caches are disposable by construction: deleting `data/cache/` must cost time and nothing else. Run outputs are the record. Exactly one directory is what production serves, and getting there is an explicit promotion step rather than a side effect of training.

Run names are validated against `^[a-z0-9][a-z0-9._-]*$` rather than sanitised. A name needing cleanup is a caller bug, and `../..` should never reach a `mkdir`.

### 7. Two records per run, deliberately not merged

- `run.json` (new) — the inputs: dataset provenance, featureset version, seed, commit, library versions, fold boundaries.
- `training_metadata.json` (existing, unchanged) — the fit: fold sizes, fraud rates, chosen hyperparameters.

One describes what went in, the other what happened during fitting. Keeping them apart means neither has to grow the other's fields, and a benchmark run that trains no model still has a complete input record.

### 8. Featuresets are versioned; v1 is frozen

`FEATURESETS` maps a version to its exact column order. `v1` is the 17 production features, unchanged in membership and order; `FEATURE_NAMES` remains as the alias the app already imports.

XGBoost is positional. If `v1` ever reorders, every artifact trained before that change becomes silently wrong — column 3 means something new, and the model produces predictions rather than errors. `tests/unit/test_feature_registry.py` therefore pins `v1` as a hand-written literal. The duplication is the point: editing the spec alone fails CI.

A future `v2` is a new registry entry, never an edit to `v1`.

### 9. The production extractor stays the single source of truth

Phase 5A does not add a batch feature path, and when one arrives (5C) it will not reimplement feature logic. It will feed `FeatureExtractor.extract()` transient objects with a pre-built history window, so the offline matrix comes from the same code that scores live traffic.

A vectorised pandas rewrite would be far faster and would need its own proof of equivalence, permanently. Paying the runtime cost once per cached run is cheaper than owning two implementations of the same semantics.

---

## Consequences

**Good.** External datasets plug in without touching scoring. Numbers become traceable to input bytes. Feature contracts are named, so an artifact can state what it was trained against. Caches are safe to delete.

**Cost.** The contracts exist before the adapters that will use them, so parts of this layer are exercised only by tests until 5B. That is the trade accepted for not retrofitting provenance onto a dataset that has already been trained on.

**Risk accepted.** `CanonicalDataset` holding ORM types couples the offline path to the ORM models. A schema change would ripple into adapters. This is deliberate: that ripple is the signal that a canonical field changed, and is preferable to a mapping layer that absorbs the change silently and drifts.

---

## Explicitly deferred

Phase 5A stops at the architecture. Not built here, in plan order:

| Deferred | Milestone |
|---|---|
| `scripts/download_data.py`, `data/manifest.json`, Kaggle retrieval | 5B |
| Sparkov adapter, category taxonomy map, entity subsampling implementation | 5B |
| Quality report generation | 5B |
| Batch feature builder, `.npz` cache, golden parity test | 5C |
| `train.py` dataset dispatch, `promote.py`, per-run artifact writing | 5D |
| ULB track | 5E |
| FX enrichment and its migration | 5F |
| Model badge, `/api/v1/model`, demo refresh | 5G |

Nothing in Phase 5A changes fraud scoring, thresholds, the v1 feature set, the database schema, the simulator, or the deployed demo.
