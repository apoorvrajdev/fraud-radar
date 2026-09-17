# Phase 5C — Batch Feature Extraction and Production Parity

**Status:** accepted · **Date:** 2026-09-16 · **Scope:** `backend/ml/features/`

---

## Context

Phase 5B can turn an external benchmark into a `CanonicalDataset`. Training and evaluation need a feature matrix from it, and the obvious way to build one — a vectorised pandas implementation of the 17 features — is the wrong way.

Two implementations of the same features do not stay identical. They start identical, then one gains a bug fix, a rounding change, or a new edge case, and the divergence surfaces as a model that scores well offline and behaves differently in production. That failure is quiet, expensive, and extremely common.

So the question for 5C was never "how do we compute features quickly", it was "how do we make a second computation impossible".

## Decisions

### 1. The batch builder computes nothing

`build_feature_matrix` contains no feature formula. It reproduces the *context* the live scoring service assembles, then calls the same `FeatureExtractor.extract()` the API calls and collects the result.

The live path ([`app/services/scoring._load_context`](../../backend/app/services/scoring.py)) loads the customer, the merchant, and that customer's transactions in `[tx.created_at - 180d, tx.created_at)`, then passes all three into `extract()`. Given those, `extract()` aggregates in memory and never queries. The batch builder supplies exactly the same three things from the canonical dataset, maintaining one bounded window per customer as it walks the stream in chronological order.

### 2. The database handle is a sentinel that raises

`extract()` takes a `Session` positionally and falls back to SQL for anything the caller withholds. The batch path withholds nothing, so it passes `_NoDatabase`, whose `__getattr__` raises.

This converts a whole class of silent divergence into a loud failure. If someone later adds a feature that queries, or removes a parameter that lets the caller pre-supply context, the batch run stops with a message naming the attribute it tried to reach — instead of producing numbers computed under different semantics.

### 3. The window is 180 days, pinned by test rather than by import

`HISTORY_WINDOW` is defined in the batch module and a test asserts it equals `scoring._RECENT_HISTORY_WINDOW`. Importing the serving constant directly would have been shorter, but it would make the offline path depend on the serving package — the coupling Phase 5A removed. The test gives the same guarantee: change one side and CI fails.

### 4. Parity is asserted element by element, on a fixture where nothing is constant

The golden test computes each fixture transaction's vector twice — once through the batch builder, once through the SQL-backed serving path with its 180-day window — and compares all 17 values with exact float equality. Approximate comparison would let a real formula change hide inside a tolerance.

A second assertion checks that **no feature is constant across the fixture**. Parity on a column that never varies proves nothing, and that is an easy way for a parity suite to look thorough while testing almost nothing.

The fixture is built to make every feature move: amounts across three orders of magnitude, weekend and 03:00 rows, country mismatches on both the customer and merchant side, a four-transaction burst inside the 1-hour window, a dormant card returning after 400 days, card-present and card-not-present rows, and merchants spanning LOW/MEDIUM/HIGH risk including a high-risk category.

### 5. The known divergence is pinned, not hidden

The existing synthetic training path (`ml/data.load_dataset_with_csv_labels`) calls `extract()` **without** `recent_transactions`, so its history is unbounded. For a gap longer than 180 days this yields a real day count where the serving path yields the `999` sentinel.

`days_since_last_tx` is the only feature affected, and a test asserts exactly that: `398.0` unbounded versus `999.0` served, with every other value identical. Recording it as a test rather than a footnote means it cannot quietly become a second difference later.

The synthetic path is left as it is. Changing it would alter the inputs the currently served model was trained on, which is a retraining decision, not a refactor.

## Feature cache

Extraction is pure Python, one production call per row. That cost is paid once per dataset rather than once per experiment.

**Format:** `.npz`, read with `allow_pickle=False` so a cache file cannot become an execution vector.

| Array | Contents |
|---|---|
| `X` | float64 matrix, columns in frozen v1 registry order |
| `y` | int64 labels, in their own array — never attached to a transaction |
| `transaction_ids` | source ids, so any row traces back to its transaction |
| `timestamps` | int64 epoch microseconds, exact and pickle-free |
| `feature_names` | the column order as written |
| `metadata_json` | the record below, JSON with sorted keys |

**Metadata:** cache version, featureset version, feature names, dataset name/version/origin, source file digests, subsample record, row and fraud counts, fingerprint, build time.

**Fingerprint:** a SHA-256 over dataset name, version, schema version, source file digests, subsample record, featureset version and cache version, truncated to 16 hex characters and embedded in the filename. Change any input and the run misses the cache rather than silently answering from an old matrix. Row counts are deliberately excluded — they are a consequence of the inputs, and including them would make an identical rebuild miss its own cache.

**Refusals on load:** wrong cache version, featureset mismatch, reordered columns, metadata/array disagreement, inconsistent row counts, duplicate ids, missing arrays, unreadable archive. A cache that returns the wrong matrix is worse than no cache, because the run still produces numbers.

Caches live under `ml/data/cache/` and are gitignored. Deleting them costs time and nothing else.

## Consequences

Benchmark features are, by construction, the features production computes. The cost is speed: a vectorised rewrite would be far faster and would need its own permanent proof of equivalence. Paying once per dataset, cached, is the cheaper trade.

Row order follows the canonical chronological order, so a row index maps to a transaction id and back, and `chronological_split` consumes the timestamps unchanged.

## Deferred

Not built here: training on an external dataset, evaluation, metrics, drift, promotion (all 5D); the ULB track (5E); FX (5F). The Phase 5B pre-benchmark findings that gated the first real Sparkov run when this record was accepted — the two timestamp parsers, `format="mixed"` cost, full-corpus memory, the unpinned-size dead end, and deciding whether `trans_date_trans_time` or `unix_time` is authoritative — have since been settled. Timestamps are parsed once, with one explicit format, which removes both the second parser and the `format="mixed"` cost. An unsubsampled load above 500,000 rows must be asked for with `--full-corpus`. A first acquisition reports a size difference instead of refusing it, and both files' digests are now pinned. [Phase 5D decision 17](PHASE_5D_BENCHMARK_METHODOLOGY.md#17-on-sparkov-trans_date_trans_time-is-the-only-clock) makes `trans_date_trans_time` the only clock. The actual memory and runtime of the full corpus are known only once it runs, and fall under [Phase 5D decision 16](PHASE_5D_BENCHMARK_METHODOLOGY.md#16-the-full-run-is-not-cut-down-to-save-time).
