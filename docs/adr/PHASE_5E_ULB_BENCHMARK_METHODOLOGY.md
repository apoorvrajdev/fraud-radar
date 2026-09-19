# Phase 5E — ULB Real-World Benchmark Methodology

**Status:** accepted · **Date:** 2026-09-18 · **Amended:** 2026-09-18 (observed at acquisition), 2026-09-19 (licence terms settled), 2026-09-20 (observed at execution) · **Milestone:** Phase 5E / M5 · **Scope:** the ULB track (`backend/ml/tracks/ulb/`), its manifest entry, and one keyword argument on `backend/ml/train.py`

---

## Context

Phase 5D measured the v1 pipeline on Sparkov, an externally generated synthetic corpus. M5 adds the one real dataset in Phase 5: the ULB credit-card fraud dataset, published on Kaggle as `mlg-ulb/creditcardfraud`. The publisher describes it as 284,807 transactions made by European cardholders over two days in September 2013, 492 of them fraudulent. Apart from `Time`, `Amount` and the label `Class`, every column is a PCA component (`V1`–`V28`) of inputs the publisher does not disclose.

That makes ULB real, and it makes it structurally unlike every other dataset in this repository. It has no card, customer, merchant, country, channel or calendar date. None of the history features, the rules or the entity-level contracts that Phases 5A–5D were built around can be computed from it.

This record fixes the method before the data is acquired. When it was accepted, ULB had not been downloaded, no ULB tooling existed, and no model had been trained on it. Everything only the files can settle is listed under [Unknown until acquisition](#unknown-until-acquisition) and will be recorded from the retrieved bytes, not from secondary sources. What acquisition later showed is recorded under [Observed at acquisition](#observed-at-acquisition-2026-09-18).

A pre-implementation review of the code against [`PHASE_5_PLAN.md`](../PHASE_5_PLAN.md) §17 and M5 found places where the plan cannot be followed as written, or has to be made precise:

- `CanonicalDataset.validate()` requires every transaction to reference a customer and a merchant. ULB has neither, so it cannot be adapted into the canonical schema without inventing both.
- Promotion, run verification, run analysis and the serving explainer all key off the production featureset registry or the 17 v1 names, and the Phase 5D benchmark card names its four runs explicitly. A ULB run can use none of them, and they are not bent to accept one.
- The run record and provenance contracts require timezone-aware datetimes. ULB's `Time` is a relative offset with no calendar anchor.
- The training procedure's random state is a module constant, so the plan's three seeds need a way to pass one in.
- The plan builds the matrix from `log1p(Amount)` with "fewer" tuning iterations, writes one run to `runs/ulb_baseline/` with a card named `MODEL_CARD_ULB.md`, and puts ULB in a three-way table beside the v1 results (§20, §29 item 5).

Where this record and the plan disagree, this record governs M5. Phase 5D decisions carry over only where a decision below names them.

---

## The question

> Does the evaluation method frozen for Sparkov carry over unchanged to real, anonymised card transactions, using only the features ULB publishes? And can its result be reproduced from pinned source bytes?

| Track | Data | Feature space | What it measures |
|---|---|---|---|
| In-house synthetic | This repository's generator | `v1`, 17 features | Whether the generator is learnable — a closed loop |
| Sparkov | Independent synthetic | `v1`, 12 of 17 live | Whether the v1 pipeline works on data it was not designed around |
| ULB | Real, anonymised, two days in 2013 | `ulb_pca_v1` | Whether the evaluation discipline holds on real data |

The ULB result does not show that the v1 features or the rules work on real data: neither is evaluated on it. It does not measure production readiness or performance on present-day fraud. It is not comparable to the Sparkov results, and not comparable to published ULB results that use random splits.

---

## Decisions

### 1. Tooling first; acquisition and each run authorised separately

[Phase 5D decision 1](PHASE_5D_BENCHMARK_METHODOLOGY.md#1-tooling-first-execution-is-authorised-separately) carries over. The ULB tooling is built and tested on fixture files before the corpus is acquired. Acquiring ULB, pinning its digest and running each benchmark run are separate steps, each authorised explicitly.

### 2. ULB is not adapted into the canonical schema

ULB does not become a `CanonicalDataset`, is not registered as a `DatasetAdapter`, and never passes through `FeatureExtractor` or the feature cache.

This is a deliberate exception to [Phase 5A decision 1](PHASE_5A_DESIGN.md#1-the-canonical-schema-is-the-orm-model). The canonical contract requires every transaction to belong to a customer and a merchant, and ULB identifies neither. Satisfying the contract would mean inventing a customer and a merchant — one per row, or one for the whole corpus — and every feature and rule built on those entities would then describe the invention. An adapter that fabricates entities is worse than no adapter, because it still produces numbers.

What ULB does satisfy, in full:

- the **provenance contract**: a manifest entry with a pinned SHA-256; a `DatasetProvenance` record with licence, citation, label definition, counts and preprocessing steps; and `run.json` version 3 with its test-fold identity;
- the **matrix contract**: a `LabelledDataset` — matrix, labels, timestamps, transaction ids and feature names — which is exactly what `ml.train.train_and_evaluate` consumes.

The field-by-field accounting is under [Schema accounting](#schema-accounting).

### 3. Featureset v1 is not evaluated on ULB

Of the 17 v1 features, one is derivable from ULB and sixteen are not available ([Feature accounting](#feature-accounting)). v1 is therefore not evaluated on ULB at all.

Filling sixteen columns with constants would produce a result labelled "v1 on ULB" that measures a single feature. Substituting look-alikes would be worse:

- an hour of day computed as `Time mod 86400` assumes the corpus begins at midnight, which is not published;
- transaction counts over all cards in a window are a different feature from counts over one card's history, and giving them a v1 name would claim otherwise;
- any constant fill makes a column look present when it is not.

None of these is built. Whether some of the undisclosed inputs behind `V1`–`V28` resemble a v1 feature cannot be known, and no correspondence is claimed.

### 4. ULB has its own featureset, `ulb_pca_v1`, outside the production registry

`ulb_pca_v1` is the 29 columns `V1`, …, `V28`, `Amount`, under their source names and in source order. `Time` and `Class` are not features.

It is defined in the ULB track and pinned there by a test that spells the list out literally, as [Phase 5A decision 8](PHASE_5A_DESIGN.md#8-featuresets-are-versioned-v1-is-frozen) pins v1. It is **not** added to `FEATURESETS` in `app/fraud/feature_spec.py`.

That registry means "the production extractor can build these columns from a transaction, and a model trained on them can be served". `ulb_pca_v1` fails both: no transaction the API receives carries PCA components, and the extractor cannot compute them. Keeping it out makes that true by construction rather than by convention:

- `ml.promote` refuses a run whose featureset is not registered, so no ULB run can reach the served artifacts ([decision 12](#12-ulb-is-never-promoted));
- run verification, and so `ml.analyze`, refuse it for the same reason, so no v1 tooling can mistake a ULB run for a v1 run;
- `FraudExplainer` accepts only the 17 v1 names, so no ULB model can be loaded where a v1 model is expected.

No ULB column name coincides with a v1 feature name, so no table can join ULB and v1 features by name. `v1` itself is untouched.

### 5. `Amount` is used as published

The plan builds the matrix from `log1p(Amount)`. This record uses `Amount` as published.

A tree ensemble splits on the order of a feature's values, and a strictly increasing transform does not change which rows a split can separate, so `log1p` changes nothing material for the model. Keeping the source value means every column of the matrix is a column of the file, unaltered, and the matrix can be checked against the source without knowing any transform.

The currency of `Amount` is not stated in the material this record was written from; that is confirmed at acquisition. No currency is assumed, and ULB amounts are never compared with the USD amounts of the other datasets.

### 6. `Time` orders the rows and is never a feature

The publisher describes `Time` as the seconds elapsed between each transaction and the first transaction in the dataset (wording confirmed at acquisition). It gives an order and nothing else: no calendar date, no time of day, no weekday.

- **Encoding.** Each row's timestamp is `1970-01-01T00:00:00+00:00` plus `Time` seconds. This exists only to satisfy the timezone-aware contracts of `SplitPeriod` and `DatasetProvenance`, and to give `chronological_split` the order it sorts on. The provenance record states, in `preprocessing` and `notes`, that these are elapsed-time positions and not calendar dates, and the ULB card shows them as elapsed hours. No calendar value is ever derived from them.
- **Order and identity.** Rows are sorted by `Time`, then by position in the source file. A row's transaction id is its 1-based position in the file, which stays stable because the file's digest is pinned. It is not a source identifier; ULB has none.
- **Not a feature.** In a chronological split every test-fold `Time` lies beyond every training value, so the feature could only measure how far a row sits past the end of training.
- **No temporal drift experiment.** Two days cannot support one.

### 7. Rows are validated and counted, never silently dropped

The loader refuses a file whose header is not exactly the expected columns. Rows with a missing or non-numeric value, a `Class` other than 0 or 1, or a negative `Amount` are excluded and counted by reason, as the Sparkov adapter counts its exclusions. Zero amounts are kept and counted. Any exclusion stops the work before a model is trained ([decision 15](#15-the-method-is-frozen-before-the-first-ulb-test-fold-is-scored--hard-gate)).

Exact duplicate rows are kept: removing them would change the published dataset on a guess about what they are. They are counted, along with any whose labels conflict. Because duplicates share a `Time`, a chronological split can separate a pair only at a fold boundary, and pairs that straddle one are counted.

### 8. The Sparkov procedure, unchanged

The ULB matrix goes through `ml.train.train_and_evaluate`, the function the Sparkov runs and the synthetic baseline went through, with the same settings. The hyperparameter search keeps the existing 25 iterations rather than the "fewer" the plan suggests: ULB is small enough that the full budget costs little, and an identical procedure is worth more than the time saved. The details are under [Frozen protocol](#frozen-protocol).

### 9. Three pre-registered seeds, all reported

The benchmark is run with random states 42, 43 and 44. Only the random state changes — it seeds the hyperparameter search, the shuffle of that search's cross-validation folds, and the final model. The data, the split and every other setting are identical across the three.

`ml/train.py` gains a keyword argument for the random state, defaulting to the existing constant, so every existing caller and every recorded run is unaffected. The fit record already stores the random state in `training_metadata.json`.

Random state 42 is the primary result, because it is the procedure Sparkov was run with. All three are reported; none is chosen for being the best, and they are shown side by side rather than averaged — three values do not make a distribution worth summarising.

The seeds show how much the result moves with the fit's randomness. They do not measure the uncertainty that comes from the test fold holding few frauds. That uncertainty is stated in words beside each result's fraud counts; no resampling-based interval is computed.

### 10. What is reported, and what is not

Each run reports what Phase 5D reported for a test fold, computed by the same code ([Frozen protocol](#frozen-protocol)): PR-AUC with its test prevalence, ROC-AUC, recall at 1% and 5% FPR, and precision, recall, F1 and the confusion matrix at the val-selected threshold, with the contextual fields of [5D decision 13](PHASE_5D_BENCHMARK_METHODOLOGY.md#13-contextual-reporting-fields). Calibration is measured without fitting a calibrator.

Not reported for ULB:

- **SHAP rankings.** The components are anonymised, so a ranking of one `V` column over another says nothing anyone can act on or check.
- **Segment breakdowns.** No column describes a segment.
- **A rules audit.** The rules read customers, merchants, countries and history, none of which exist.
- **A transfer measurement**, in either direction. ULB and v1 share no feature space, so scoring one track's model on the other's rows is undefined.

[5D decisions 11 and 12](PHASE_5D_BENCHMARK_METHODOLOGY.md#11-numbers-in-artifacts-interpretation-in-prose) carry over: numbers live in machine-readable run records and are never hand-edited, and ULB `metrics.json` carries no synthetic-era targets.

### 11. ULB results are never merged with v1 results

ULB results are laid out in their own generated card, `backend/ml/ULB_BENCHMARK_CARD.md`, built from the ULB run records only. The name mirrors `BENCHMARK_CARD.md`. The plan's `MODEL_CARD_ULB.md` is not used, because `MODEL_CARD.md` already names the analysis card in every run directory and the served model's card.

ULB never appears in the Phase 5D benchmark card, and never as a row or column in a table of v1 results. This supersedes the three-column headline table of plan §20 and the three-way table of §29 item 5: the README gets a separate real-data table for ULB instead.

Every ULB number is shown with its featureset, its test prevalence and its fold fraud counts. The card states that the results come from a chronological split and are not comparable to ULB results from random splits.

### 12. ULB is never promoted

No ULB run is promoted to the served artifacts, in M5 or later. The featureset guard of [decision 4](#4-ulb-has-its-own-featureset-ulb_pca_v1-outside-the-production-registry) enforces it, and a test proves that `ml.promote` refuses a ULB run. The guard is not relaxed to accommodate ULB.

### 13. Verification before recording

ULB runs are verified by a read-only ULB verifier, since the Phase 5D verifier refuses unregistered featuresets. It rebuilds the matrix from the pinned file, re-splits it, checks that the test fold holds exactly the transactions `run.json` identifies, and reproduces `metrics.json` and `calibration_metrics.json` exactly from the saved `model.json`, in an environment with the recorded library versions. A run is committed only after it has been verified.

### 14. Run names

`ulb_pca_v1_seed42`, `ulb_pca_v1_seed43` and `ulb_pca_v1_seed44`. The featureset is in the name because `ulb_v1` would read as "featureset v1 on ULB", which is exactly what this benchmark is not.

There is no development run. Sparkov had one because it could be subsampled by card; ULB has no entity to subsample by, and it is small enough to run whole.

### 15. The method is frozen before the first ULB test fold is scored — hard gate

The protocol below is fixed before `ulb_pca_v1_seed42` scores its test fold. There is no development run whose test fold could be seen first, so the gate applies to the primary run directly.

After that point, any change to the method is a deviation on the terms of [5D decision 15](PHASE_5D_BENCHMARK_METHODOLOGY.md#15-the-method-is-frozen-before-the-dev-test-fold-is-scored--hard-gate): it is recorded here with its reason, and every result it affects is labelled as produced under the changed method. A bug fix that changes a number counts. Seeds 43 and 44 are part of the frozen method and are run whatever seed 42 shows. [5D decision 16](PHASE_5D_BENCHMARK_METHODOLOGY.md#16-the-full-run-is-not-cut-down-to-save-time) carries over: the procedure is never cut down to save time.

Work stops, and the finding is reported before anything changes, if:

- the file's header, row count or digest does not match what acquisition recorded;
- any row is excluded;
- the val or test fold holds no fraud, so early stopping, threshold selection or the test metrics are undefined;
- no val threshold meets the FPR target and the fallback is used;
- a run cannot be verified.

### 16. The raw file is external and local-only

The source file lands in `backend/ml/data/raw/ulb/`, which is already gitignored, and is never committed, whatever its licence permits. What is committed is what [`DATA_LICENSES.md`](../DATA_LICENSES.md) allows for every source: digests, aggregate statistics, run records and the generated card — never rows or matrices.

Acquisition uses the existing `ml.datasets.download` command, which works from a manifest entry; no new download code is needed. The `ulb` manifest entry starts with an unpinned digest. The first retrieval's SHA-256 is pinned as a separately authorised step, after which any other bytes are refused. The licence, citation and dataset version are recorded as the source displays them on the retrieval date, and the licence's terms for derived works are recorded in `DATA_LICENSES.md` and checked against what is committed.

---

## Schema accounting

What each ULB column maps to in the canonical schema, and what the canonical schema needs that ULB does not have.

| ULB field | Canonical field | Status | Transformation | Notes |
|---|---|---|---|---|
| `Time` | `Transaction.created_at` | Available, relative only | `1970-01-01T00:00:00+00:00` plus `Time` seconds | Orders and splits rows; carries no calendar date or time of day (decision 6) |
| `Amount` | `Transaction.amount` | Available | None | Currency not stated; none assumed (decision 5) |
| `Class` | `labels[id]` | Available | Integer; must be 0 or 1 | The label, held outside any row object as in 5A decision 2 |
| `V1`–`V28` | None | No canonical field | Used as published | PCA components of undisclosed inputs; ULB-only features |
| Row position | `Transaction.id` | Derived | 1-based position in the source file | Stable because the digest is pinned; not a source identifier |
| — | `Transaction.customer_id`, `Customer.*` | Unavailable | — | No card or account key: no history, velocity or customer features, and no dormant-account rule |
| — | `Transaction.merchant_id`, `Merchant.*` | Unavailable | — | No merchant name, category, MCC, risk rating or country |
| — | `Transaction.country`, `is_card_present`, `payment_method`, `status`, `currency` | Unavailable | — | No geography or channel |

The extractor and the rules need a customer, a merchant, the customer's recent history, countries, channel, amount and time. ULB supplies an amount of unstated currency and a relative time.

## Feature accounting

| v1 feature | On ULB | Why |
|---|---|---|
| `log_amount` | Derivable | The same `log1p`, applied to `Amount`; the currency is unstated, so values are not comparable to v1's |
| `hour_of_day` | Not available | `Time` has no clock anchor |
| `is_weekend` | Not available | No calendar date |
| `is_off_hours` | Not available | Needs the hour of day |
| `is_card_present` | Not available | No channel field |
| `country_mismatch_customer` | Not available | No countries |
| `country_mismatch_merchant` | Not available | No countries |
| `tx_count_1h` | Not available | No card key |
| `tx_count_24h` | Not available | No card key |
| `log_amount_sum_24h` | Not available | No card key |
| `customer_account_age_days` | Not available | No customer |
| `customer_risk_tier_encoded` | Not available | No customer |
| `avg_amount_30d` | Not available | No card key; 30 days also exceeds the two-day span |
| `amount_zscore_30d` | Not available | No card key; 30 days also exceeds the two-day span |
| `days_since_last_tx` | Not available | No card key |
| `merchant_risk_encoded` | Not available | No merchant or category |
| `is_high_risk_category` | Not available | No category |

None available, one derivable, sixteen not available. `log_amount` is not carried into `ulb_pca_v1` under that name: `Amount` is used as published (decision 5).

---

## Frozen protocol

"Existing" refers to the code at the commit of this record.

**Data.** The file whose SHA-256 is pinned in the manifest, read by the ULB loader: the schema refused on mismatch, exclusions counted by reason, duplicates kept and counted, rows ordered by `Time` then file position, and `Time` encoded as elapsed time (decisions 6 and 7). No subsample, so `run.json` records a null seed.

**Features.** `ulb_pca_v1`: `V1`–`V28` and `Amount`, as published. No feature cache.

**Split.** The existing `chronological_split`: 70/15/15 by row count, in that order, checked by `assert_no_temporal_leakage`, with each fold's first and last timestamp written to `run.json` in the elapsed-time encoding. At the advertised 284,807 rows the folds hold 199,364, 42,721 and 42,722 rows. Transactions that share a `Time` may fall on both sides of a boundary; that is recorded, not corrected.

A stratified random split, the common choice for this dataset, is not used. It would train on transactions later than ones it is scored on, place exact duplicates in both the training and test folds, and make the procedure differ from Sparkov's. The cost of the chronological split is stated with the results: the test fold is one slice at the end of the second day, with its own prevalence and few frauds.

**Class imbalance.** No resampling, undersampling or synthetic minority rows. `scale_pos_weight` is computed from the training labels, as the existing fit does.

**Tuning.** The existing `tune_hyperparameters` on the training fold only: existing search space, 25 iterations, 4 stratified folds, with the run's random state (decision 9). Its cross-validated PR-AUC is a diagnostic, not a held-out estimate.

**Fit.** The best parameters refit on the training fold, with early stopping after 50 rounds against the val fold.

**Operating threshold.** The highest score threshold whose FPR on the val fold is at most 1%. A fallback is recorded in `threshold.json` and stops the work (decision 15).

**Test evaluation**, once per run, on the test fold, by the existing `evaluate_test_fold`:

- PR-AUC, shown with the test-fold prevalence, and ROC-AUC;
- recall at 1% and 5% FPR, labelled as points on the test ROC curve;
- precision, recall, F1 and the confusion matrix at the val-selected threshold;
- the contextual fields of 5D decision 13: test prevalence, fraud counts per fold, the realised test FPR at the operating threshold, and the label-delay note.

**Calibration.** Measured on the test fold with `ml/analysis/calibration.py`, from the same scores the metrics are computed from: Brier score, ECE, their positive-class variants and the reliability curve. No calibrator is fitted. Two reasons the positive-class values are hard to read are stated with them: `scale_pos_weight` inflates scores by design, and few frauds leave the positive-class bins sparse.

**Live features.** Counted on the training fold and recorded, as [5D decision 6](PHASE_5D_BENCHMARK_METHODOLOGY.md#6-live-features-are-counted-not-asserted) requires. All 29 are expected to vary; that is checked, not assumed.

**Records.** Per run, in `backend/ml/artifacts/runs/<run>/`:

- `run.json` version 3: dataset provenance with the file digest, licence, citation and preprocessing; `featureset_version: ulb_pca_v1`; fold periods; test-fold identity; code commit; library versions;
- `training_metadata.json`: fold sizes and fraud rates, hyperparameters, random state, `scale_pos_weight`, best iteration, live features, and the SHA-256 of `model.json`;
- `metrics.json`, `threshold.json`, `feature_list.json` and `calibration_metrics.json`;
- `ulb_quality_report.json`: row, fraud, exclusion and duplicate counts, `Time` coverage and ties, and each fold's size, frauds and elapsed-time span.

`model.json` and plots stay gitignored; the digest in `training_metadata.json` is the link to the model.

**Verification.** Decision 13, before any run is committed.

**Promotion.** Never (decision 12).

---

## Unknown until acquisition

These are unknown, not undecided. Each is recorded from the retrieved bytes, or from the source's own metadata on the retrieval date, and none is filled in from secondary sources.

| Unknown | Why it matters | Answered by |
|---|---|---|
| The licence as the source displays it, the citation it asks for, and its terms for derived works | The provenance record; what may be committed | Kaggle dataset metadata at retrieval |
| The dataset version or last-updated date | `DatasetProvenance.version` | Kaggle dataset metadata at retrieval |
| The files in the download, and each file's size and SHA-256 | The manifest entry and digest pinning | The Kaggle files listing, then the retrieved bytes |
| The exact header and value formats, including whether `Class` is quoted | The loader's schema check | Retrieved file |
| Row and fraud counts | The publisher's 284,807 and 492 are advertised, not observed | Retrieved file |
| Whether `Time` starts at 0, is in whole seconds and is in file order; the sizes of tied groups | Row order, identity and boundary ties (decision 6) | Retrieved file |
| Missing or non-numeric values; negative and zero amounts | Exclusions (decision 7) | Retrieved file |
| Exact duplicate rows, conflicting labels among them, and pairs across a fold boundary | Decision 7 | Retrieved file |
| Frauds in each fold, and each fold's elapsed-time span | Whether val and test each hold a fraud (decision 15); how few frauds the results rest on | Quality report |
| The currency of `Amount` | Decision 5; expected to remain unstated | Dataset description at retrieval |
| Which rows the publisher fitted the PCA on | An accepted limitation either way; expected to remain unpublished | Dataset description at retrieval |
| Whether the Kaggle CLI can list the dataset from this account | Automatic or manual acquisition | First acquisition attempt |

### Observed at acquisition (2026-09-18)

`creditcard.csv` was retrieved into `backend/ml/data/raw/ulb/` and verified read-only against the manifest. A second download from Kaggle the same day, into a temporary directory outside the repository, was byte-identical to it, with the same SHA-256 from two independent implementations; that copy was then deleted. The digest was pinned in the manifest afterwards, as a separately authorised step (decision 1).

Nothing was split, featurised, trained or scored. Each answer below comes from the retrieved bytes or the source's own metadata, read on this date.

| Unknown | Observed |
|---|---|
| Licence, citation and terms for derived works | `DbCL-1.0`, from the Kaggle dataset API and printed again by the Kaggle CLI at download; the dataset page names the same selection "Database: Open Database, Contents: Database Contents". The citation list the publisher asks for is recorded in [`DATA_LICENSES.md`](../DATA_LICENSES.md). **Settled 2026-09-19**, from the full licence texts: the DbCL covers the contents and requires compliance with the ODbL (DbCL §2.2), which covers the database. Committed ULB records are ODbL Produced Works, and each carries the notice below. The terms are recorded in [`DATA_LICENSES.md`](../DATA_LICENSES.md). |
| Dataset version or last-updated date | The dataset metadata carries neither. The Kaggle files API dates `creditcard.csv` 2019-09-20 00:04:39. The pinned digest is what identifies the bytes. |
| Files, sizes and SHA-256 | The download is a 66.0 MB zip holding one file, `creditcard.csv`: 150,828,752 bytes, exactly the advertised size, SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`, now pinned. |
| Header and value formats | ASCII, no byte-order mark, LF line endings, a final newline, no blank lines, and 31 fields on every row. The header is exactly `Time`, `V1`–`V28`, `Amount`, `Class`, every name double-quoted. In data rows only `Class` is quoted, as `"0"` or `"1"`. `Amount` has at most two decimal places. `Time` is written as a plain integer on every row but one, which writes 100000 as `1e+05`. |
| Row and fraud counts | 284,807 rows and 492 frauds (0.1727%), the publisher's figures exactly. The fold sizes stated under [Frozen protocol](#frozen-protocol) therefore apply as written. |
| `Time` | Runs from 0 to 172,792 seconds (47.998 hours), in whole seconds, and the file is already in `Time` order. 124,592 distinct values; 239,644 rows share their `Time` with at least one other row, and the largest group sharing one value has 36 rows. |
| Missing or non-numeric values; negative and zero amounts | None missing, non-numeric or non-finite in any column. No negative amounts. 1,825 zero amounts, kept and counted (decision 7). No row would be excluded. |
| Exact duplicates and conflicting labels | 773 groups of identical rows: 1,854 rows in all, 1,081 of them repeating an earlier row, the largest group 18 rows; 13 of the groups are fraud rows. No two rows with identical feature values carry different labels. Whether any pair straddles a fold boundary is answered by the quality report. |
| Frauds in each fold, and each fold's elapsed-time span | Not observed at acquisition; answered by the quality report, which splits the data. |
| The currency of `Amount` | Not stated in the dataset description. |
| Which rows the PCA was fitted on | Not stated in the dataset description. |
| Whether the Kaggle CLI can list the dataset | Yes: Kaggle CLI 2.2.4 listed and downloaded it. |

**What acquisition changes.** Nothing in the method. No row is excluded, so the exclusion stop in decision 15 is not triggered, and every duplicate and zero amount stays in, as decision 7 requires. Two properties of the file bind the loader rather than the method: `Time` must be read as a number, not matched as an integer string, because of the one `1e+05`; and `Class` arrives quoted.

**What the licence terms change.** Nothing in the method either. Everything decision 16 commits is permitted, and the raw file and the matrix stay local. They add one requirement for what is committed: every published ULB-derived record carries the ODbL §4.3 notice, and every record computed through the ULB loader or matrix is accompanied by the ODbL §4.6 method offer. Both are worded in [`DATA_LICENSES.md`](../DATA_LICENSES.md). The figures in this section are among those records:

> Contains information from the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) database of the Machine Learning Group, ULB, which is available under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/); its contents are under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

---

## Observed at execution (2026-09-19)

The three pre-registered runs were trained in order from the pinned file, `ulb_pca_v1_seed42` first, at the commit each `run.json` records as its code version, `da1b7aa`. Only the random state differed between them. Each run was verified with `ml.tracks.ulb.verify` before its records were committed, and [`backend/ml/ULB_BENCHMARK_CARD.md`](../../backend/ml/ULB_BENCHMARK_CARD.md) lays the results out from those records. Every value below is read from them.

| Left to execution | Observed |
|---|---|
| Frauds in each fold, and each fold's elapsed-time span | Train: 199,364 rows, 384 frauds, `Time` 0–132,928 s. Val: 42,721 rows, 56 frauds, 132,929–151,328 s. Test: 42,722 rows, 52 frauds, 151,328–172,792 s. The same in all three runs. |
| Transactions sharing a `Time` across a boundary, and duplicates straddling one | The val and test folds share `Time` 151,328 s: 2 of its rows are in val and 1 in test. The train and val folds share none. No exact-duplicate group straddles either boundary. |
| Decision 15 stops | None. No row was excluded, the val and test folds each hold fraud, every run's threshold met the 1% FPR ceiling on its val fold without the fallback, and every run verified. |
| Live features | All 29 are live in every run's training fold. |
| Reproduction | For each run, the matrix rebuilt from the pinned file gives the recorded folds and exactly the test-fold transactions `run.json` identifies, and the saved model reproduces `metrics.json` and `calibration_metrics.json` exactly. |

**What the results say.** They answer [the question](#the-question) on both counts. The evaluation method frozen for Sparkov carried over to real, anonymised card transactions unchanged: the same procedure, search budget, split and threshold rule ran with only the random state varied, and no stop condition was met. Each result is also reproduced exactly from the pinned source bytes and the committed records.

On the primary run, random state 42, the test PR-AUC is 0.7670 against a test prevalence of 0.0012; the repeats under random states 43 and 44 score 0.7569 and 0.7751. The operating thresholds the runs chose on their val folds differ — 0.0396, 0.0155 and 0.0027 — so a threshold is read only with the run that chose it, and each run's realised FPR stays under the 1% ceiling on val and on test. At those thresholds the runs catch 43, 40 and 44 of the test fold's 52 frauds.

**What they do not say.** Every figure rests on 52 test frauds in one slice at the end of the second day, so the differences between the random states amount to a handful of frauds; the seeds measure the fit's randomness, not that uncertainty. The results do not show that featureset v1 or the rules work on real data, since neither is evaluated here. They are not comparable to the Sparkov or synthetic results, nor to ULB results from random splits, and they say nothing about production performance or present-day fraud. No calibrator was fitted, and the positive-class calibration values are read with the two caveats recorded beside them.

> Contains information from the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) database of the Machine Learning Group, ULB, which is available under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/); its contents are under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/).

---

## Not done in M5

- No ULB run is promoted, and the promotion guard is not relaxed (decision 12).
- No production feature, featureset `v1`, the extractor, the rules or the served model changes to accommodate ULB.
- The test fold is scored once per pre-registered run. Tuning, threshold selection and early stopping never see it, and a recorded run is never retrained under its name.
- No post-hoc reruns: the seed set is 42, 43 and 44, fixed here. Exploratory analyses made while reviewing results are not written into any artifact.
- No ULB number is placed beside, merged with or averaged with a v1 result, and no transfer is measured between the tracks.
- No claim of real-world production performance. The results describe a two-day, anonymised 2013 corpus under one chronological split.
- No resampling, synthetic minority rows, fitted calibrator, resampling-based intervals, hour-of-day proxy, drift experiment, rules audit, SHAP ranking or segment breakdown.

---

## Implementation sequence

Each step is reported and approved before the next starts. Steps marked *gate* are authorised separately.

| Step | Work | Lands in |
|---|---|---|
| M5A | This record | `docs/adr/`, with pointers from the plan and the README |
| M5B | The `ulb` manifest entry with an unpinned digest; its licence and provenance terms | `backend/ml/data/manifest.json`, `docs/DATA_LICENSES.md` |
| *gate* | Acquisition and read-only verification, then digest pinning | The manifest, once pinned |
| M5C | Reading and validating the source file with its provenance; the quality report; CI gating of the ULB track | `backend/ml/tracks/ulb/load.py`, `quality.py`, `.github/workflows/ci.yml` |
| M5D | The `ulb_pca_v1` matrix; tests proving a ULB run can be neither promoted nor analysed as v1 | `backend/ml/tracks/ulb/matrix.py` |
| M5E | The random-state keyword; the ULB trainer and verifier, tested on fixtures | `backend/ml/train.py`, `backend/ml/tracks/ulb/train.py`, `verify.py` |
| *gate* | Run `ulb_pca_v1_seed42`, then `_seed43` and `_seed44`, each verified; their records committed | `backend/ml/artifacts/runs/` |
| M5F | The generated ULB card, and the test pinning it to its records | `backend/ml/tracks/ulb/card.py`, `backend/ml/ULB_BENCHMARK_CARD.md` |
| M5G | What acquisition and execution observed, and its interpretation; README and architecture updates | This record, `README.md`, `docs/ARCHITECTURE.md`, `docs/DATA_LICENSES.md` |

Each code step carries its own unit tests, on fixture files only; no test needs the real corpus.

---

## Consequences

**Good.** The one real dataset in Phase 5 is measured with the same procedure, metrics and records as Sparkov, from pinned bytes, with no fabricated field. It cannot reach the serving path or be read as a v1 result, because the code that would allow either refuses it.

**Cost.** ULB needs its own loader, verifier and card, because the Phase 5D tools are bound to the production registry and stay that way. The result says nothing about the v1 features, the rules or the served model.

**Accepted limitations**, stated with the results rather than engineered away:

- **Unknown PCA fit.** The publisher does not say which rows the PCA was fitted on. If it was fitted on all of them, the components carry unsupervised information from the test period. No labels are involved, and nothing downstream can undo it.
- **No card identifiers.** Whether a card appears in several folds cannot be checked, and rows are treated as independent, although frauds on one card probably are not.
- **One chronological slice.** The test fold is one slice of the second day. Its prevalence and fraud count are what they are, and with few frauds a handful of rows can move the metrics substantially.
- **Reused val fold.** The val fold is used both for early stopping and for threshold selection, as in 5D, so the realised val FPR is slightly optimistic.
- **Label delay.** The 5D label-delay note applies.
- **Age and scope.** The data is from 2013, covers two days, and comes from European cardholders only.

---

## Deferred

| Deferred | Where |
|---|---|
| Frauds in each fold, each fold's elapsed-time span, and duplicates straddling a fold boundary | The ULB quality report (M5C) |
| The results and their interpretation | This record and the ULB card, in M5G, from the committed run records |
| Multi-currency FX | M6 |
| Featureset version in `feature_list.json`, `/explain` and the model endpoint | M7 |
