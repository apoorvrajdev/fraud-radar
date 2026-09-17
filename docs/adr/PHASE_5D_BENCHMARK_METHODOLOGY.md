# Phase 5D — Benchmark Methodology

**Status:** accepted · **Date:** 2026-09-16 · **Amended:** 2026-09-17 (decision 17; observed at acquisition; observed at execution) · **Milestone:** Phase 5D / M4 · **Scope:** `backend/ml/train.py`, `backend/ml/analyze.py`, `backend/ml/promote.py`, `backend/ml/experiments/`

---

## Context

Phase 5C stops at a cached feature matrix. Phase 5D turns it into numbers: a synthetic baseline, a model trained and evaluated on Sparkov, a cross-generator transfer measurement, a temporal drift experiment, and a rules audit.

A benchmark's method has to be fixed before its test data is scored. Once a test fold has been seen, every later adjustment — another threshold rule, a different tuning budget, a dropped feature — is chosen with knowledge of the answer, and the number that comes out is no longer a held-out measurement. This record fixes the method first. When it was accepted, nothing in it had been run: no Sparkov data had been downloaded, no model trained, no metric produced. What acquisition later showed is recorded under [Observed at acquisition](#observed-at-acquisition-2026-09-17).

A pre-implementation review of the code against [`PHASE_5_PLAN.md`](../PHASE_5_PLAN.md) found places where the plan cannot be followed as written:

- The plan copies the drift experiment's hyperparameters from the main run, whose tuning data reaches into 2020 — the year the drift experiment evaluates.
- `rule_dormant_account_high_value` reads `customer.created_at`. Sparkov customers have no account-open timestamp, so the rule raises on them; the plan expected it merely never to fire.
- Every Sparkov category maps to a LOW or MEDIUM taxonomy entry, so `is_high_risk_category` is constant as well. The plan's "13 of 17 live features" is expected to be 12.
- The plan puts dataset provenance in `training_metadata.json`; [`PHASE_5A_DESIGN.md`](PHASE_5A_DESIGN.md) puts it in `run.json`.
- The plan's comparison table includes ULB, which is M5.

Where this record and the plan disagree, this record governs Phase 5D.

---

## Decisions

### 1. Tooling first; execution is authorised separately

Phase 5D tooling is built and tested on fixtures before any real data is acquired. Acquiring Sparkov, pinning its hashes, and running each benchmark are separate steps, each authorised explicitly.

Tooling proven on fixtures that then fails on real data has failed for a data reason, not a code reason. Keeping the two apart is what makes that failure diagnosable.

### 2. Promotion is a mechanism, not an outcome

`python -m ml.promote <run>` is built and tested in 5D, and not run as part of it. Replacing the served model requires explicit approval after the benchmark results have been reviewed.

A Sparkov-trained model learns nothing from the five features that are constant on Sparkov, and the traffic the live demo scores is not Sparkov. Whether to serve it is a decision to make with the numbers in hand, not a side effect of finishing a milestone.

The promotion guard reads `featureset_version` from the run's `run.json` and refuses any version absent from the featureset registry. The serving explainer is not changed in 5D; carrying the featureset version into `feature_list.json`, `/explain` and the model endpoint remains M7 work.

### 3. The drift model is tuned on its own training period only

The temporal drift experiment trains on 2019-01 → 2019-10, selects its threshold on 2019-11 → 2019-12, and evaluates each month of 2020. Its hyperparameters come from the same tuning procedure as the main run, applied to the 2019-01 → 2019-10 fold alone. No hyperparameter selected with any 2020 data is used.

Reusing the main run's hyperparameters would import a choice made on data up to mid-2020 into a model scored on 2020. The leak is small, but the experiment exists to measure how a model behaves on a future it could not have seen, so it gets none.

### 4. The dormant-account rule is reported as not evaluable

On Sparkov, `dormant_account_high_value` is reported as **not evaluable: no account-open timestamp**. The other five rules are evaluated and reported.

The rule compares transaction time with the account-open time. Sparkov has no account-open date, and inventing one — the first observed transaction, the start of the dataset — would fabricate the very quantity the rule tests. Production rules are not modified to accommodate a dataset.

Evaluability is decided from the data (no customer carries an account-open timestamp), not by catching the exception the rule raises. Suppressing exceptions would equally hide a genuine failure in any other rule.

Any rules-only decision outcome in the audit is therefore computed from five rules, and says so.

### 5. Provenance in `run.json`, fit facts in `training_metadata.json`

This follows decision 7 of [`PHASE_5A_DESIGN.md`](PHASE_5A_DESIGN.md) and supersedes plan §14.

- `run.json` holds the inputs: dataset provenance and file digests, subsample record and seed, featureset version, split periods, code commit, library versions.
- `training_metadata.json` holds the fit: fold sizes and fraud rates, chosen hyperparameters, and the remaining facts of the fit — model seed, `scale_pos_weight`, best iteration, early-stopping rounds, tuning iterations and folds. Additions to it are additive.

### 6. Live features are counted, not asserted

The live-feature count is computed from each run's feature matrix and reported with the names of the constant columns. It is not hard-coded anywhere.

A feature is *live* in a run when it takes more than one distinct value in that run's training fold: a column that never varies in training contributes nothing to the model. The count over the whole matrix is reported alongside it when the two differ.

Sparkov is expected to show 12 live features, with `country_mismatch_customer`, `country_mismatch_merchant`, `customer_account_age_days`, `customer_risk_tier_encoded` and `is_high_risk_category` constant. That expectation is checked against the matrix, not assumed.

### 7. Three results, never merged

| Result | Model trained on | Evaluated on | Threshold selected on | Written to |
|---|---|---|---|---|
| Synthetic baseline | Synthetic train fold | Synthetic test fold | Synthetic val fold | `runs/synthetic_v1/` |
| Sparkov in-domain | Sparkov train fold | Sparkov test fold | Sparkov val fold | `runs/sparkov_v1_200cards/`, `runs/sparkov_v1_full/` |
| Cross-generator transfer | Synthetic (`synthetic_v1`, not retrained) | The Sparkov test fold — the same rows the in-domain run is scored on | Nothing is selected on Sparkov | The Sparkov run directory whose test fold it scores |

Each is reported as its own row, stating what it was trained on and evaluated on. No figure combines them. ULB is M5 and is not part of this comparison.

The floor of PR-AUC is the positive rate, and the datasets have different positive rates, so every PR-AUC is shown with its test-fold prevalence.

### 8. Transfer uses the reproducible synthetic re-run

The transfer source is `runs/synthetic_v1` — the synthetic pipeline re-run into a run directory — not the artifact currently served, because only the re-run carries a run record. Every transfer output names its source run and the Sparkov run whose test fold it scores.

Threshold-free results — PR-AUC, ROC-AUC, recall at fixed FPR — are the transfer measurement. A confusion matrix may be reported only at the synthetic model's own threshold and is labelled as such. Selecting a threshold on Sparkov data would turn a transfer measurement into an adaptation.

### 9. Segments only where they mean something

Country segmentation is not reported for Sparkov: every row is US, so the breakdown is a single bucket. Category and card-present breakdowns are included only if they can be produced within the 5D analysis work without reworking `ml/analysis/`; otherwise they remain the NICE-TO-HAVE item in plan §27.

### 10. Every rules audit states its row population

The final rules audit runs over the full Sparkov corpus. It may also run over the dev run's data, including its test fold, to validate the tooling. Every audit output states its row population: which run, which rows, how many, and which period.

Rule precision depends on prevalence, and prevalence differs between the corpus and any fold, so an unlabelled audit number cannot be read correctly.

### 11. Numbers in artifacts, interpretation in prose

Numerical results live in machine-readable run artifacts — `metrics.json`, `calibration_metrics.json`, `drift_metrics.json`, `rules_audit.json` and the like — and are never hand-edited. Human interpretation — what drifted, why the rules transfer only partially, what the transfer number means — lives in the generated `MODEL_CARD.md` and in this decision record.

### 12. Benchmark metrics carry observed results only

Sparkov `metrics.json` does not contain the synthetic-era targets (`target_pr_auc`, `target_recall_at_1pct_fpr`). They were set against the in-house generator and mean nothing on another dataset; a benchmark file records what was measured. The existing synthetic training path is unchanged.

### 13. Contextual reporting fields

Every result in the comparison is reported with:

- the test-fold prevalence;
- fraud counts for the train, val and test folds;
- the realised FPR on the test fold at the operating threshold (for transfer, the synthetic threshold, per decision 8);
- a label-delay note: the chronological split treats every training label as known at the val boundary, whereas real fraud labels arrive days to months later through chargebacks and investigations, so the results are optimistic in a way this benchmark does not measure.

These describe the conditions a number was measured under. They are not additional performance metrics.

### 14. Run names

`synthetic_v1`, `sparkov_v1_200cards` and `sparkov_v1_full`. Each Sparkov run's quality report is written into its own run directory. The adapter CLI's default run name (`sparkov_v1`) is not used for benchmark runs, because two reports written under it would overwrite each other.

### 15. The method is frozen before the dev test fold is scored — hard gate

The protocol below is fixed **before** the `sparkov_v1_200cards` test fold is scored for the first time.

After that point, any change to the method before `sparkov_v1_full` is a deviation. It is recorded in this decision record with its reason, and every comparison it affects is labelled as produced under the changed method. A bug fix that changes a number counts as a deviation.

The reason is specific to how the runs relate. The 200 dev cards are a subset of the full corpus, so the dev test fold lies inside the full run's test period. Adjusting anything after seeing dev test results adjusts it against the full run's test data.

### 16. The full run is not cut down to save time

`sparkov_v1_full` uses exactly the dev run's tuning procedure. If memory or runtime makes that infeasible, work stops and the constraint is reported before any part of the procedure changes. A reduced procedure is a deviation under decision 15.

### 17. On Sparkov, `trans_date_trans_time` is the only clock

*Added 2026-09-17, after the corpus was acquired and before any feature was built from it, on the evidence recorded under [The `unix_time` offset](#the-unix_time-offset).*

For Sparkov, `trans_date_trans_time` is the authoritative transaction timestamp. `unix_time` is not used for:

- feature extraction;
- chronological ordering;
- train, validation and test splitting;
- customer history windows;
- temporal drift analysis;
- any other temporal modelling.

`unix_time` is kept only as a quality and provenance diagnostic: the offset fields of the quality report, and the data-quality row the run card derives from them.

`unix_time` is not reconstructed or corrected with an inferred offset — seven years, a whole number of days, or a leap-day rule. The observed whole-day shift is reason enough not to treat it as a clock. A correction would rest on a guess about how the corpus was produced, and nothing in the benchmark needs one.

Two properties of the wall clock are therefore recorded, not corrected, since correcting either would mean reconstructing time from `unix_time`: the date 2019-02-28 holds rows from two `unix_time` dates, and 2020-02-29 holds no rows.

The adapter already derives every transaction timestamp from `trans_date_trans_time` alone, so this decision fixes existing behaviour as method rather than changing code. It settles the clock question that [`PHASE_5C_FEATURE_PARITY.md`](PHASE_5C_FEATURE_PARITY.md) left open as a gate on the first real Sparkov run.

---

## Frozen protocol

"Existing" refers to the code at the commit of this record.

**Data.** The existing Sparkov adapter: `trans_date_trans_time` as the authoritative clock, interpreted as UTC, with `unix_time` a diagnostic only (decision 17); the 14 → 12 category map; counted exclusions; manifest hash verification. Dev: 200 cards with whole histories, subsample seed 42. Final: the full corpus.

**Features.** Featureset `v1` from the batch builder — the production `FeatureExtractor.extract()` with the 180-day serving window — read through the fingerprinted feature cache.

**Split.** `ml/splits.chronological_split`: 70/15/15 by row count in canonical chronological order, checked by `assert_no_temporal_leakage`, with each fold's first and last timestamp written to `run.json`. Transactions that share a timestamp may fall on both sides of a boundary; that is recorded, not corrected.

**Tuning.** `ml/tuning.tune_hyperparameters` on the training fold only, with the existing search space, 25 iterations, 4 stratified folds and random state 42 (the existing `ml/train.py` defaults). The cross-validated PR-AUC it reports comes from folds shuffled within the training period; it is a diagnostic, not a held-out estimate.

**Fit.** The best parameters refit on the training fold, with `scale_pos_weight` computed from training labels and early stopping after 50 rounds against the val fold.

**Operating threshold.** The highest score threshold whose FPR on the val fold is at most 1%. If no threshold qualifies, the run artifacts record that explicitly and it is reported before the result is used.

**Test evaluation**, once per run, on the test fold:

- PR-AUC and ROC-AUC;
- recall at 1% and 5% FPR, labelled as points on the test ROC curve, since their thresholds come from the test fold;
- precision, recall, F1 and the confusion matrix at the val-selected operating threshold;
- the contextual fields of decision 13.

**Calibration.** Measured on the test fold with `ml/analysis/calibration.py`: Brier score, ECE, their positive-class variants, and the reliability curve. No calibrator is fitted.

**Explanation.** Global mean-absolute SHAP ranking on the test fold.

**Transfer.** Decision 8. The Sparkov test fold is reconstructed with the target run's split and checked to contain exactly the target run's test transaction ids.

**Drift.** Decision 3. Periods are half-open intervals on the wall clock interpreted as UTC: train `[2019-01-01, 2019-11-01)`, val `[2019-11-01, 2020-01-01)`, and one evaluation interval per calendar month of 2020. The threshold is selected once, on the val period, and never re-selected. Each month reports PR-AUC, recall, precision, realised FPR, fraud rate and volume; a month without positives or without negatives reports `null` for the metrics that are then undefined. The final drift result comes from the full corpus. Drift results never inform the main runs, because their evaluation months overlap the main test period.

**Rules audit.** Decisions 4 and 10. Each row receives the context the scoring service builds: that customer's transactions in the 180 days before it, newest first, excluding the row itself. Per rule: rows fired, fired on fraud, precision, fraud recall. The rules-only outcome uses the five evaluable rules — a hard-block rule gives DECLINE, otherwise a review rule gives REVIEW, otherwise APPROVE — and is compared with the labels.

**Promotion.** Decision 2.

---

## Unknown until execution

These are unknown, not undecided. Each is recorded from the data when it arrives and none is filled in from secondary sources.

| Unknown | Why it matters | Answered by |
|---|---|---|
| The complete `unix_time` offset distribution | See below | First acquisition |
| Actual file sizes and SHA-256 digests | Hash pinning; the manifest's sizes come from the Kaggle API, not from downloaded bytes | Acquisition |
| Row counts, fraud counts, date coverage, and where the two files meet | Currently known only from secondary sources | Quality reports |
| Rows excluded, by reason | A material exclusion rate needs investigating before training | Quality reports |
| Merchant names appearing under more than one canonical category | Tests the merchant-identity assumption | Quality reports |
| Whether exactly the five expected features are constant | Decision 6 | First feature build |
| Fold boundaries and fraud counts per fold, for dev and full | Whether the 200-card test fold holds enough positives for stable metrics | Dev and full runs |
| How fraud is distributed across cards and over time | Effective sample size, and fraud episodes that span a split boundary | Quality reports and folds |
| 2020 months with too few positives for defined drift metrics | Completeness of the drift series | Full-corpus drift run |
| Which of the five evaluable rules fire on Sparkov, and how precisely | The audit result itself | Rules audit |
| Memory and runtime of the full-corpus load, feature build and tuning | Decision 16 | Full run |
| Whether the `synthetic_v1` re-run reproduces the committed synthetic metrics | The synthetic path reads the local operational database, which is not versioned; a mismatch is reported before `synthetic_v1` is used as the transfer source | `synthetic_v1` re-run |

### Observed at acquisition (2026-09-17)

What acquisition could answer from the table above, computed from the retrieved bytes with the adapter's own reader, timestamp parser, exclusion gate and merchant identity. No feature matrix, fold or run exists yet; each run's quality report restates these from its own load.

- **Files.** Both sizes equal the manifest's advertised bytes. SHA-256 digests were computed, then pinned in the manifest as a separate, separately authorised step (decision 1).
- **Rows and frauds.** `fraudTrain.csv`: 1,296,675 rows, 7,506 frauds. `fraudTest.csv`: 555,719 rows, 2,145 frauds. Together: 1,852,394 rows, 9,651 frauds.
- **Coverage and where the files meet.** On the wall clock, train runs 2019-01-01 00:00:18 → 2020-06-21 12:13:37 and test 2020-06-21 12:14:25 → 2020-12-31 23:59:34. The files meet 48 seconds apart, with no overlap and no transaction number in both.
- **Rows excluded.** None: 0 under every exclusion reason.
- **Entity counts.** Observed across both files: 999 cards and 693 merchant names. The adapter identifies a merchant by (name, canonical category); 4 of the names appear under two canonical categories, so it builds 697 canonical merchants. Kaggle's description of 1,000 cards and 800 merchants is the publisher's advertised figure, not an observed count. Neither the data nor the adapter is adjusted to match it.
- **`unix_time` offsets.** Recorded under [The `unix_time` offset](#the-unix_time-offset); decided by decision 17.

### The `unix_time` offset

**Before acquisition (2026-09-16).** No interpretation of how `unix_time` relates to `trans_date_trans_time` was assumed in advance — neither that a non-uniform offset is benign nor that it is an error. On first acquisition the complete observed distribution of offsets was to be inspected and recorded, and that evidence would decide what the offset means.

The account then given in [`DATA_LICENSES.md`](../DATA_LICENSES.md), that the epoch carries the generating machine's timezone, was derived from the generator's source code and had not been checked against the published files. `trans_date_trans_time` remained the authoritative clock; anything in the observed distribution that called that into question was to be reported before any feature matrix was built from the real corpus.

**Observed on the published files (2026-09-17).** Inspected after acquisition and before any feature build, over all 1,852,394 rows of both files. Every row has a numeric `unix_time`, and the observed offsets fit well inside the 50-entry listing, so nothing is truncated.

| Offset (wall clock minus `unix_time`) | Rows | Wall-clock dates it covers |
|---|---|---|
| 220,924,800 s = 2,557 days | 928,083 | 2019-01-01 – 2019-02-27; on 2019-02-28, the 1,282 rows whose `unix_time` date is 2012-02-28; 2020-03-01 – 2020-12-31 |
| 220,838,400 s = 2,556 days | 924,311 | on 2019-02-28, the 1,859 rows whose `unix_time` date is 2012-02-29; 2019-03-01 – 2020-02-28 |

- Every offset is a whole number of days; none has an hours, minutes or seconds component. Read as UTC datetimes, the `unix_time` values run from 2012-01-01 00:00:18 to 2013-12-31 23:59:34, the same times of day as the wall-clock range 2019-01-01 00:00:18 to 2020-12-31 23:59:34.
- `fraudTrain.csv` holds both offsets (924,311 rows at 2,556 days, 372,364 at 2,557); `fraudTest.csv` holds only 2,557 days.
- Wall-clock 2019-02-28 holds 3,141 rows from two `unix_time` dates, 2012-02-28 and 2012-02-29, interleaved through the day. Wall-clock 2020-02-29 holds no rows: 2020-02-28 carries `unix_time` date 2013-02-28, and 2020-03-01 carries 2013-03-01. Every other date from 2019-01-01 to 2020-12-31 has rows.
- `fraudTrain.csv` is stored in `unix_time` order. Its wall clock steps backwards exactly once, by 86,226 seconds at row 100,532, where the offset changes on 2019-02-28. `fraudTest.csv` is in wall-clock order.
- `unix_time_offset_is_uniform` is therefore false on the real corpus.

**What the observation rules out.** A timezone offset is a matter of hours. The observed offsets are whole days, about seven years, and differ between rows, so the published files do not support the timezone account. That account is kept in [`DATA_LICENSES.md`](../DATA_LICENSES.md), labelled as the earlier interpretation, next to the observation. What produced the shift is not established here and does not need to be: decision 17 follows from the shift itself.

The quality report records the distribution under `extra.unix_time_offset_distribution`. A row's offset is its parsed `trans_date_trans_time`, in UTC epoch seconds, minus its `unix_time`; the field states that definition as `offset_definition`. It carries:

- `offsets` — each distinct offset with its row count, most frequent first, ties ordered by the smaller offset, so the same rows always serialise identically;
- `rows_compared` and `rows_without_unix_time` — rows that yield an offset, and rows whose `unix_time` is missing or non-numeric;
- `distinct_offsets`, `min_offset_seconds` and `max_offset_seconds`, which always cover every compared row;
- `listing_limit` (50), `truncated`, `unlisted_offsets` and `unlisted_rows` — at most 50 offsets are listed, so the report stays bounded, and a truncated listing states exactly what it left out.

The field records observations and makes no judgement. The existing `unix_time_mismatches`, `unix_time_modal_offset_seconds`, `unix_time_rows_at_modal_offset` and `unix_time_offset_is_uniform` fields are kept for compatibility. `unix_time_offset_is_uniform` is true exactly when `distinct_offsets` is 1: any second offset makes it false, even one second from the first, and so does having no comparable rows. It summarises the distribution; the first acquisition is still inspected from the distribution itself. If the listing is truncated on the real corpus, that is reported before any offset is interpreted.

### Observed at execution (2026-09-17)

What execution answered from the table above, after the runs, the transfer, the drift experiment and the rules audit had all been produced. Results themselves are not restated here: every value lives in the run records and is laid out in the generated [benchmark card](../../backend/ml/BENCHMARK_CARD.md), which is written from those records and never by hand.

- **Whether exactly the five expected features are constant.** Yes. Each Sparkov run records exactly the five features decision 6 expected as constant, in its training fold and over its whole matrix (`training_metadata.json`), and each of them has a mean absolute SHAP value of zero on the test fold (`feature_importance.json`). Twelve of the seventeen are live, counted from the matrices rather than asserted.
- **Fold boundaries and fraud counts per fold.** Recorded per run: fold periods in `run.json`, fold sizes in `training_metadata.json`, fold fraud counts in the `context` of `metrics.json`. On the full corpus the 70% row cut falls where the published files meet, so the train fold is exactly `fraudTrain.csv` and the val and test folds divide `fraudTest.csv`; no fold boundary lands on a shared instant.
- **Whether the 200-card test fold holds enough positives.** Its frauds sit on far fewer cards than the full run's do (`quality_report.json`, `cards_with_fraud` per fold), so the development result rests on correspondingly fewer independent fraud episodes. No threshold for "enough" was fixed before the runs, so this is reported rather than judged, and the development row is labelled as the development run wherever it appears.
- **How fraud is distributed across cards and over time.** Nearly every card in the corpus carries fraud, with a median of about ten frauds each, so a fold's fraud count rests on far fewer independent episodes than rows. A few cards carry fraud on both sides of a fold boundary. The monthly series in each `quality_report.json` shows volume roughly doubling each December while the fraud rate falls; December 2020 is the corpus's lowest-rate month on its highest volume.
- **2020 months with too few positives for defined drift metrics.** None. Every month of the drift series holds both frauds and legitimate rows, so no monthly metric is null and the series is complete.
- **Which of the five evaluable rules fire on Sparkov.** `velocity_burst`, `amount_ceiling` and `off_hours_high_value` fire; `geo_velocity_impossible` and `high_risk_country` are evaluable but cannot fire on a corpus where every row is US, and fired on nothing; `dormant_account_high_value` is reported as not evaluable, as decision 4 requires. Firings, precisions and fraud recalls are in `rules_audit.json`.
- **Memory and runtime of the full-corpus load, feature build and tuning.** Every step ran on one 16 GB machine without exhausting memory, so decision 16 was never triggered and no part of the procedure was reduced. Measured runtimes: the full run's analysis 357 s, the transfer 234 s, the drift experiment 2,413 s including its own hyperparameter search, and the rules audit 1,444 s. The quality-report, feature-build and training commands were run interactively and their runtimes were not recorded.
- **Whether the `synthetic_v1` re-run reproduces the committed synthetic metrics.** It does, exactly: the run's `metrics.json` and `threshold.json` match the served `backend/ml/artifacts/` records value for value, so the baseline row and the transfer source describe the same model the API serves.

### Execution record

- Every run was produced at commit `b48cfa6`, with the same library versions, and the full run used the development run's procedure unchanged (decision 16).
- No deviation arose under decision 15: nothing in the method changed between scoring the development test fold and scoring the full one.
- The drift experiment wrote to a directory of its own, named `sparkov_v1_drift` when it was run. Decision 14 named the three benchmark runs only, so this name is recorded here rather than frozen there.
- The optional development-run transfer and development-run rules audit were not run. Neither is required: decision 10 permits a development audit to validate tooling, and the transfer tooling was already proven on fixtures.
- Exploratory analyses made while reviewing the results — cluster bootstrap intervals over cards, a paired comparison of the two Sparkov models on the rows their test folds share, and feature-distribution comparisons — are not benchmark results, were designed after the test folds had been scored, and were not written into any artifact (decision 11).
- The run records, experiment results and the generated card are committed. Model files, plots, raw data, feature caches and the synthetic database are not, per [`DATA_LICENSES.md`](../DATA_LICENSES.md); the digests in each `run.json` and `training_metadata.json` remain the link to them.

---

## Consequences

**Good.** Every Sparkov number either comes from a procedure fixed before its test data was seen or carries a documented deviation. The three results cannot be read as one number. The drift experiment measures a future on which nothing was selected. The served model changes only by explicit decision.

**Cost.** The drift experiment runs its own hyperparameter search. The dormant-account rule yields no measurement on Sparkov, and the audit says so instead of producing one. Changing the method after the dev run is still possible but has to be documented, which is slower than just changing it, by design.

**Accepted limitations**, stated in the model card rather than engineered away in 5D:

- The same card appears in several folds with different transactions. This matches production, which scores cards it already knows; a held-out-cards evaluation stays future work.
- The val fold is used both for early stopping and for threshold selection, so the realised val FPR is slightly optimistic. The test fold is untouched by both.
- Label delay, per decision 13.

---

## Interpretation of the results

Every value referred to here is in the generated [benchmark card](../../backend/ml/BENCHMARK_CARD.md). This section says what the values mean, and says plainly where a cause is not established.

**The three results answer three questions.** The synthetic baseline measures how learnable this project's own generator is. The Sparkov runs measure how the same pipeline behaves on a corpus someone else generated. The transfer measures what the synthetic-trained model does on Sparkov rows it never saw. They are not a ranking of models, and decision 7 keeps them apart for that reason: each is read against its own test prevalence, and the development run is labelled as such wherever it appears.

**What the transfer number means.** Measured: the synthetic-trained model ranks Sparkov fraud better than chance but far below the model trained on Sparkov, and its threshold does not carry its false-positive rate across datasets — the realised rate on the Sparkov test fold is many times the ceiling it was selected under. Interpretation: a model and a threshold tuned around one generator do not transfer for free, which is the point of measuring it. Hypothesis, not established: the synthetic model's reliance on columns that are constant or out of range on Sparkov contributes to the drop. Transfer attributes nothing; it only measures.

**What the drift experiment shows.** Measured: at a threshold selected once on late 2019 and never re-selected, the realised false-positive rate stays near that ceiling through every month of 2020, recall eases downward over the later months, and PR-AUC and precision move with each month's fraud rate. Interpretation: an operations team watching a fixed threshold would have seen a stable false-positive budget and a slowly softening catch rate. Hypothesis, not established: most of the PR-AUC movement follows the monthly fraud rate rather than a decline in ranking quality, and December's behaviour is seasonal — the drift record carries no prevalence-independent metric, so neither can be separated from the artifact alone.

**Why the rules transfer only partially.** Measured: two rules are evaluable but cannot fire on a corpus where every row is US; the amount ceiling fires only on legitimate rows, because no Sparkov fraud reaches it; the off-hours rule fires with precision well above the corpus fraud rate while catching a small share of all fraud; the dormant-account rule is not evaluable at all. Interpretation: the rules are domain priors calibrated against this project's own generator and its injection patterns, so on another generator they measure how far those priors happen to agree. A rule that cannot fire is a property of the data, not evidence about the rule.

**The findings the plan asked to be stated plainly.**

- **F3, the history window.** Training once used unbounded history while serving used a 180-day window, so `days_since_last_tx` could differ between them past that window. The batch builder uses the serving window, which is what makes the Sparkov features the features production computes; the divergence is pinned by the parity tests rather than left to be discovered.
- **F4, leakage built in by construction.** The synthetic generator injects fraud preferentially into higher-risk customers and merchants, and its fraud rows carry the high-risk countries; the synthetic model's SHAP ranking leans on exactly those columns, and its country segments show a group that is entirely fraud. Part of the synthetic headline is therefore generator design rather than detectability. Sparkov has no risk tiers and one country, which is why those columns are constant there and why the Sparkov numbers are the more honest measurement.
- **F5, static account age.** `customer_account_age_days` is set when a customer is seeded and never updated, so it means "age at seeding". It is constant on Sparkov, and every Sparkov test row falls outside the range the synthetic model trained on. It is a dead feature on this corpus and a production wart left unchanged in featureset v1.

**What the benchmark does not establish.** Why the development and full Sparkov results differ: an exploratory comparison pointed at which cards fall in each test fold rather than at the models, but it was designed after the folds were scored and is not part of the benchmark. Nor does the benchmark establish performance on held-out cards, the cost of label delay, or anything about real card fraud: every dataset here is simulated.

---

## Deferred

| Deferred | Where |
|---|---|
| Promoting a Sparkov run to the served artifacts | Separate approval after benchmark review |
| Featureset version in `feature_list.json`, the explainer's version check, `/explain` and the model endpoint | M7 |
| ULB benchmark | M5 |
| Category segments beyond decision 9, featureset v2, time-series cross-validation for tuning | NICE-TO-HAVE, plan §27 |
| Held-out-cards evaluation, time-aware account age | FUTURE, plan §27 |
