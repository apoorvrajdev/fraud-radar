# Fraud Radar — Final Phase 5 Implementation Plan

**Status:** Implemented. Phases 5A–5G complete; as-built deviations are recorded in §18.1 and §20.1.
**Basis:** Full inspection of the repository at commit `128ffc9` (2026-05-28, "docs: add architecture diagrams and screenshot conventions (4D)"), the README, and the previous Phase 5 report. Every claim about the codebase below was verified by reading the source, not inferred from the README.
**Timebox:** MUST-HAVE scope sized for ~3 focused weeks; NICE-TO-HAVE fills week 4 only if MUST is done.
**Phase 5D / M4 methodology:** locked in [`adr/PHASE_5D_BENCHMARK_METHODOLOGY.md`](adr/PHASE_5D_BENCHMARK_METHODOLOGY.md), which governs wherever it differs from §4.3, §12, §14–§16, M4, §29 and the run names in M2's definition of done — the benchmark runs are `sparkov_v1_200cards` and `sparkov_v1_full`, each with its own quality report, per decision 14; the `sparkov_v1` of M2 is not used.
**Phase 5E / M5 methodology:** locked in [`adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md`](adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md), which governs wherever it differs from §7, §17, §20, §23, M5 and §29 item 5.

---

## 1. Executive Summary

Phase 5 turns Fraud Radar from "a fraud pipeline validated only against its own generator" into "a fraud pipeline validated against an independent external benchmark, with train/serve parity proven by tests." It adds an offline dataset track (Sparkov → adapter → canonical records → production feature extractor → training → evaluation → traceable artifact), a small real-data benchmark (ULB), a minimal FX enrichment in the live path, and the documentation/tests that make all of it defensible in an interview. It does **not** add streaming, services, a feature store, or any infrastructure.

Three code-level findings reshape the previous plan:

1. **No new feature extractor is needed.** `FeatureExtractor.extract()` already has an in-memory execution path (`recent_transactions=`) that the scoring service uses in production, with an existing parity test against the SQL path. The offline batch builder simply drives that same path per customer with a rolling 180-day window. The "vectorized extractor + golden parity test" from the prior report is cut; the parity test that remains is cheaper and stronger because it compares the batch builder to the *actual* production semantics.
2. **The offline track never touches the operational database or the HTTP endpoint.** `TransactionCreate` has no timestamp field (`created_at` is server-stamped), so historical replay through the API was never viable. The adapter yields transient ORM objects in memory; features are built without a session; only artifacts are written. The DB replay loader from the prior report is cut. Offline and live paths are cleanly separated, as required.
3. **Two latent train/serve issues already exist and Phase 5 fixes them as a by-product.** (a) Training extracts `days_since_last_tx` over unbounded history (SQL path) while serving uses a 180-day window (in-memory path) — on the 30-day synthetic dataset this never bites, on Sparkov's 24 months it would. (b) The synthetic generator injects fraud preferentially into MEDIUM/HIGH-risk customers and HIGH-risk merchants, so `customer_risk_tier_encoded` and `merchant_risk_encoded` are partially label-leaky by construction — one reason the 0.93 PR-AUC is flattering. Sparkov has no risk tiers, which is exactly why it is the honest benchmark.

**Recommended order:** deploy the existing demo mode to Vercel now (4F-lite, hours, not days), start Phase 5, and record the Loom (4E) once at the end so it shows the Phase 5 result. Details in §28.

**Framing that must hold everywhere (README, ADR, model cards, interviews):**
- Our simulator = our own synthetic live traffic.
- Sparkov = an independent, externally generated **synthetic** card-transaction benchmark, valuable because it has entity-level structure we did not design.
- ULB = a real, anonymized historical fraud benchmark, used on its own track because its PCA features cannot map to ours.
- Frankfurter = real external reference data used for enrichment/analytics, not for fraud labels.
- Fraud Radar never receives real bank or card transactions. Legitimate public APIs do not provide open access to identifiable card-transaction streams; payment transaction data is generally subject to privacy, contractual, security, and regulatory restrictions.

---

## 2. Why Phase 5 Exists

The current system is a closed loop: the dataset generator, the six rules, the fraud-injection patterns, and the model were all designed by the same author with knowledge of each other. The rules are documented as "calibrated to Phase 2E injection patterns," and the feature extractor's `is_off_hours` window (2–5am) is literally the generator's off-hours pattern. High metrics in a closed loop are expected and prove little.

Phase 5 opens the loop in the narrowest useful way: hold the pipeline fixed, feed it a dataset it was not designed around, measure honestly, and make the measurement reproducible. It also converts several implicit engineering properties (feature ordering, artifact provenance, history-window semantics) into explicit, tested contracts. That is the material a Barclays SWE interviewer will actually engage with: not "PR-AUC 0.93" but "here is how I found and closed a train/serve skew, here is why I subsample entities instead of rows, here is why the rules only transfer partially and what I did about it."

---

## 3. Current Architecture Assessment (verified in code)

Strengths that Phase 5 relies on and must not weaken:

| Strength | Where | Why it matters for Phase 5 |
|---|---|---|
| Single feature semantics for train and serve | `app/fraud/features.py`; `ml/data.py` calls the production extractor | The batch builder reuses `extract()` unchanged |
| Two execution paths already parity-tested | `extract(recent_transactions=...)` + `tests/unit/test_features_parity.py` | The third caller (batch builder) plugs into the same harness |
| Positional feature contract enforced at load | `feature_spec.FEATURE_NAMES`; `FraudExplainer.__init__` validates `feature_list.json` order | Registry is an extension of an existing check, not a new system |
| Committed artifact metadata | `ml/artifacts/{feature_list,threshold,metrics,training_metadata}.json` (model.json gitignored) | Traceability = add fields, not a registry |
| Chronological 70/15/15 split + leakage assertion | `ml/splits.py` | Reused verbatim on Sparkov and ULB |
| Fraud-appropriate evaluation | `ml/evaluation.py`: PR-AUC, `find_threshold_at_fpr`, `recall_at_fpr`, confusion at threshold; `ml/analysis/` calibration + segments | Reused verbatim |
| Rules as pure functions over a frozen `TransactionContext` | `app/fraud/rules.py`, `transaction_context.py` | Rules audit on Sparkov = construct contexts, call `evaluate_all` |
| Conservative-wins decision matrix, audit log with model version in actor | `app/services/scoring.py` | Badge and provenance already have a home |
| Idempotent ingestion, keyset pagination, analyst loop | untouched in Phase 5 | Advertised as data-source-agnostic |
| ADR discipline | `docs/adr/PHASE_*` | Phase 5 opens with `PHASE_5A_DESIGN.md` |

Weaknesses Phase 5 addresses: closed-loop validation; latent 180-day window skew; construction-time label leakage via risk tiers; static `account_age_days`; single-currency assumptions in stats and frontend; no CI; artifact metadata lacks dataset/featureset/period provenance.

---

## 4. Actual Repository Findings

### 4.1 Differences between the previous report / README and the code

| # | Previous assumption or README claim | What the code actually does | Consequence for Phase 5 |
|---|---|---|---|
| F1 | Geo features use distances; Sparkov lat/long "maps with identical semantics" | The two geo features are `country_mismatch_customer` and `country_mismatch_merchant` — ISO country string comparisons. No coordinates exist anywhere in the schema. | Sparkov's lat/long is **unused** by featureset v1. Both geo features are constant (0) on Sparkov (all US). Distance features would need new columns → deferred to FUTURE. |
| F2 | A new vectorized extractor + golden parity test is required | `extract()` already supports an in-memory path (`recent_transactions`), used by `scoring.py` in production and covered by 4 parity tests | **Cut** the vectorized extractor. Batch builder = production in-memory path driven per customer. |
| F3 | Training and serving share one code path, so no skew | Training (`ml/data.py`) uses the SQL path with **unbounded** history; serving (`scoring._load_context`) passes a **180-day** window. `days_since_last_tx` returns the true gap in training but 999 at serving when gap > 180d. Latent on 30-day synthetic data. | Batch builder uses the 180-day window (production semantics). Training on Sparkov therefore matches serving; the skew is documented as found-and-fixed. |
| F4 | Risk tiers are neutral customer/merchant attributes | `_inject_account_takeover` selects only MEDIUM/HIGH customers; `_inject_merchant_concentration` selects only HIGH-risk merchants. Two of 17 features carry label information by construction. | Report it. Sparkov has no tiers (customer tier constant; merchant tier derived from category via the same table the generator uses). |
| F5 | `account_age_days` is a time-aware feature | It is a static column set at seeding; never updated per transaction. Semantics: "age at seed time." | Constant on Sparkov (documented dead feature). Noted as a production semantic wart; not fixed in Phase 5 (would change v1). |
| F6 | Bulk replay of Sparkov into the DB | `TransactionCreate` has no timestamp; `created_at` is `server_default`. DB load is possible via ORM but pollutes live stats/alerts and buys nothing for training. | Offline track is **DB-free**. Sparkov never enters the operational database. |
| F7 | Model artifacts are not in the repo | `feature_list.json`, `threshold.json`, `metrics.json`, `training_metadata.json`, calibration/segment/importance JSONs are committed; only `model.json`/joblib/PNGs are gitignored | Fresh clone must run `generate_dataset` + `train` to serve; Phase 5 adds a `ml.promote` step and per-run directories. |
| F8 | Model version is an unspecified concept | `_MODEL_VERSION = training_metadata.trained_at_utc`; audit actor = `scorer:<ts>` | Extend metadata with `dataset`, `featureset_version`; expose via a small `/api/v1/model` endpoint for the badge. |
| F9 | 197 tests (README) | 220 test functions on disk; **no CI** (no `.github/`) | Add GitHub Actions in 5A; fix README count. |
| F10 | Simulator is config-driven | Amounts `uniform(5, 500)`, currency `USD`, country `US`, merchant chosen independent of category, 3 hard-coded patterns; hour-of-day is wall-clock (server-stamped) | Calibration = profile-driven amount/category/card-present/currency sampling. Time-of-day **cannot** be calibrated in a live simulator — dropped from 5G. |
| F11 | Multi-currency is a display concern | `stats.py` sums raw `amount`; frontend `format.ts` hard-codes USD | Non-USD traffic would corrupt volume KPIs → FX needs a stored base-currency amount (one additive migration). |
| F12 | — | Hyperparameter search uses shuffled `StratifiedKFold` inside the train fold | Acceptable; note it. `TimeSeriesSplit` is a NICE item, not a fix. |
| F13 | — | Alembic runs with `render_as_batch=True` | Additive SQLite migrations are safe. |
| F14 | — | `HIGH_RISK_CATEGORIES` = JEWELRY, MONEY_TRANSFER, GAMBLING, CRYPTO, ELECTRONICS; merchant taxonomy = 12 categories with fixed risk ratings | Sparkov's 14 categories map onto this taxonomy; `is_card_present` derives from Sparkov's `_net`/`_pos` suffix. |
| F15 | — | Dev machine is Windows (per `training_metadata.library_versions.platform`) | No symlinks for "active model"; use a copy-based `promote` command. |

### 4.2 Sparkov-specific facts that drive the adapter (verify against the downloaded files in 5B)

`fraudTrain.csv` ≈ 1,296,675 rows (2019-01-01 → 2020-06-21), `fraudTest.csv` ≈ 555,719 rows (2020-06-21 → 2020-12-31); ~983 cards, ~693 merchants, 14 categories; fraud ≈ 0.58% (train) / 0.39% (test). Columns: `trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud`. Timestamps are naive; Phase 5 interprets them as UTC (only relative time-of-day structure matters).

### 4.3 Featureset v1 availability on Sparkov (honest accounting)

| Feature | On Sparkov | Note |
|---|---|---|
| log_amount, hour_of_day, is_weekend, is_off_hours | live | direct |
| is_card_present | live | `category` suffix `_pos` → True, `_net` → False |
| country_mismatch_customer / merchant | **dead (constant 0)** | all US; no country-level signal exists in Sparkov |
| tx_count_1h, tx_count_24h, log_amount_sum_24h, avg_amount_30d, amount_zscore_30d, days_since_last_tx | live | per `cc_num` history, 180-day window |
| customer_account_age_days | **dead (constant)** | not derivable without changing v1 semantics |
| customer_risk_tier_encoded | **dead (constant LOW)** | Sparkov has no tiers; deriving one from labels would be leakage |
| merchant_risk_encoded, is_high_risk_category | live | via category → taxonomy → risk table (same as generator; label-free) |

13 of 17 live. This is reported, not hidden, and motivates featureset v2 (NICE) with features derivable from fields Sparkov and our schema both have.

---

## 5. Phase 5 Goals

1. Ingest an external dataset through a clean adapter into the canonical transaction representation, with a validation/quality report and explicit provenance.
2. Train and evaluate the existing pipeline on that dataset using the production feature semantics, chronological splitting, fraud-specific metrics, and a temporal generalization experiment.
3. Make feature versions and model artifacts explicit and traceable (dataset, source hash, featureset, periods, threshold, metrics).
4. Prove batch/production feature parity and past-only construction of every history feature with tests.
5. Audit how the deterministic rules transfer to the external dataset.
6. Run a small, separate real-data benchmark (ULB) with the same evaluation discipline.
7. Add a small, failure-tolerant FX enrichment to the live path for multi-currency analytics.
8. Calibrate the live simulator's amount/category mix from the external dataset (NICE).
9. Surface model/dataset/featureset provenance in the dashboard; refresh docs, README, demo.
10. Add CI.

## 6. Phase 5 Non-Goals / Explicitly Out of Scope

Kafka/Redpanda/streaming, Kubernetes, microservices, cloud data lake, Spark/Flink/Airflow, MLOps platform, feature store, model registry service, online learning, deep learning, model-serving infra, real bank/payment integrations, scraping, paid data providers, production payment processing, RBAC, 3DS, WebSockets, a second dashboard, non-additive migrations, performance engineering beyond "don't do obviously dumb things," SMOTE, market-data APIs (Alpha Vantage/Finnhub/FMP) in any form, IEEE-CIS (future research track only), FRED (deferred; omitted unless it materially helps the story — current judgment: omit), coordinate-based geo features (needs schema change), loading Sparkov into the operational DB.

---

## 7. Final Architecture

```
OFFLINE — dataset & model development (no DB, no HTTP)
════════════════════════════════════════════════════════════════════
 Kaggle CSVs (gitignored)          ULB CSV (gitignored)
   │ scripts/download_data.py        │
   │ + SHA-256 manifest              │
   ▼                                 │
 ml/datasets/sparkov.py  ──adapter──►│ (no adapter: PCA space,
   │  column mapping, ID hashing,    │  own thin loader)
   │  category taxonomy mapping,     │
   │  entity subsampling             │
   ▼                                 │
 CanonicalDataset (in memory)        │
   transient ORM Customer /          │
   Merchant / Transaction objects    │
   + labels + provenance             │
   │                                 │
   ├─► ml/datasets/quality.py ──► quality_report.json
   │                                 │
   ▼                                 ▼
 ml/features/batch.py            ml/tracks/ulb/
   per-customer chronological      X = V1..V28 + log1p(Amount)
   rolling 180-day window →        chronological split on Time
   FeatureExtractor.extract(       tune / train / threshold / eval
     recent_transactions=...)      (reuses ml.splits / ml.evaluation
   [= production path] → cache       / ml.tuning / ml.analysis)
   │
   ▼
 LabelledDataset ──► ml.splits (70/15/15 chronological)
   │
   ├─► ml/train.py --dataset sparkov --featureset v1
   │      RandomizedSearchCV → final fit → threshold @ FPR ≤ 1% on val
   │      → ml/artifacts/runs/<run_name>/ {model.json, feature_list.json,
   │         threshold.json, metrics.json, training_metadata.json}
   ├─► ml/analyze.py → segment / calibration / global SHAP / model card
   ├─► ml/experiments/temporal_drift.py → drift_metrics.json + PNG
   ├─► ml/experiments/rules_audit.py    → rules_audit.json
   └─► ml/promote.py <run_name> ──copy──► ml/artifacts/  (what the API loads)

LIVE — demo path (unchanged core + one enrichment)
════════════════════════════════════════════════════════════════════
 app/simulator (USD default; --profile sparkov, --currency-mix optional)
   │ HTTP POST /api/v1/transactions  (Idempotency-Key)
   ▼
 FastAPI router → transactions service
   │
   ├─► app/enrichment/fx.py  (only if currency != USD)
   │     cache table → Frankfurter (2s timeout) → stale fallback → unavailable
   │     writes amount_base / fx_rate / fx_rate_date / fx_source; never blocks scoring
   ▼
 scoring service: rules (TransactionContext, 180d) → FeatureExtractor (in-memory path)
   → XGBoost → SHAP → conservative-wins decision → audit log (actor scorer:<version>)
   ▼
 SQLite  ──►  stats (volume in base currency)  ──►  React dashboard
                                                     + model/dataset/featureset badge
 analyst review loop (unchanged)
```

The only place the two worlds touch is the artifact directory: offline produces it, live loads it. That boundary is the interview line: "training never needs the API, the API never needs the training data."

---

## 8. Data Sources

**Sparkov — primary external synthetic benchmark.** Kaggle `kartik2112/fraud-detection`, generated with Brandon Harris's Sparkov tool. Independent generator, entity-level (`cc_num`, merchant, category, amount, timestamps), labelled. Used for training/evaluation/rules audit/simulator calibration. License listed CC0 on the dataset card — confirm at download time and record in `docs/DATA_LICENSES.md`. Data never committed.

**ULB Credit Card Fraud — real, anonymized historical benchmark.** Kaggle `mlg-ulb/creditcardfraud`: 284,807 European card transactions over two days in Sept 2013, 492 frauds (0.172%), features `Time, Amount, V1–V28` (PCA). Used on its own track. Attribution to the ULB Machine Learning Group / Worldline and the Dal Pozzolo et al. reference in the model card. Data never committed.

**Frankfurter — FX reference rates.** Open API over ECB daily reference rates (~30 currencies, history to 1999, no key, self-hostable). Used only to compute a base-currency display/analytics amount at ingestion, cached locally. Not a model input in MUST-HAVE.

**FRED — omitted.** Macro series do not change any fraud decision, and a context tile would visually suggest a relationship that does not exist. If ever added: dashboard-only, labelled "context," after Phase 5.

**Market-data APIs — excluded.** Alpha Vantage (25 req/day free), Finnhub (60/min, personal/non-commercial), FMP (250/day, personal use) serve prices, fundamentals and news. None carries card transactions or fraud labels; none can train, evaluate, enrich, or score a card-fraud model. Adding them would dilute the fraud story and signal a misunderstanding of the domain.

**IEEE-CIS — future research track only.** Richest real CNP dataset, but masked semantics, relative timestamps, and competition-rules licensing (non-commercial, no redistribution). Mentioned in Future Work; not part of Phase 5.

---

## 9. Dataset Adapter Architecture

**Location:** `backend/ml/datasets/`. Pure Python + pandas; no SQLAlchemy session, no FastAPI imports.

```python
# ml/datasets/base.py
@dataclass(frozen=True)
class DatasetProvenance:
    name: str                # "sparkov"
    source_url: str
    citation: str
    license: str             # "CC0 (per Kaggle dataset card, retrieved <date>)"
    files: dict[str, str]    # filename -> sha256
    row_count: int
    period_start: datetime
    period_end: datetime
    subsample: dict | None   # e.g. {"strategy": "cards", "max_cards": 200, "seed": 42}

@dataclass(frozen=True)
class CanonicalDataset:
    customers: dict[str, Customer]        # transient ORM instances, keyed by id
    merchants: dict[str, Merchant]
    transactions: list[Transaction]       # sorted by created_at ASC (guaranteed)
    labels: dict[str, int]                # transaction.id -> 0/1
    provenance: DatasetProvenance

class DatasetAdapter(Protocol):
    name: str
    def load(self, root: Path, *, max_cards: int | None, seed: int) -> CanonicalDataset: ...
```

**Why transient ORM instances instead of a new dataclass schema:** the production extractor and rules take `Customer`/`Merchant`/`Transaction` objects. Reusing those types *is* the canonical schema decision — one representation, zero mapping code between "offline canonical" and "what the extractor expects." The objects are never added to a session.

**Sparkov mapping (`ml/datasets/sparkov.py`):**

| Sparkov | Canonical | Policy |
|---|---|---|
| `cc_num` | `Customer.id`, `Transaction.customer_id` | `uuid5(NAMESPACE, str(cc_num))` — 36-char id; raw PAN-like value never stored or logged |
| `first`,`last`,`gender`,`dob`,`street`,`city`,`zip`,`job`,`city_pop`,`lat`,`long` | not mapped | no canonical field; recorded as "dropped" in the quality report |
| `state` | `Customer.country = "US"` | derived; flagged `derived` |
| — | `Customer.risk_tier = "LOW"`, `account_age_days = 0` | constants; flagged `unavailable` |
| `merchant` | `Merchant.id = uuid5(NAMESPACE, merchant)`, `name` | one merchant row per distinct name |
| `category` | `Merchant.category`, `mcc`, `risk_rating` | 14 → 12 taxonomy map (below); `risk_rating` from `CATEGORIES[...]["risk"]` (label-free) |
| `merch_lat`,`merch_long` | not mapped | `Merchant.country = "US"` |
| `amt` | `Transaction.amount` (Decimal, 2dp) | reject ≤ 0 (quality report counts) |
| `trans_date_trans_time` | `Transaction.created_at` (UTC-aware) | naive → UTC; `unix_time` used as cross-check |
| `category` suffix | `Transaction.is_card_present` | `_pos` → True, `_net` → False, else per map default |
| — | `currency="USD"`, `payment_method="CARD"`, `country="US"`, `status="APPROVED"` | constants |
| `trans_num` | `Transaction.id` (uuid5 of `trans_num`), `idempotency_key = trans_num` | deterministic |
| `is_fraud` | `labels[id]` | never placed on the Transaction object |

Category map: `grocery_pos/grocery_net→GROCERY`, `food_dining→RESTAURANT`, `shopping_pos/shopping_net/misc_pos/misc_net/home/kids_pets/personal_care→RETAIL`, `gas_transport→GAS_STATION`, `entertainment→ENTERTAINMENT`, `travel→TRAVEL`, `health_fitness→ONLINE_SERVICE` (closest MEDIUM-risk bucket; document as judgment). A test asserts every Sparkov category is mapped.

**Entity subsampling, not row subsampling.** `--max-cards N` selects N `cc_num` values with a seeded RNG and keeps *all* their transactions. Random row sampling would silently corrupt every velocity feature by deleting history. Default for development: `--max-cards 200` (≈370k rows); full corpus is an explicit flag. Fraud rate of the sample is reported and must be within a tolerance of the full-corpus rate (test).

**Quality report (`ml/datasets/quality.py`):** row counts per file, date range, duplicates on `trans_num`, nulls per column, amount distribution (p1/p50/p99/max), transactions per card (p50/p99), fraud rate overall and per category, dropped/derived/unavailable field inventory, and the subsampling record. Written to `ml/artifacts/runs/<run>/quality_report.json` and summarized in the model card.

---

## 10. Canonical Transaction Schema

Unchanged: `Transaction`, `Customer`, `Merchant` ORM models are the canonical schema for both worlds. Phase 5 adds only the FX columns (§18) — the single migration in Phase 5, additive and nullable:

- `transactions.amount_base NUMERIC(19,4) NULL`, `fx_rate NUMERIC(18,8) NULL`, `fx_rate_date DATE NULL`, `fx_source VARCHAR(16) NULL`
- new table `fx_rates(base CHAR(3), quote CHAR(3), rate_date DATE, rate NUMERIC(18,8), fetched_at TIMESTAMP, PRIMARY KEY(base, quote, rate_date))`

The ADR states the invariant explicitly: *"The canonical schema is the ORM model. External datasets are adapted into it; nothing is adapted out of it."*

---

## 11. Feature Registry / Feature Versions

Minimal extension of `app/fraud/feature_spec.py`:

```python
FEATURESETS: dict[str, list[str]] = {
    "v1": [ ...the existing 17 names, unchanged order... ],
}
DEFAULT_FEATURESET = "v1"
def feature_names(version: str) -> list[str]: ...
FEATURE_NAMES = FEATURESETS[DEFAULT_FEATURESET]   # backward-compatible alias
```

- `FeatureExtractor.extract(..., featureset="v1")` — v1 is the only implementation in MUST; the parameter exists so v2 is additive later.
- `feature_list.json` gains `"featureset_version": "v1"`; `FraudExplainer` validates names **and** version; `/explain` and the new `/api/v1/model` endpoint echo the version.
- A unit test pins the v1 list to a literal copy so accidental reordering fails CI.

**Featureset v2 (NICE-TO-HAVE, only if MUST is done):** three history features derivable from fields both Sparkov and our schema already have — `tx_count_7d`, `distinct_merchants_24h`, `amount_ratio_to_30d_avg` — implemented in `extract()` behind the version switch, evaluated v1-vs-v2 on Sparkov with the same split/threshold protocol, and adopted only if PR-AUC and Recall@1%FPR both improve. This is the concrete "feature versioning" demonstration; without an actual v2 the registry is hollow, which is why v2 is the first NICE item.

---

## 12. Leakage Prevention

Every history feature is documented in `docs/adr/PHASE_5A_DESIGN.md` with its construction rule. Verified predicates in code:

| Feature | Construction (verified) | Leakage guard |
|---|---|---|
| tx_count_1h / 24h, log_amount_sum_24h, avg_amount_30d, amount_zscore_30d | `cutoff <= t.created_at < tx.created_at` over the same customer's prior transactions | strict `<` excludes the current row and any same-timestamp later rows; batch builder feeds only rows with `created_at < tx.created_at` |
| days_since_last_tx | max prior `created_at < tx.created_at`; 999 if none in window | window fixed at 180d in both training (new) and serving |
| hour_of_day, is_weekend, is_off_hours, log_amount, is_card_present | transaction-local | none needed |
| country mismatches | vs static customer/merchant country | none needed |
| account_age_days, risk tiers | static entity attributes | **construction-time label leakage in synthetic data (F4)**; constant on Sparkov; reported in model card |

Additional guards:
- **Split boundaries:** chronological only (`ml/splits.py`), `assert_no_temporal_leakage` retained; train/val/test period bounds written into `training_metadata.json`.
- **Entity contamination:** the same card appears across folds with different transactions. This is deliberate and correct for a scoring system (production scores known cards). Stated in the model card. A held-out-cards variant is not built.
- **Target-derived fields:** `labels` live outside the `Transaction` object; adapter test asserts no attribute of any canonical object equals or is derived from `is_fraud`. Merchant risk is derived from category, never from fraud rate.
- **Hyperparameter search:** on the train fold only (existing); threshold chosen on val only (existing).
- **Future-information test (new):** for a fixture customer, compute features for row *i* with the full sorted history, then again with all rows after *i* deleted; vectors must be identical. Run for several *i* including the last row.
- **Subsampling:** by entity (§9) so no history is deleted.
- **Sparkov drift experiment:** train period strictly earlier than every evaluation month; hyperparameters reused from the main run, not re-tuned on 2020.

---

## 13. Batch vs Production Feature Parity

`ml/features/batch.py` — `build_feature_matrix(ds: CanonicalDataset, featureset: str) -> LabelledDataset`:

1. Group transactions by `customer_id`; each group is already chronological.
2. Maintain a deque of that customer's prior transactions; before extracting row *t*, pop entries with `created_at < t.created_at - 180d` (exact `scoring._RECENT_HISTORY_WINDOW`).
3. Call `FeatureExtractor().extract(db=None-safe stub, tx, customer=..., merchant=..., recent_transactions=list(deque), featureset=...)`. Because `customer`, `merchant`, and `recent_transactions` are all supplied, `extract()` never touches the session; a `_NoDatabase` sentinel that raises on any attribute access guarantees it (test).
4. Append `values`, label, timestamp, id. Cache the matrix as `.npz` keyed by `(dataset file sha256, subsample record, featureset_version)` so re-runs are seconds.

**Parity test (golden):** a fixture of ~40 transactions across 3 customers with dense and sparse histories, including gaps > 180 days, is (a) inserted into the test SQLite session and featurized via the **SQL path with a 180-day-bounded `recent_transactions=None` equivalent**, and (b) featurized via `build_feature_matrix` from transient objects. Vectors must match exactly (float equality, as the existing parity test enforces). A second assertion documents the F3 finding: the unbounded SQL path differs from the 180-day path *only* in `days_since_last_tx` for the > 180-day-gap case.

**Cost:** pure-Python per-row extraction over an in-memory window; ~5–10 minutes for 200 cards, tens of minutes for the full corpus — acceptable, cached, and honest ("I chose exact production semantics over a 50× faster pandas rewrite that would need its own parity proof"). Training on the full corpus is a flag, not the default.

---

## 14. ML Training + Evaluation

`ml/train.py` gains `--dataset {synthetic,sparkov} --featureset v1 --max-cards N --run-name <name>`; `--dataset synthetic` keeps today's behaviour (DB + CSV labels). Output goes to `ml/artifacts/runs/<run-name>/`; `python -m ml.promote <run-name>` copies the five artifact files into `ml/artifacts/` (what the API loads). Copy, not symlink (F15).

`training_metadata.json` gains: `dataset_name`, `dataset_source_url`, `dataset_files_sha256`, `dataset_row_count`, `subsample`, `featureset_version`, `train_period`, `val_period`, `test_period`, `seed`, `git_commit` (best-effort), `run_name`. `metrics.json` and `threshold.json` unchanged in shape.

**Metrics (all existing code):** PR-AUC, ROC-AUC, Recall@1% FPR, Recall@5% FPR, precision/recall/F1 and confusion at the operating threshold, threshold chosen on val at target FPR ≤ 1%, calibration (existing `ml/analysis/calibration.py`), segments where cheap (category, card-present, amount bucket).

**Why these and not accuracy — stated in the model card:** at 0.5% prevalence a constant "legit" classifier is 99.5% accurate. What matters operationally is (1) the precision–recall trade-off at the fraud base rate, (2) the false-positive budget — every FP is a declined legitimate customer and an analyst minute — so we fix FPR and read recall, (3) the operating threshold is a business decision made on validation data and recorded with its realised FPR, (4) calibration, because REVIEW/DECLINE bands are score-based and an uncalibrated score makes the bands meaningless, (5) drift, because fraud is adversarial and non-stationary, (6) segment behaviour, because a model that is great on average and blind on card-not-present is not deployable.

**Expected and pre-announced:** Sparkov numbers will be lower than 0.9327 PR-AUC. The model card leads with the comparison table and explains the drop (13/17 live features, no risk-tier leakage, independent generator).

**Cross-generator transfer (cheap, included):** score the Sparkov test fold with the *synthetic-trained* v1 model (same feature space). Expect poor PR-AUC; report it as the "why you cannot validate on your own generator" number.

## 15. Temporal Drift / Generalization Experiment

`ml/experiments/temporal_drift.py --dataset sparkov`:
- Train fold: 2019-01 → 2019-10; val: 2019-11 → 2019-12 (threshold @ FPR ≤ 1%); hyperparameters copied from the main run.
- Evaluate each month of 2020 at the **fixed** threshold: PR-AUC, recall, precision, realised FPR, fraud rate, volume.
- Output `drift_metrics.json` + one PNG (PR-AUC and realised FPR by month). Model card gets a paragraph: which quantity drifted, whether FPR or recall moved, what the retraining trigger would be.

Realised FPR at a fixed threshold is the headline series — it is what an operations team actually watches, and it moves even when PR-AUC looks stable. Budget: half a day once features are cached.

## 16. Rules + ML + SHAP Integration

No changes to the rules, decision matrix, explainer, or scoring service in MUST-HAVE. Two additions:

- `ml/experiments/rules_audit.py`: builds a `TransactionContext` per Sparkov row (same 180-day window, sorted desc as `_load_context` does) and calls `evaluate_all`. Reports per rule: fired count, fired-on-fraud, precision, fraud recall, and the would-be decision-matrix outcome with rules alone. Expected: `velocity_burst` and `off_hours_high_value` live; `geo_velocity_impossible`, `high_risk_country`, `dormant_account_high_value` structurally dead on Sparkov (no country variation; static account age). This is written up as "rules are domain priors that do not transfer automatically — and here is the measurement," which is a stronger interview answer than pretending they generalize.
- `/explain` and `/api/v1/model` echo `featureset_version` and `dataset_name` so an analyst can see *which* model explained *which* decision (SHAP contributions already stored per transaction at scoring time — that part is untouched).

NICE: parameterize rule thresholds via settings so the audit can test a Sparkov-tuned variant. FUTURE: rule versioning.

## 17. ULB Benchmark Track

`backend/ml/tracks/ulb/` — deliberately small (≤ 1.5 days):
- `load.py`: read `creditcard.csv`; `X = [V1..V28, log1p(Amount)]`, `y = Class`, timestamps = `Time` (seconds); no `Time` as a feature.
- `train.py`: `chronological_split` on `Time`, `tune_hyperparameters` (fewer iterations), `_final_fit` with `scale_pos_weight`, threshold at FPR ≤ 1% on val, `metrics.json`/`threshold.json`/`training_metadata.json` written to `ml/artifacts/runs/ulb_baseline/`; calibration via `ml/analysis/calibration.py`; a one-page `MODEL_CARD_ULB.md`.
- Never promoted to the API (feature space differs; a guard in `ml.promote` refuses artifacts whose `featureset_version` is not a registered production featureset).

Why ULB gets its own track (stated in the doc): the V-features are PCA components of undisclosed inputs; entity identity, merchants and geography are unrecoverable, so any "velocity" feature built on them would be fiction. The value is real-data pathology — 0.17% prevalence, 492 positives, high metric variance (report seed variance across 3 seeds) — and proving the evaluation discipline is dataset-agnostic.

## 18. FX / Multi-Currency Design (minimal)

- `app/enrichment/fx.py`: `FxService.get_rate(quote: str, on: date) -> FxLookup(rate, rate_date, source)` with resolution order: (1) `fx_rates` cache for the latest `rate_date <= on` within 7 days (ECB publishes business days only; weekend/holiday dates resolve to the prior business day — same rule offline and live); (2) Frankfurter `GET /v1/{date}?base=USD&symbols=<quote>` with a 2-second timeout, upsert into cache; (3) stale fallback: the most recent cached rate for that pair regardless of age, `source="stale"`; (4) `source="unavailable"`, rate `None`. Never raises into the scoring path.
- Ingestion (`services/transactions.py`): after validation, if `currency == "USD"` set `amount_base = amount`, `fx_rate = 1`, `fx_source = "identity"`, no call. Otherwise call `FxService`, store `amount_base`, `fx_rate`, `fx_rate_date`, `fx_source`. The scoring audit payload includes `fx_source` when not identity, so a decision made with a stale or missing rate is visible in the audit trail.
- Stats: volume KPIs and amount filters use `COALESCE(amount_base, amount)`. Frontend shows the original amount with its currency and, when different, the base amount.
- Rules and model continue to use the raw `amount` in MUST-HAVE. Documented limitation: the simulator's non-USD share is optional and amounts are of comparable magnitude, so skew is bounded; whether the model should see `amount_base` is a NICE experiment decided by evaluation, not assumed.
- Historical consistency: offline datasets in Phase 5 are USD-only (`amount_base = amount`, rate 1). The batch builder exposes the same `FxService.get_rate(quote, on=created_at.date())` hook using the *same cache table* so a future non-USD dataset joins historical daily rates with identical semantics to the live path.
- Failure modes are tested (§21). No retries, no circuit breaker, no background refresh — the cache plus stale fallback is the whole resilience story, and it is enough.

### 18.1 As built (5F)

Implemented as specified above, with the contract written up in [`FX_CONTRACT.md`](FX_CONTRACT.md) rather than a further ADR: a methodology record exists to freeze a measurement before the data is seen, and nothing in FX has that property. Four deviations, each forced by something the plan could not know when it was written:

1. **Frankfurter v2, not v1.** The live API is `GET /v2/rates?base=…&quotes=…&date=…`, confirmed against it on 2026-09-20; `symbols` is now `quotes`, and the response is an array of `{date, base, quote, rate}` rather than a rates map. The transaction's currency is sent as the API's `base` and the reporting currency as its `quotes`, because Frankfurter reports quote-per-base and this contract stores reporting-per-transaction; the provider validates that the response names the pair it asked for, so an API that flipped direction would fail loudly rather than invert every converted amount.

2. **Weekends are normalised locally, before the call.** The plan assumed the provider would report the true business day. It does not: asked for a Saturday it returns the carried-forward rate stamped *with the Saturday*, which is indistinguishable from a Saturday observation. Saturdays and Sundays therefore resolve back to the preceding Friday before the lookup, which also lets a weekend's transactions share one cache entry. Holidays are still left to the provider's carry-forward — there is no local TARGET calendar and adding one would be a dependency to answer a question the provider already answers.

3. **Amount filters keep filtering on `amount`.** §18 said filters as well as volume KPIs would use `COALESCE(amount_base, amount)`. They do not. Aggregates are corrected because summing mixed currencies adds euros to yen; a filter is a different question — "show me transactions over 250" asks what the cardholder was charged. Changing it would also alter existing query semantics and touch `repositories/transaction.py`, which §25 protects.

4. **No FX hook in the batch feature builder.** §18 proposed exposing `get_rate` there for a future non-USD dataset. Every offline dataset is single-currency, so the hook would be dead code inside the benchmark tooling — which is exactly the contamination Phase 5 is careful about elsewhere. When a non-USD dataset arrives, the hook can be added against the same cache table with the same semantics, which is what that paragraph was really protecting.

One addition the plan did not call for: `fx_source` and `fx_rate_date` are written into the scoring audit payload whenever the source is not `identity`. §18 asked for `fx_source` in the audit; the rate date is there too, because "priced with a stale rate" is only actionable if you can see *how* stale.

The UI work — currency-aware amount rendering and the `fx_source` chip (§20) — stayed in 5G with the rest of the dashboard pass. 5F ended at the API boundary; 5G rendered it.

## 19. Simulator Calibration (NICE-TO-HAVE; first NICE item after v2)

`ml/analysis/simulator_profile.py --dataset sparkov` fits from the canonical dataset: category weights (12-taxonomy), per-category log-normal `(mu, sigma)` for amounts, card-present share per category. Writes `app/simulator/profiles/sparkov.json` (small derived statistics, committed; source CC0). Simulator gains `--profile sparkov` (merchant chosen by category weight, amount from category log-normal, card-present from category share) and `--currency-mix "USD:0.85,EUR:0.08,GBP:0.05,INR:0.02"` (amounts emitted in that currency; FX path exercised). Existing three fraud patterns and `--fraud-rate` unchanged. Time-of-day is not calibrated (wall-clock stamped — F10).

## 20. Dashboard/Demo Changes (small)

- `GET /api/v1/model` → `{dataset_name, featureset_version, trained_at_utc, threshold, train_period, test_period, pr_auc, recall_at_1pct_fpr}` read from the loaded artifacts.
- Sidebar footer badge: "model: sparkov · v1 · 2026-xx-xx" (replaces the static phase text); tooltip with the metrics.
- Transaction detail: currency shown next to amount; base amount when different; `fx_source` chip when stale/unavailable; `featureset_version` line in the score panel.
- Transactions list: amount column shows currency code.
- Demo snapshot (`scripts/export_demo_snapshot.py`) regenerated with the promoted Sparkov model; `model.json` snapshot added to `demo-data/`.
- README: Phase 5 section, honest framing block (§1), corrected test count, updated headline table with three columns (synthetic v1 / Sparkov v1 / ULB baseline).

No new pages. No market-data pane.

### 20.1 As built (5G)

Delivered as specified, with four deviations and one addition.

1. **The badge reports `synthetic_v1`, not `sparkov_v1`.** §20 assumed a promoted Sparkov model. Promotion has still not been run — deliberately — so the served model is the in-house synthetic one, and the badge says so. Writing the planned string would have been a false claim about what is answering requests.

2. **The `/model` envelope is larger than the field list in §20.** It carries the serving block *and* a `benchmarks` list naming all three tracks with a `served` flag on each. The field list alone would let a reader assume the numbers describe production performance on real data. `metrics_caveat` ships with the metrics for the same reason: a bare figure on a dashboard is the most common way an honest number becomes a dishonest claim.

3. **`dataset_name` is a documented constant, not an artifact read.** The current artifacts predate recording their own lineage, and §24's plan to add provenance fields to `ml/artifacts.py` would only populate them on a retrain — which 5G is not permitted to do. The constant's provenance is argued in `app/services/model_info.py`; a future run that records the field will be preferred automatically.

4. **No `fx_source` chip and no `featureset_version` line in the score panel.** The FX state is carried by the `Amount` component instead, where the converted figure actually appears — a separate chip elsewhere on the page would put the caveat further from the number it qualifies. The featureset version is on the dashboard's Model & data panel rather than repeated on every detail page.

The addition: `--currency-mix` on the simulator (the currency half of §19, without the profile calibration). Without it the simulator emits USD only, so the demo snapshot could not show the FX path the previous milestone built. Its currency set is restricted to currencies within roughly a factor of two of USD, because the rules engine's thresholds are denominated in the raw amount — see the note on `_MIXABLE_CURRENCIES`.

`export_demo_snapshot.py` gained `model.json`, currency and `fx_source` coverage in its detail-page selection, and pruning of detail pages left behind by a previous export.

## 21. Testing Strategy

Preserve all 220 existing tests. Add roughly 40–50 high-value tests:

| Area | Tests (unit unless noted) |
|---|---|
| Adapter | every Sparkov category mapped; `_pos`/`_net` → card-present; deterministic `uuid5` ids; no attribute of any canonical object contains raw `cc_num` or `is_fraud`; output sorted by `created_at`; entity subsample keeps complete card histories; fraud-rate tolerance; provenance sha256 recorded |
| Quality report | counts/nulls/dupes/percentiles on a fixture; dropped/derived/unavailable inventory |
| Batch builder | 180-day window eviction; `_NoDatabase` sentinel never touched; future-information test (§12); **golden parity vs SQL path** incl. > 180-day gap case; cache key changes when featureset changes |
| Registry | v1 literal pin; `feature_list.json` version validation; explainer rejects wrong version; `/explain` echoes version (integration) |
| Artifacts | metadata fields present and typed; `promote` copies exactly the five files and refuses non-production featuresets |
| Splits / drift | period bounds recorded; drift script months strictly after train period |
| Rules audit | per-rule counts on a fixture with known outcomes |
| FX | identity for USD (no call); cache hit; API success upserts; weekend → prior business day; API timeout → stale with flag; API down + empty cache → unavailable, transaction still scored (integration); audit payload carries `fx_source` |
| Stats | volume uses base amount (integration) |
| Simulator (NICE) | profile loads; amounts within category p1–p99 of profile; currency-mix proportions within tolerance over N samples |
| Endpoint | `/api/v1/model` shape (integration) |
| CI | GitHub Actions: `ruff`, `mypy --strict`, `pytest` (backend, with a tiny generated dataset + fast train step if integration tests require artifacts), `tsc`/`vite build` (frontend) |

## 22. Data Licensing / Provenance

- `docs/DATA_LICENSES.md`: per source — name, URL, license as displayed at retrieval date, citation, what we derive from it, what we commit (only derived statistics and metrics), what we never commit (raw data). Sparkov: CC0 (confirm); ULB: Kaggle Open Database–family license, attribution to ULB MLG/Worldline; Frankfurter: open API over public ECB reference rates; simulator profile JSON: derived statistics from Sparkov.
- `scripts/download_data.py`: uses the Kaggle CLI/API if credentials exist, otherwise prints the two dataset URLs and expected filenames; verifies SHA-256 against `data/manifest.json` (committed) and refuses to proceed on mismatch. `.gitignore` already excludes `backend/ml/data/*.csv`.
- Every artifact directory carries `training_metadata.json` with the file hashes, so a model can always be traced to exact input bytes.

---

## 23. Files to Add

```
.github/workflows/ci.yml
docs/adr/PHASE_5A_DESIGN.md
docs/DATA_LICENSES.md
scripts/download_data.py
backend/ml/data/manifest.json                      # expected sha256 per raw file (committed)
backend/ml/datasets/__init__.py
backend/ml/datasets/base.py                        # DatasetAdapter protocol, CanonicalDataset, DatasetProvenance
backend/ml/datasets/sparkov.py
backend/ml/datasets/quality.py
backend/ml/features/__init__.py
backend/ml/features/batch.py                       # build_feature_matrix (production in-memory path)
backend/ml/promote.py
backend/ml/experiments/__init__.py
backend/ml/experiments/temporal_drift.py
backend/ml/experiments/rules_audit.py
backend/ml/tracks/ulb/{__init__,load,train}.py
backend/ml/MODEL_CARD_ULB.md
backend/ml/artifacts/runs/.gitkeep                 # per-run dirs; model.json still gitignored
backend/app/enrichment/__init__.py
backend/app/enrichment/fx.py
backend/app/models/fx_rate.py
backend/app/repositories/fx_rate.py
backend/app/api/v1/model.py                        # GET /api/v1/model
backend/app/schemas/model_info.py
backend/alembic/versions/<ts>_add_fx_columns_and_fx_rates.py
backend/tests/unit/test_dataset_sparkov.py
backend/tests/unit/test_dataset_quality.py
backend/tests/unit/test_features_batch_parity.py
backend/tests/unit/test_feature_registry.py
backend/tests/unit/test_promote.py
backend/tests/unit/test_rules_audit.py
backend/tests/unit/test_fx_service.py
backend/tests/integration/test_fx_ingestion.py
backend/tests/integration/test_model_endpoint.py
frontend/src/components/layout/ModelBadge.tsx
frontend/src/hooks/useModelInfo.ts
frontend/public/demo-data/model.json
# NICE:
backend/ml/analysis/simulator_profile.py
backend/app/simulator/profiles/sparkov.json
backend/tests/unit/test_simulator_profile.py
```

## 24. Files to Modify

| File | Change |
|---|---|
| `backend/app/fraud/feature_spec.py` | `FEATURESETS`, `DEFAULT_FEATURESET`, `feature_names()`; keep `FEATURE_NAMES` alias |
| `backend/app/fraud/features.py` | `featureset: str = DEFAULT_FEATURESET` parameter; assert against `feature_names(featureset)`; no v1 logic changes |
| `backend/app/fraud/explainer.py` | validate `featureset_version`; expose `featureset_version`, `dataset_name` |
| `backend/ml/artifacts.py` | `TrainingMetadata` gains provenance fields; `feature_list.json` gains version; `runs/` path helper |
| `backend/ml/train.py` | `--dataset`, `--featureset`, `--max-cards`, `--run-name`; dataset dispatch; period bounds in metadata |
| `backend/ml/data.py` | synthetic path unchanged; add `from_canonical()` that delegates to `ml/features/batch.py` |
| `backend/ml/analyze.py` | accept `--run-name`; write model card per run; comparison table across runs |
| `backend/app/services/transactions.py` | FX enrichment call + new columns |
| `backend/app/services/scoring.py` | audit payload includes `fx_source` when not identity; actor string gains dataset/featureset (`scorer:sparkov/v1/<ts>`) |
| `backend/app/models/transaction.py` | four nullable FX columns |
| `backend/app/repositories/stats.py` | `COALESCE(amount_base, amount)` for volume aggregates |
| `backend/app/schemas/transaction.py`, `explanation.py` | `amount_base`, `fx_source`, `featureset_version` in responses |
| `backend/app/schemas/stats.py` | no shape change; document base-currency semantics |
| `backend/app/main.py`, `api/v1/__init__.py` | mount `/api/v1/model` |
| `backend/app/config.py` | `FX_BASE_CURRENCY="USD"`, `FX_API_BASE_URL`, `FX_TIMEOUT_SECONDS=2.0`, `FX_ENABLED=True` |
| `backend/app/simulator/main.py` | NICE: `--profile`, `--currency-mix` |
| `scripts/export_demo_snapshot.py` | export `model.json` |
| `frontend/src/lib/format.ts`, `types/`, `components/transactions/*`, `Sidebar.tsx`, `demoApi.ts` | currency-aware amount formatting; badge; demo route for `/model` |
| `README.md`, `docs/ARCHITECTURE.md` | Phase 5 section, framing block, offline/live diagram, test count |
| `.gitignore` | `backend/ml/artifacts/runs/**/model.json`, `backend/ml/data/cache/` |

## 25. Files NOT to Modify Unless Necessary

`app/fraud/rules.py`, `decision.py`, `transaction_context.py`; `services/idempotency.py`, `alerts.py`, `review.py`, `transaction_detail.py`; `repositories/transaction.py` (keyset pagination), `audit.py`; `api/v1/alerts.py`; all existing Alembic versions; `ml/splits.py`, `ml/evaluation.py`, `ml/tuning.py`, `ml/analysis/*` (reuse, don't edit); `ml/synthesis/*` (the synthetic track is frozen as the "v1 baseline"); frontend alerts and dashboard components; every existing test (may add fixtures, never weaken assertions).

---

## 26. Implementation Sequence

### M0 — Phase 4F-lite + CI baseline (≤ 1 day)
- **Objective:** a public URL exists; every PR runs the test suite.
- **Files:** `.github/workflows/ci.yml`, `vercel.json` (exists), README top section.
- **Details:** deploy `VITE_DEMO_MODE=true` build to Vercel; CI job matrix backend (uv sync → ruff → mypy → pytest; if integration tests need artifacts, add a fast `generate_dataset --n-transactions 3000` + `train --n-iter 2` step) and frontend (npm ci → tsc → build).
- **Tests:** existing suite green in CI.
- **DoD:** green badge in README; live URL at the top of README; Loom deferred to M8.

### M1 — 5A: ADR, licensing, download script (1–1.5 days)
- **Objective:** design locked before code; provenance rules written down.
- **Files:** `docs/adr/PHASE_5A_DESIGN.md`, `docs/DATA_LICENSES.md`, `scripts/download_data.py`, `ml/data/manifest.json`.
- **Details:** ADR covers adapter protocol, canonical-schema invariant, 180-day window contract, subsampling policy, featureset registry, artifact provenance fields, FX semantics and failure policy, the F3/F4/F5 findings, MUST/NICE/FUTURE table. Download script verifies sha256.
- **Tests:** `test_download_manifest` (hash mismatch refuses).
- **DoD:** ADR merged; both datasets download and verify locally.

### M2 — 5B: Sparkov adapter + quality report (3 days)
- **Objective:** `CanonicalDataset` from Sparkov, validated.
- **Files:** `ml/datasets/{base,sparkov,quality}.py`; tests.
- **Details:** §9 mapping; entity subsampling; provenance; quality JSON; CLI `python -m ml.datasets.sparkov --max-cards 200 --report`.
- **Tests:** adapter suite (§21).
- **DoD:** full-corpus load < 3 min; quality report for full and sampled runs committed under `ml/artifacts/runs/sparkov_v1/`.

### M3 — 5C: batch feature builder + registry + parity (2.5 days)
- **Objective:** production-semantics features for the whole canonical dataset; versions explicit.
- **Files:** `ml/features/batch.py`, `feature_spec.py`, `features.py`, `explainer.py`, `ml/artifacts.py`; tests.
- **Details:** §11, §13; `.npz` cache; `_NoDatabase` sentinel.
- **Tests:** golden parity, future-information test, registry pin, explainer version check.
- **DoD:** parity suite green; feature matrix for 200 cards builds and caches; `feature_list.json` carries version.

### M4 — 5D: train, evaluate, drift, rules audit, model card (3–4 days)
> **Methodology locked before implementation** in [`adr/PHASE_5D_BENCHMARK_METHODOLOGY.md`](adr/PHASE_5D_BENCHMARK_METHODOLOGY.md). Where it differs from this plan, the decision record governs: live features are counted from the matrix rather than stated as 13/17 (§4.3); dataset provenance goes in `run.json` and Sparkov `metrics.json` carries no synthetic-era targets (§14); the M4 comparison is synthetic baseline, Sparkov in-domain and synthetic→Sparkov transfer, with ULB left to M5; drift hyperparameters are tuned on 2019-01 → 2019-10 only (§12, §15); the dormant-account rule is reported as not evaluable on Sparkov (§16); `ml.promote` is built but a Sparkov run is promoted only on separate approval after benchmark review, so the promotion step in the DoD below is not performed as part of M4.

- **Objective:** honest external numbers with provenance.
- **Files:** `ml/train.py`, `ml/analyze.py`, `ml/promote.py`, `ml/experiments/*`; `MODEL_CARD.md` regeneration; README metrics table.
- **Details:** §14–§16; runs `synthetic_v1` (re-run today's pipeline into `runs/` for a like-for-like baseline), `sparkov_v1_200cards` (dev), `sparkov_v1_full` (once, overnight if needed); cross-generator transfer number; drift PNG; rules audit JSON.
- **Tests:** promote guard; metadata fields; drift period ordering; rules audit fixture.
- **DoD:** `python -m ml.promote sparkov_v1_full` makes the API serve the Sparkov model; model card shows the three-way comparison, drift plot, rules audit, and the 13/17 live-feature accounting.

### M5 — 5E: ULB track (1.5 days)
> **Methodology locked before acquisition** in [`adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md`](adr/PHASE_5E_ULB_BENCHMARK_METHODOLOGY.md). Where it differs from this plan, the decision record governs: ULB is not adapted into the canonical schema and featureset v1 is not evaluated on it; its own featureset `ulb_pca_v1` is `V1`–`V28` and `Amount` as published, not `log1p(Amount)` (§7, §17), and stays outside the production registry so promotion refuses it; tuning keeps the full 25 iterations (§17); the three seeds are the runs `ulb_pca_v1_seed42`, `_seed43` and `_seed44` rather than `runs/ulb_baseline/`; the card is `backend/ml/ULB_BENCHMARK_CARD.md` rather than `MODEL_CARD_ULB.md` (§23); and ULB results get their own table and card instead of a column beside the v1 results (§20, §29 item 5).

- **Objective:** real-data benchmark with the same discipline.
- **Files:** `ml/tracks/ulb/*`, `MODEL_CARD_ULB.md`.
- **Details:** §17; 3 seeds for variance.
- **Tests:** loader shape/label test; promote refusal test.
- **DoD:** `runs/ulb_baseline/` metrics + card; README row.

### M6 — 5F: FX enrichment (2.5 days) — **DONE**
- **Objective:** multi-currency analytics with tested failure modes.
- **Files:** migration, `models/fx_rate.py`, `repositories/fx_rate.py`, `enrichment/fx.py`, `services/transactions.py`, `scoring.py` (audit payload), `stats.py`, schemas, config; tests.
- **Details:** §18, as built in §18.1.
- **Tests:** FX suite (§21) incl. integration "API down, still scored."
- **DoD:** a `EUR` POST is scored, stored with `amount_base`, appears in volume KPIs in USD, and its audit row shows `fx_source`; with network disabled the same POST still returns 201 with `fx_source="unavailable"`.
- **As built:** all of the above, plus `enrichment/provider.py` splitting the HTTP boundary from the conversion logic, and `docs/FX_CONTRACT.md`. Ingestion is enriched in `api/v1/transactions.py` rather than `services/transactions.py`, which is the list service — the router calls one `fx.enrich(db, tx)` beside the `score_transaction` call it already orchestrates. 149 tests added across five files; the four deviations from §18 are recorded in §18.1. The UI half of the DoD ("currency-aware amounts") belongs to M7.

### M7 — 5G: badge, currency UI, docs, demo snapshot (2 days) — **DONE**
- **Objective:** the demo tells the Phase 5 story.
- **Files:** `/api/v1/model`, `ModelBadge.tsx`, currency formatting, `export_demo_snapshot.py`, README, `ARCHITECTURE.md`.
- **Tests:** endpoint integration; `tsc`.
- **DoD:** Vercel demo shows the badge and currency-aware amounts; README has the framing block, Phase 5 section, and the comparison table.
- **As built:** §20.1. The badge is a sidebar line plus a dashboard Model & data panel rather than a single `ModelBadge.tsx`; currency formatting is one `Amount` component over an extended `format.ts`; the snapshot was regenerated against the live stack with a `--currency-mix` simulator run, so every FX state in it was produced by the system rather than authored. The README carries the framing and the Phase 5 section; the three-column comparison table was **not** added, because the results are laid out in their own cards and the user's standing decision is not to duplicate benchmark numbers into the README.

### M8 — NICE (only if M0–M7 are done): v2 features → simulator profile + currency mix → Loom (week 4)
- v2 (§11, 1.5 days incl. evaluation and decision); simulator profile (§19, 1.5 days); Loom 4E recorded last against the live stack with `--profile sparkov --currency-mix ...` (0.5 day).

---

## 27. MUST HAVE vs NICE TO HAVE vs FUTURE

**MUST HAVE (Phase 5 is not done without these):** M0 CI + Vercel deploy; M1 ADR, licenses, download/manifest; M2 Sparkov adapter, entity subsampling, quality report; M3 batch builder on the production path, featureset registry (v1), golden parity + future-information tests; M4 Sparkov training with provenance, promote, three-way metrics, temporal drift experiment, rules audit, cross-generator transfer number, regenerated model card; M5 ULB track (minimal); M6 FX enrichment (minimal, tested failure modes); M7 model badge, currency-aware amounts, README/ARCHITECTURE/demo refresh.

**NICE TO HAVE (week 4, in this order, stop when time runs out):** featureset v2 with v1-vs-v2 evaluation; simulator profile + `--currency-mix`; `amount_base`-as-model-input experiment (only after v2, only with evaluation); `TimeSeriesSplit` for hyperparameter search; segment analysis on Sparkov by category; Loom re-record (do this one regardless, last).

**FUTURE (documented, not built):** coordinate-based geo features and `merchant_lat/long` columns (v3); IEEE-CIS research track; FRED context tile; rule threshold parameterization and rule versioning; analyst-label feedback into retraining; time-aware `account_age_days`; entity-held-out evaluation variant; Postgres deployment; everything in §31.

---

## 28. 2–4 Week Execution Plan

| Week | Days | Milestones | Exit |
|---|---|---|---|
| 1 | 1 | M0 (deploy + CI) | live URL, green CI |
| 1 | 2–5 | M1, M2 | ADR merged; Sparkov loads and validates |
| 2 | 6–8 | M3 | parity green; cached features |
| 2 | 9–10 | M4 (start: synthetic baseline re-run, Sparkov dev run) | first honest numbers |
| 3 | 11–12 | M4 (full run, drift, rules audit, model card) | Sparkov model promoted |
| 3 | 13 | M5 | ULB card |
| 3 | 14–15 | M6 | FX end-to-end with failure tests |
| 4 | 16–17 | M7 | demo + README refreshed — **Phase 5 DONE** |
| 4 | 18–20 | M8 NICE + Loom | v2 decision, profile, video |

Phase 4 ordering, explicitly: do **4F-lite first** (M0, hours) because a referral conversation needs a clickable URL and the deploy work already exists in 4C. Do **not** finish 4E first — a Loom recorded now would be obsolete in three weeks; record it once at the end of M8. Mark the README's Phase 4 as "4E deferred to end of Phase 5" so the roadmap stays honest.

## 29. Final Definition of Done

Phase 5 is complete when all of the following are true on `main`:
1. CI is green on ruff, mypy --strict, pytest (≥ 260 tests), tsc/build; a Vercel URL is at the top of the README.
2. `python scripts/download_data.py` verifies both datasets; no raw data is tracked by git.
3. `python -m ml.train --dataset sparkov --run-name sparkov_v1_full` reproduces the committed `metrics.json` (same seed, same hashes) and `python -m ml.promote sparkov_v1_full` makes the API serve it; `/api/v1/model` and `/explain` report `sparkov` / `v1`.
4. `tests/unit/test_features_batch_parity.py` passes, including the > 180-day-gap case and the future-information test.
5. `MODEL_CARD.md` contains the three-way table (synthetic v1 / Sparkov v1 / ULB baseline), the temporal drift plot with a written interpretation, the rules audit table, the live-feature accounting (13/17), the cross-generator transfer number, and the leakage/skew findings (F3, F4, F5) written in plain language.
6. A non-USD transaction posted to the API is scored, stored with `amount_base`, counted correctly in volume KPIs, and its audit row records the FX source; the "FX unavailable" integration test passes.
7. `docs/adr/PHASE_5A_DESIGN.md` and `docs/DATA_LICENSES.md` exist; README carries the honesty framing block and a Phase 5 section; `docs/ARCHITECTURE.md` shows the offline/live split.
8. Nothing in §6 was built.

---

## 30. Barclays SWE Interview Talking Points

1. **Canonical schema as the integration contract.** "Every data source — my simulator, Sparkov, the API — is adapted *into* one ORM schema; nothing is adapted out. The extractor and rules only ever see that schema, so adding a dataset is an adapter, not a rewrite." (Layered architecture, dependency direction.)
2. **Offline/live separation with one touchpoint.** "Training never calls the API; the API never reads training data. The only shared surface is the artifact directory, which carries the dataset hash, featureset version, periods, threshold and metrics." (Traceability without a registry.)
3. **Train/serve skew — found, measured, fixed.** "My training path used unbounded history and my serving path used a 180-day window. Identical on 30 days of data, divergent on 24 months. I made the batch builder call the production in-memory path with the production window and wrote a parity test that fails if they ever drift." (The single strongest story in the repo.)
4. **Leakage you build in yourself.** "My generator injected fraud preferentially into high-risk-tier customers, so a 'risk tier' feature was partly the label. External data with no tiers exposed it. Metrics went down; credibility went up."
5. **Entity subsampling.** "Row sampling deletes history and silently corrupts velocity features; I sample cards and keep every transaction they made."
6. **Chronological splits and why random splits lie.** "Fraud is non-stationary and history features encode time; a shuffled split leaks the future. Split by time, tune on train, threshold on val, report on test — and then re-measure month by month."
7. **Why PR-AUC and Recall@FPR, not accuracy.** "At 0.5% prevalence a constant classifier is 99.5% accurate. The business question is 'how much fraud do I catch within a false-positive budget?' So I fix FPR at 1% and read recall, and I record the realised FPR of the chosen threshold."
8. **Rules and ML as defence in depth — and rules don't transfer for free.** "Hard rules short-circuit, review rules escalate, the model decides the rest; conservative wins. On the external dataset only two of six rules were structurally live, and I have the table that shows it."
9. **Explainability as an audit requirement, not a feature.** "SHAP contributions are computed at decision time and stored on the row. The detail page reads what was decided, not a recomputation — that's what an auditor needs."
10. **External dependency in a hot path.** "FX rates come from a public ECB feed with a two-second timeout, a local cache, a stale fallback, and an 'unavailable' state that is recorded in the audit log. The scoring path never blocks on it."
11. **Idempotency done properly.** "Same key + same canonical body → cached response with a replay header; same key + different body → 409. Hash of the validated model, not raw bytes."
12. **Knowing where to stop.** "I could have built Kafka, a feature store and a model registry. None of them would have changed a single fraud decision in this system; the parity test and the drift plot did. At bank scale, here's how each of those would come in…" (→ §31).

## 31. Future Architecture — What We Would Do at Bank Scale

Not built in Phase 5; described so the interview answer exists.

- **Ingestion:** the HTTP endpoint becomes a producer to a durable log (Kafka/Redpanda); scoring consumers read partitions keyed by customer so per-customer ordering — the property the velocity features depend on — is preserved. The idempotency key becomes the message key.
- **Feature serving:** the 180-day in-memory window becomes a materialised online feature store (Redis/Feast-style) updated from the same stream; the offline batch builder becomes the point-in-time-correct backfill job. The parity test from Phase 5 becomes the contract test between the two.
- **Model lifecycle:** `runs/` + `promote` become a model registry with champion/challenger, shadow scoring on live traffic, and the drift script running as a scheduled job with alerting on realised FPR.
- **Rules:** externalised to a rules service with versioned configuration and an approval workflow (four-eyes), because rule changes are control changes.
- **Storage:** Postgres with partitioning by time; append-only audit log shipped to immutable storage for retention.
- **Data:** real transactions never appear in a portfolio — at a bank they would arrive through internal event streams under the institution's data-governance, tokenisation and access controls, and every model would go through model-risk validation before promotion.
- **Observability:** OpenTelemetry traces across ingestion → scoring → decision, latency SLOs per stage, and per-segment metric dashboards.

Everything above scales the same logical pipeline that Phase 5 leaves behind: adapter → canonical schema → past-only features → rules + model + SHAP → decision → audit → review.
