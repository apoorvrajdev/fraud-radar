# Data Sources, Licences and Provenance

Fraud Radar trains and evaluates on datasets it does not own. This file records what each source is, under what terms it is used, and what of it ever enters this repository.

**Status:** Phase 5A. Nothing external has been downloaded yet. Licence rows below are what the source advertises publicly; each is confirmed at retrieval time (Phase 5B) and the confirmed wording, with its retrieval date, is written into the dataset's `DatasetProvenance.license` field and into every run that uses it.

---

## What is committed, and what is not

| Committed | Never committed |
|---|---|
| SHA-256 of every raw input file | The raw data files themselves |
| Derived statistics: counts, rates, percentiles, quality reports | Any row-level extract of a third-party dataset |
| Metrics, model cards, run metadata | Feature matrices built from third-party data (cached locally, gitignored) |

`backend/ml/data/raw/` and `backend/ml/data/cache/` are gitignored in full. Because raw bytes are absent, the file hash recorded in `run.json` is the durable link between a published number and the data that produced it — anyone can re-download, hash, and compare.

---

## Sources

### Fraud Radar synthetic generator (in-house)

| | |
|---|---|
| **Origin** | Synthetic |
| **Location** | `backend/ml/synthesis/`, reproducible from `seed=42` |
| **Licence** | MIT, with this repository |
| **Label** | `is_fraud`, assigned by the injection pattern that generated the row |
| **Used for** | The v1 baseline model currently served by the API |
| **Caveat** | Features were designed alongside the generator, so its numbers measure learnability of the generator, not detectability of fraud. This is the closed loop Phase 5 exists to break. |

### Sparkov — external synthetic benchmark (planned, 5B)

| | |
|---|---|
| **Origin** | Synthetic, generated independently with Brandon Harris's Sparkov tool |
| **Source** | Kaggle `kartik2112/fraud-detection` |
| **Licence** | Listed as CC0 on the dataset card — **to be confirmed at retrieval** |
| **Citation** | Kartik Shenoy, "Credit Card Transactions Fraud Detection Dataset", Kaggle; generated with Sparkov Data Generation (Brandon Harris) |
| **Label** | `is_fraud` |
| **Used for** | Training, evaluation, temporal-drift experiment, rules audit |
| **Derived and committed** | Quality report, metrics, model card, feature-importance summaries |

### ULB Credit Card Fraud — real-world benchmark (planned, 5E)

| | |
|---|---|
| **Origin** | Real, anonymised |
| **Source** | Kaggle `mlg-ulb/creditcardfraud` — 284,807 European card transactions over two days in September 2013, 492 frauds (0.172%) |
| **Licence** | Open Database–family licence as displayed on the dataset card — **to be confirmed at retrieval** |
| **Attribution** | Machine Learning Group, Université Libre de Bruxelles, with Worldline |
| **Citation** | Dal Pozzolo, Caelen, Johnson and Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI, 2015 |
| **Label** | `Class` |
| **Used for** | A separate benchmark track only. Features are PCA components of undisclosed inputs, so entity identity, merchants and geography are unrecoverable and no velocity feature can honestly be built on them. Never promoted to the serving API. |

### Frankfurter — FX reference rates (planned, 5F)

| | |
|---|---|
| **Origin** | Public reference data |
| **Source** | `https://frankfurter.dev` — open API over European Central Bank daily reference rates |
| **Licence** | Open API over public ECB data; no key required, self-hostable |
| **Used for** | Converting a transaction amount to a base currency for display and analytics at ingestion time. Cached locally. Not a model input. |
| **Failure policy** | Never blocks scoring: on timeout or outage the transaction is scored and stored with the FX source marked stale or unavailable. |

---

## Sources deliberately excluded

**IEEE-CIS.** The richest real card-not-present dataset available, but competition-rules licensing restricts redistribution and use, and its semantics are masked. Documented as possible future research; not used in Phase 5.

**FRED and market-data APIs** (Alpha Vantage, Finnhub, FMP). None carries card transactions or fraud labels, so none can train, evaluate or enrich a fraud model. A macro tile on the dashboard would imply a relationship to fraud decisions that does not exist.

---

## Adding a source

1. Record it here before writing the adapter: name, URL, licence as displayed, retrieval date, citation, what is derived from it, what is committed.
2. Add expected filenames and SHA-256 digests to the download manifest (5B).
3. Populate `DatasetProvenance` in the adapter so every run carries licence and hashes with it.
4. If the licence forbids redistribution of derived artifacts, say so here and keep those artifacts out of the repository.
