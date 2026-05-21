# Model card — Fraud Radar XGBoost classifier

> Auto-regenerated from `backend/ml/analyze.py` using the artifacts in
> `backend/ml/artifacts/`. Re-run after every training run; do not hand-edit.
> Last generated: `2026-05-21T17:09:43+00:00`.

## 1. Model details

- **Name:** Fraud Radar XGBoost classifier
- **Version:** trained at `2026-05-21T12:47:25+00:00`
- **Type:** gradient-boosted decision tree ensemble (binary classifier)
- **Training framework:** XGBoost 3.2.0, scikit-learn 1.8.0, Python 3.13.13
- **Intended deployment:** real-time scoring of card transactions inside the Fraud Radar service
- **Owner:** [apoorvrajdev](https://github.com/apoorvrajdev)

## 2. Intended use

Produce a fraud-probability score for a single card transaction at the point
of authorisation. Output is one of three actions — `APPROVE`, `REVIEW`,
`DECLINE` — determined by a calibration threshold and a business-tuned review
band (`threshold * 0.5` for the REVIEW/APPROVE boundary today).

**Not intended for:**

- credit-worthiness scoring
- customer segmentation or marketing
- identity verification or KYC
- any decision that requires demographic fairness guarantees — the training
  data is synthetic and contains no protected attributes

## 3. Training data

50,010 synthetic transactions across 500 customers and
200 merchants. Train fraud rate `0.0160`,
validation `0.0141`, test
`0.0124`. Six fraud patterns are injected
by [`backend/ml/synthesis/`](synthesis/): card testing, geo-velocity, account
takeover, amount anomaly, off-hours, and merchant concentration. Fully
reproducible from `seed=42`.

Splits are strictly chronological by `created_at`: oldest 70% is train, next
15% val, last 15% test (`7502` rows). No row sees
its own future.

## 4. Performance — overall

| Metric | Value |
|---|---|
| PR-AUC | 0.9327 |
| ROC-AUC | 0.9989 |
| Recall @ 1% FPR | 0.9785 |
| Recall @ 5% FPR | 1.0000 |
| Operating threshold | 0.7431 |
| Precision @ threshold | 0.6138 |
| Recall @ threshold | 0.9570 |
| F1 @ threshold | 0.7479 |

Source: `artifacts/metrics.json`.

## 5. Performance — by segment

Geographic segmentation. Buckets are mutually exclusive and collectively
exhaustive. **Note:** this is performance stability, not demographic
fairness — the synthetic dataset has no protected attributes.

| Segment | n | n_frauds | n_neg | fraud_rate | PR-AUC | Recall@1%FPR |
|---|---|---|---|---|---|---|
| US | 3945 | 49 | 3896 | 0.0124 | 0.9210 | 0.9388 |
| Developed | 2933 | 20 | 2913 | 0.0068 | 0.9245 | 1.0000 |
| Other | 608 | 8 | 600 | 0.0132 | 1.0000 | 1.0000 |
| High-fraud | 16 | 16 | 0 | 1.0000 | — | — |

Unmeasurable on this synthetic dataset: High-fraud. Those country codes appear only in fraud-injected transactions by design of the synthetic generator, so PR-AUC and Recall@FPR have no defined value on those buckets — no claim about model behaviour on real high-fraud geographies can be made from these numbers.

Source: `artifacts/segment_metrics.json`.

## 6. Calibration

| Metric | Aggregate | Positives only |
|---|---|---|
| Brier score | 0.0082 | 0.0159 |
| Expected calibration error | 0.0131 | 0.4247 |

Aggregate Brier and ECE look strong because **96.5%** of test samples land in bin 0 and are correctly assigned near-zero fraud probability. The positive-restricted variants strip out this dominant well-calibrated negative class and surface the systematic over-prediction on harder cases — bin 7 predicts 0.75 fraud probability where only 6% of those transactions are actually fraud. Production deployment would apply Platt or isotonic calibration as a post-hoc step before using the probabilities for thresholded decisions.

See [`artifacts/calibration_curve.png`](artifacts/calibration_curve.png).

## 7. Feature importance

Top 5 features by mean absolute SHAP value on the test set:

| Rank | Feature | Mean abs SHAP | Interpretation |
|---|---|---|---|
| 1 | `log_amount` | 2.7717 | transaction amount on a log scale — large amounts are unusual and risky |
| 2 | `is_card_present` | 2.5195 | card-present vs card-not-present at the merchant terminal |
| 3 | `hour_of_day` | 0.7230 | the wall-clock hour the transaction occurred |
| 4 | `country_mismatch_customer` | 0.7009 | transaction country differs from the cardholder's home country |
| 5 | `merchant_risk_encoded` | 0.5105 | the merchant's category risk tier (LOW/MEDIUM/HIGH) |

See [`artifacts/global_shap_beeswarm.png`](artifacts/global_shap_beeswarm.png)
and [`artifacts/global_shap_bar.png`](artifacts/global_shap_bar.png).

## 8. Limitations

- **Synthetic data.** The training set is generated, not collected. Real
  fraud patterns drift continuously; this model captures only the six
  patterns explicitly injected by the generator.
- **No concept-drift handling.** There is no online retraining, no
  population-stability monitoring, no drift alerts.
- **No adversarial robustness testing.** A fraudster who learns the feature
  schema could craft transactions that score below threshold.
- **No real-world deployment.** This model has never scored a real
  transaction. Performance numbers reflect a synthetic distribution and
  will not transfer to production without retraining on real data.

## 9. Ethical considerations

The synthetic dataset contains no demographic protected attributes (race,
gender, religion, etc.), so no demographic fairness analysis was performed.
The segment-by-country breakdown above is a partial proxy for geographic
stability — it is **not** a substitute for a proper fairness audit, which
would require a real dataset and a defined protected-class schema.

If this model were ever deployed against real data, a fairness audit across
protected classes would be mandatory before launch.
