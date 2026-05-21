"""Post-training analysis runner.

Loads the trained model + dataset, re-applies the same chronological split,
scores the held-out test fold, and writes the segment metrics, calibration
metrics, global feature importance, plot PNGs, and the regenerated
MODEL_CARD.md.

Usage:
    cd backend
    uv run python -m ml.analyze
    uv run python -m ml.analyze --limit 5000  # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from app.db import SessionLocal
from app.fraud import initialize_explainer
from app.fraud.feature_spec import FEATURE_NAMES
from ml.analysis import (
    compute_calibration_metrics,
    compute_feature_importance,
    compute_segment_metrics,
    render_bar_plot,
    render_beeswarm_plot,
    render_calibration_plot,
)
from ml.data import load_dataset_with_csv_labels
from ml.splits import chronological_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger("analyze")


# ---------------------------------------------------------------------------
# Feature interpretations for the model-card table. Hand-written; one line each.
# ---------------------------------------------------------------------------

_FEATURE_INTERPRETATIONS: dict[str, str] = {
    "log_amount": "transaction amount on a log scale — large amounts are unusual and risky",
    "hour_of_day": "the wall-clock hour the transaction occurred",
    "is_weekend": "whether the transaction fell on a Saturday or Sunday",
    "is_off_hours": "whether the transaction happened between 2 AM and 5 AM",
    "is_card_present": "card-present vs card-not-present at the merchant terminal",
    "country_mismatch_customer": "transaction country differs from the cardholder's home country",
    "country_mismatch_merchant": "transaction country differs from the merchant's home country",
    "tx_count_1h": "number of transactions from this customer in the past hour",
    "tx_count_24h": "number of transactions from this customer in the past 24 hours",
    "log_amount_sum_24h": "total spend in the past 24 hours on a log scale",
    "customer_account_age_days": "how long the cardholder has been a customer",
    "customer_risk_tier_encoded": "the cardholder's pre-assigned risk tier (LOW/MEDIUM/HIGH)",
    "avg_amount_30d": "the customer's average transaction amount over the past 30 days",
    "amount_zscore_30d": "how many std-devs above/below this customer's recent average the amount is",
    "days_since_last_tx": "days since the customer's previous transaction (high = dormancy)",
    "merchant_risk_encoded": "the merchant's category risk tier (LOW/MEDIUM/HIGH)",
    "is_high_risk_category": "whether the merchant category is one of the high-risk categories",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-training analyses")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("ml/artifacts"),
        help="Where model.json, metrics.json, etc. live",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ml/data/synthetic_transactions.csv"),
        help="Synthetic labels CSV",
    )
    parser.add_argument(
        "--card-path",
        type=Path,
        default=Path("ml/MODEL_CARD.md"),
        help="Where to write the regenerated model card",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for fast smoke runs",
    )
    return parser.parse_args()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def _format_segment_row(name: str, block: dict[str, Any]) -> str:
    """Return one markdown table row for the segment table.

    `pr_auc` and `recall_at_1pct_fpr` are rendered as `—` when null so a
    reader can't mistake "skipped because undefined" for "computed and equals
    zero".
    """
    pr_auc = block.get("pr_auc")
    rec1 = block.get("recall_at_1pct_fpr")
    n_neg = block.get("n_negatives", block["n_transactions"] - block["n_frauds"])
    pr_auc_str = f"{pr_auc:.4f}" if pr_auc is not None else "—"
    rec1_str = f"{rec1:.4f}" if rec1 is not None else "—"
    return (
        f"| {name} | {block['n_transactions']} | {block['n_frauds']} | "
        f"{n_neg} | {block['fraud_rate']:.4f} | {pr_auc_str} | {rec1_str} |"
    )


def _segment_commentary(segment_metrics: dict[str, Any]) -> str:
    """Honest one-paragraph note about per-segment behaviour.

    Distinguishes "underperforming but measured" from "unmeasurable" — the
    latter happens when a synthetic bucket contains only fraud-injected
    transactions and therefore has no negatives, which makes PR-AUC and
    Recall@FPR mathematically undefined. We must not claim "no segment falls
    materially behind" when in fact one segment was never measured.
    """
    global_pr = segment_metrics["global"].get("pr_auc")
    weak: list[tuple[str, float]] = []
    skipped_no_neg: list[str] = []
    skipped_few_pos: list[str] = []

    for name, block in segment_metrics["segments"].items():
        reason = block.get("skipped_reason")
        if reason:
            if "no negative" in reason:
                skipped_no_neg.append(name)
            else:
                skipped_few_pos.append(name)
            continue
        seg_pr = block.get("pr_auc")
        if seg_pr is None or global_pr is None:
            continue
        if seg_pr < global_pr - 0.05:
            weak.append((name, float(seg_pr)))

    notes: list[str] = []
    if weak:
        joined = ", ".join(f"{n} (PR-AUC {v:.4f})" for n, v in weak)
        notes.append(
            f"Underperforming segments relative to the global PR-AUC of "
            f"{global_pr:.4f}: {joined}."
        )
    if skipped_no_neg:
        notes.append(
            f"Unmeasurable on this synthetic dataset: {', '.join(skipped_no_neg)}. "
            "Those country codes appear only in fraud-injected transactions "
            "by design of the synthetic generator, so PR-AUC and Recall@FPR "
            "have no defined value on those buckets — no claim about model "
            "behaviour on real high-fraud geographies can be made from these "
            "numbers."
        )
    if skipped_few_pos:
        notes.append(
            "Segments with too few positive examples for stable metrics: "
            + ", ".join(skipped_few_pos)
            + "."
        )
    if not notes:
        # Reached only when every segment was measurable AND none was weak.
        # Avoid the absolute "everything is fine across geographies" framing;
        # state the bound that was actually checked.
        notes.append(
            f"All measurable segments fall within 0.05 PR-AUC of the global "
            f"score of {global_pr:.4f}."
        )
    return " ".join(notes)


def _calibration_commentary(calibration_metrics: dict[str, Any]) -> str:
    """Honest paragraph for the calibration section.

    Quotes the dominant low-prediction bin's share and the worst
    positive-containing bin's predicted/observed pair so the over-prediction
    is named explicitly with real numbers, not invoked abstractly.
    """
    total = int(calibration_metrics["n_test_samples"])
    bin_counts = list(calibration_metrics["bin_counts"])
    dominant_idx = int(np.argmax(bin_counts)) if bin_counts else 0
    dominant_count = int(bin_counts[dominant_idx]) if bin_counts else 0
    dominant_pct = (dominant_count / total * 100) if total else 0.0

    worst = calibration_metrics.get("worst_positive_bin")
    if worst is None:
        worst_clause = (
            "No bin in the test set contains both a positive and a measurable "
            "calibration gap, so the over-prediction cannot be illustrated "
            "with a worst-case bin on this run."
        )
    else:
        worst_clause = (
            f"bin {worst['bin_index']} predicts "
            f"{worst['mean_predicted']:.2f} fraud probability where only "
            f"{worst['mean_observed'] * 100:.0f}% of those transactions are "
            "actually fraud"
        )

    return (
        f"Aggregate Brier and ECE look strong because **{dominant_pct:.1f}%** "
        f"of test samples land in bin {dominant_idx} and are correctly assigned "
        "near-zero fraud probability. The positive-restricted variants strip "
        "out this dominant well-calibrated negative class and surface the "
        f"systematic over-prediction on harder cases — {worst_clause}. "
        "Production deployment would apply Platt or isotonic calibration as a "
        "post-hoc step before using the probabilities for thresholded decisions."
    )


def _format_top_features_table(feature_importance: dict[str, Any]) -> str:
    """Return a 5-row markdown table for the model-card feature section."""
    lines = [
        "| Rank | Feature | Mean abs SHAP | Interpretation |",
        "|---|---|---|---|",
    ]
    for entry in feature_importance["features"][:5]:
        interp = _FEATURE_INTERPRETATIONS.get(entry["feature"], "—")
        lines.append(
            f"| {entry['rank']} | `{entry['feature']}` | "
            f"{entry['mean_abs_shap']:.4f} | {interp} |"
        )
    return "\n".join(lines)


def _build_model_card(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    segment_metrics: dict[str, Any],
    calibration_metrics: dict[str, Any],
    feature_importance: dict[str, Any],
) -> str:
    """Compose the full MODEL_CARD.md content as a string."""
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    libs = metadata.get("library_versions", {})
    op = metrics["at_operating_threshold"]
    segment_rows = "\n".join(
        _format_segment_row(name, segment_metrics["segments"][name])
        for name in segment_metrics["segments"]
    )

    return f"""# Model card — Fraud Radar XGBoost classifier

> Auto-regenerated from `backend/ml/analyze.py` using the artifacts in
> `backend/ml/artifacts/`. Re-run after every training run; do not hand-edit.
> Last generated: `{generated_at}`.

## 1. Model details

- **Name:** Fraud Radar XGBoost classifier
- **Version:** trained at `{metadata.get("trained_at_utc", "unknown")}`
- **Type:** gradient-boosted decision tree ensemble (binary classifier)
- **Training framework:** XGBoost {libs.get("xgboost", "?")}, scikit-learn {libs.get("scikit-learn", "?")}, Python {libs.get("python", "?")}
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

{metadata.get("dataset_size", "—"):,} synthetic transactions across 500 customers and
200 merchants. Train fraud rate `{metadata.get("train_fraud_rate", 0):.4f}`,
validation `{metadata.get("val_fraud_rate", 0):.4f}`, test
`{metadata.get("test_fraud_rate", 0):.4f}`. Six fraud patterns are injected
by [`backend/ml/synthesis/`](synthesis/): card testing, geo-velocity, account
takeover, amount anomaly, off-hours, and merchant concentration. Fully
reproducible from `seed=42`.

Splits are strictly chronological by `created_at`: oldest 70% is train, next
15% val, last 15% test (`{metadata.get("test_size", "—")}` rows). No row sees
its own future.

## 4. Performance — overall

| Metric | Value |
|---|---|
| PR-AUC | {metrics["test_pr_auc"]:.4f} |
| ROC-AUC | {metrics["test_roc_auc"]:.4f} |
| Recall @ 1% FPR | {metrics["recall_at_1pct_fpr"]:.4f} |
| Recall @ 5% FPR | {metrics["recall_at_5pct_fpr"]:.4f} |
| Operating threshold | {op["threshold"]:.4f} |
| Precision @ threshold | {op["precision"]:.4f} |
| Recall @ threshold | {op["recall"]:.4f} |
| F1 @ threshold | {op["f1"]:.4f} |

Source: `artifacts/metrics.json`.

## 5. Performance — by segment

Geographic segmentation. Buckets are mutually exclusive and collectively
exhaustive. **Note:** this is performance stability, not demographic
fairness — the synthetic dataset has no protected attributes.

| Segment | n | n_frauds | n_neg | fraud_rate | PR-AUC | Recall@1%FPR |
|---|---|---|---|---|---|---|
{segment_rows}

{_segment_commentary(segment_metrics)}

Source: `artifacts/segment_metrics.json`.

## 6. Calibration

| Metric | Aggregate | Positives only |
|---|---|---|
| Brier score | {calibration_metrics["brier_score"]:.4f} | {calibration_metrics["positive_class_brier"]:.4f} |
| Expected calibration error | {calibration_metrics["expected_calibration_error"]:.4f} | {calibration_metrics["positive_class_ece"]:.4f} |

{_calibration_commentary(calibration_metrics)}

See [`artifacts/calibration_curve.png`](artifacts/calibration_curve.png).

## 7. Feature importance

Top 5 features by mean absolute SHAP value on the test set:

{_format_top_features_table(feature_importance)}

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
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    artifact_dir: Path = args.artifact_dir
    log.info("Reading existing artifacts from %s", artifact_dir.resolve())

    with (artifact_dir / "metrics.json").open(encoding="utf-8") as f:
        metrics = json.load(f)
    with (artifact_dir / "training_metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)

    log.info("Loading dataset and feature matrix (this takes a while)...")
    with SessionLocal() as db:
        dataset = load_dataset_with_csv_labels(
            db, csv_path=str(args.csv_path), limit=args.limit
        )
    log.info("Loaded %d rows; running chronological split", dataset.n_rows)

    splits = chronological_split(dataset.timestamps)
    X_test = dataset.X[splits.test]
    y_test = dataset.y[splits.test]
    test_ids = [dataset.transaction_ids[i] for i in splits.test]
    log.info("Test fold: %d rows, %d frauds", len(y_test), int(y_test.sum()))

    # Initialise the same explainer the inference API uses; this loads
    # model.json + threshold.json + feature_list.json under the hood.
    explainer = initialize_explainer(artifact_dir)

    log.info("Scoring test fold...")
    dmat = xgb.DMatrix(X_test, feature_names=FEATURE_NAMES)
    y_score = np.asarray(explainer._booster.predict(dmat))  # noqa: SLF001  reuse loaded booster
    if y_score.ndim == 2:
        y_score = y_score[:, 1] if y_score.shape[1] > 1 else y_score.ravel()

    # ----- Recover the country column from the synthetic CSV -----
    # The feature matrix doesn't carry the raw country string; pull it from
    # the same CSV the labels came from so segment routing is consistent.
    log.info("Loading country column from %s", args.csv_path)
    countries_by_id = _load_country_column(args.csv_path)
    test_countries = [countries_by_id[tid] for tid in test_ids]

    # ----- Segment metrics -----
    log.info("Computing segment metrics...")
    segment_metrics = compute_segment_metrics(y_test, y_score, test_countries)
    _json_dump(artifact_dir / "segment_metrics.json", segment_metrics)

    # ----- Calibration -----
    log.info("Computing calibration metrics...")
    calibration_metrics = compute_calibration_metrics(y_test, y_score)
    _json_dump(artifact_dir / "calibration_metrics.json", calibration_metrics)
    render_calibration_plot(calibration_metrics, artifact_dir / "calibration_curve.png")

    # ----- Global SHAP & feature importance -----
    log.info("Computing global SHAP values on test fold...")
    shap_matrix = explainer.compute_global_shap(X_test)
    feature_importance = compute_feature_importance(shap_matrix, FEATURE_NAMES)
    _json_dump(artifact_dir / "feature_importance.json", feature_importance)
    render_beeswarm_plot(
        shap_matrix, X_test, FEATURE_NAMES, artifact_dir / "global_shap_beeswarm.png"
    )
    render_bar_plot(
        shap_matrix, X_test, FEATURE_NAMES, artifact_dir / "global_shap_bar.png"
    )

    # ----- Model card -----
    log.info("Writing model card to %s", args.card_path)
    card_text = _build_model_card(
        metadata=metadata,
        metrics=metrics,
        segment_metrics=segment_metrics,
        calibration_metrics=calibration_metrics,
        feature_importance=feature_importance,
    )
    args.card_path.parent.mkdir(parents=True, exist_ok=True)
    args.card_path.write_text(card_text, encoding="utf-8")

    _print_summary(metrics, segment_metrics, calibration_metrics, feature_importance)


def _load_country_column(csv_path: Path) -> dict[str, str]:
    """Read just (id, country) from the synthetic CSV — keeps memory small."""
    import csv

    mapping: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["id"]] = row["country"]
    return mapping


def _print_summary(
    metrics: dict[str, Any],
    segments: dict[str, Any],
    calibration: dict[str, Any],
    feature_importance: dict[str, Any],
) -> None:
    """One-screen headline summary so the developer doesn't have to open JSON."""
    log.info("=" * 64)
    log.info("Fraud Radar — post-training analysis summary")
    log.info("=" * 64)
    log.info("Overall    PR-AUC=%.4f  Recall@1%%FPR=%.4f",
             metrics["test_pr_auc"], metrics["recall_at_1pct_fpr"])
    log.info(
        "Calibration  aggregate Brier=%.4f  ECE=%.4f  |  positives-only Brier=%.4f  ECE=%.4f",
        calibration["brier_score"],
        calibration["expected_calibration_error"],
        calibration["positive_class_brier"],
        calibration["positive_class_ece"],
    )
    log.info("Segments:")
    for name, block in segments["segments"].items():
        pr = block.get("pr_auc")
        reason = block.get("skipped_reason")
        if pr is not None:
            pr_str = f"{pr:.4f}"
        elif reason and "no negative" in reason:
            pr_str = "skipped (no negatives)"
        elif reason:
            pr_str = "skipped (too few positives)"
        else:
            pr_str = "skipped"
        log.info(
            "  %-12s  n=%-6d  n_frauds=%-4d  n_neg=%-4d  PR-AUC=%s",
            name,
            block["n_transactions"],
            block["n_frauds"],
            block.get("n_negatives", block["n_transactions"] - block["n_frauds"]),
            pr_str,
        )
    log.info("Top 5 features by mean |SHAP|:")
    for entry in feature_importance["features"][:5]:
        log.info("  %-30s  %.4f", entry["feature"], entry["mean_abs_shap"])
    log.info("=" * 64)


if __name__ == "__main__":
    main()
