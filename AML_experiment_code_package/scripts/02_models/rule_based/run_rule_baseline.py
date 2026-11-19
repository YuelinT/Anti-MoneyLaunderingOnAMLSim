#!/usr/bin/env python3
"""Transparent, transaction-statistics AML rule baseline for the course splits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT
OUTPUT = ROOT / "results" / "rule_based"
SPLITS = ("train", "validation", "test")
STAT_COLUMNS = ("max_in_7d", "max_out_7d", "max_matched_amount_14d", "max_total_7d")


def transaction_statistics(split):
    accounts = pd.read_csv(DATASET / "model_input" / f"{split}_account_features.csv")[
        ["split", "acct_id", "is_sar"]
    ]
    transactions = pd.read_csv(
        DATASET / "raw" / split / "transactions.csv",
        usecols=["orig_acct", "bene_acct", "base_amt", "tran_timestamp"],
    )
    dates = pd.to_datetime(transactions["tran_timestamp"], utc=True).dt.floor("D")
    outgoing = pd.DataFrame(
        {"acct_id": transactions["orig_acct"], "day": dates, "in_amt": 0.0,
         "out_amt": transactions["base_amt"], "in_n": 0, "out_n": 1}
    )
    incoming = pd.DataFrame(
        {"acct_id": transactions["bene_acct"], "day": dates, "in_amt": transactions["base_amt"],
         "out_amt": 0.0, "in_n": 1, "out_n": 0}
    )
    daily = pd.concat([incoming, outgoing], ignore_index=True).groupby(
        ["acct_id", "day"], sort=False, as_index=False
    ).sum()
    rows = []
    for acct_id, group in daily.groupby("acct_id", sort=False):
        group = group.sort_values("day")
        day = (group["day"].astype("int64").to_numpy() // 86_400_000_000_000)
        values = group[["in_n", "out_n", "in_amt", "out_amt"]].to_numpy(dtype=float)
        cumulative = np.vstack([np.zeros(4), np.cumsum(values, axis=0)])
        end = np.arange(1, len(day) + 1)
        seven = cumulative[end] - cumulative[np.searchsorted(day, day - 6)]
        fourteen = cumulative[end] - cumulative[np.searchsorted(day, day - 13)]
        rows.append((acct_id, seven[:, 0].max(), seven[:, 1].max(),
                     np.minimum(fourteen[:, 2], fourteen[:, 3]).max(),
                     (seven[:, 0] + seven[:, 1]).max()))
    statistics = pd.DataFrame(rows, columns=["acct_id", *STAT_COLUMNS])
    result = accounts.merge(statistics, on="acct_id", how="left", validate="one_to_one")
    result[list(STAT_COLUMNS)] = result[list(STAT_COLUMNS)].fillna(0)
    if result[list(STAT_COLUMNS)].isna().any().any() or result["acct_id"].duplicated().any():
        raise ValueError(f"Invalid account statistics for {split}")
    if not result["split"].eq(split).all() or not result["is_sar"].isin([0, 1]).all():
        raise ValueError(f"Invalid split or labels for {split}")
    return result


def risk_score(frame, limits):
    # Four fixed, equal-weight rules. The small strength term resolves ranking ties.
    ratios = np.column_stack([frame[column].to_numpy() / limits[column] for column in STAT_COLUMNS])
    flags = ratios >= 1
    return flags.sum(axis=1) + 0.1 * np.minimum(ratios, 2).sum(axis=1)


def choose_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(thresholds), where=(precision[:-1] + recall[:-1]) > 0)
    return float(thresholds[np.flatnonzero(np.isclose(f1, f1.max()))[-1]])


def evaluate(labels, scores, threshold):
    predicted = scores >= threshold
    tp = int(((labels == 1) & predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    tn = int(((labels == 0) & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    k = max(1, int(np.ceil(len(labels) * .01)))
    top_found = int(labels[np.argsort(-scores, kind="stable")[:k]].sum())
    return {
        "n_accounts": len(labels), "n_sar": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "sar_precision": precision, "sar_recall": recall,
        "sar_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "recall_at_top_1_percent": top_found / int(labels.sum()),
        "top_1_percent_k": k, "top_1_percent_sar_found": top_found,
        "accuracy": (tp + tn) / len(labels),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main():
    data = {split: transaction_statistics(split) for split in SPLITS}
    # Unlabelled training distribution only; the 90th percentile is an experimental choice.
    limits = {column: max(1.0, float(data["train"][column].quantile(.90)))
              for column in STAT_COLUMNS}
    scores = {split: risk_score(frame, limits) for split, frame in data.items()}
    threshold = choose_threshold(data["validation"]["is_sar"].to_numpy(), scores["validation"])
    metrics = {split: evaluate(data[split]["is_sar"].to_numpy(), scores[split], threshold)
               for split in ("validation", "test")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        frame = data[split][["split", "acct_id", "is_sar"]].copy()
        frame["risk_score"] = scores[split]
        frame["predicted_sar"] = (scores[split] >= threshold).astype(int)
        frame.to_csv(OUTPUT / f"{split}_predictions.csv", index=False)
    report = {
        "model": "M1 Rule-Based (transaction statistics only)",
        "dataset": str(DATASET), "rule_statistics": list(STAT_COLUMNS),
        "rule_limits": limits, "rule_limit_method": "90th percentile of unlabelled train accounts",
        "risk_score": "number of fired rules + 0.1 * sum(min(statistic / train_limit, 2))",
        "threshold_selection": "maximize SAR F1 on validation; highest threshold on ties",
        "selected_threshold": threshold, "validation_metrics": metrics["validation"],
        "test_metrics": metrics["test"],
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
