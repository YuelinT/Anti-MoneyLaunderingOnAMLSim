"""Post-hoc test-set SAR error analysis by AMLSim injection pattern.

This reads alert membership only after predictions are fixed. Pattern and alert
IDs are never supplied to the LR or GINE models.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATTERNS = ("fan_in", "fan_out", "cycle")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def keyed(rows, path):
    result = {}
    for row in rows:
        key = row["acct_id"]
        if key in result:
            raise ValueError(f"Duplicate acct_id {key} in {path}")
        result[key] = row
    return result


def ranks(predictions):
    # Same stable descending-score order as the model scripts; CSV order is
    # the original account-input order, used to break exact score ties.
    ordered = sorted(predictions, key=lambda row: -float(row["risk_score"]))
    return {row["acct_id"]: rank for rank, row in enumerate(ordered, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = ROOT / "raw/test"
    inputs = ROOT / "model_input/test_account_features.csv"
    lr_path = ROOT / "results/lr_base_17_features/test_predictions.csv"
    gine_path = ROOT / "results/gine_17_features/test_predictions.csv"
    lr_rows = read_csv(lr_path)
    gine_rows = read_csv(gine_path)
    lr = keyed(lr_rows, lr_path)
    gine = keyed(gine_rows, gine_path)
    features = keyed(read_csv(inputs), inputs)
    if len(lr) != len(gine) or set(lr) != set(gine) or set(lr) != set(features):
        raise ValueError("Test account sets do not match")
    if len(lr) != 4000:
        raise ValueError("Expected exactly 4,000 test accounts")

    alert_rows = read_csv(raw / "alert_accounts.csv")
    patterns = {}
    for row in alert_rows:
        acct = row["acct_id"]
        if row["alert_type"] not in PATTERNS or row["is_sar"].lower() != "true":
            raise ValueError(f"Unexpected alert metadata for account {acct}")
        if acct in patterns:
            raise ValueError(f"Overlapping/duplicate alert membership for account {acct}")
        patterns[acct] = (row["alert_type"], row["alert_id"])
    sar_file = {row["ACCOUNT_ID"] for row in read_csv(raw / "sar_accounts.csv") if row["IS_SAR"] == "YES"}
    labelled_sar = {acct for acct, row in lr.items() if int(row["is_sar"]) == 1}
    if len(labelled_sar) != 232 or labelled_sar != set(patterns) or labelled_sar != sar_file:
        raise ValueError("Pattern membership, raw SAR list, and model labels disagree")

    lr_rank = ranks(lr_rows)
    gine_rank = ranks(gine_rows)
    records = []
    for acct in sorted(labelled_sar, key=int):
        l, g = lr[acct], gine[acct]
        if any(row["split"] != "test" for row in (l, g)) or int(l["is_sar"]) != int(g["is_sar"]):
            raise ValueError(f"Split/label disagreement for account {acct}")
        lp, gp = int(l["predicted_sar"]), int(g["predicted_sar"])
        category = ("both_detected" if lp and gp else
                    "lr_only_detected" if lp else
                    "gine_only_detected" if gp else "both_missed")
        pattern, alert_id = patterns[acct]
        feature = features[acct]
        records.append({
            "split": "test", "acct_id": acct, "is_sar": 1,
            "pattern": pattern, "alert_id_analysis_only": alert_id,
            "unique_counterparties": int(feature["unique_counterparties"]),
            "unique_in_counterparties": int(feature["unique_in_counterparties"]),
            "unique_out_counterparties": int(feature["unique_out_counterparties"]),
            "total_tx_count": int(feature["total_tx_count"]),
            "lr_risk_score": float(l["risk_score"]),
            "lr_predicted_sar": lp, "lr_rank": lr_rank[acct],
            "lr_top_40": int(lr_rank[acct] <= 40),
            "lr_top_400": int(lr_rank[acct] <= 400),
            "gine_risk_score": float(g["risk_score"]),
            "gine_predicted_sar": gp, "gine_rank": gine_rank[acct],
            "gine_top_40": int(gine_rank[acct] <= 40),
            "gine_top_400": int(gine_rank[acct] <= 400),
            "comparison": category,
        })

    fields = list(records[0])
    write_csv(out / "all_sar_account_comparison.csv", fields, records)
    missed = [row for row in records if row["comparison"] != "both_detected"]
    write_csv(out / "missed_sar_accounts.csv", fields, missed)

    summary = []
    for pattern in PATTERNS:
        subset = [row for row in records if row["pattern"] == pattern]
        n = len(subset)
        lr_detected = sum(row["lr_predicted_sar"] for row in subset)
        gine_detected = sum(row["gine_predicted_sar"] for row in subset)
        count = Counter(row["comparison"] for row in subset)
        summary.append({
            "pattern": pattern, "sar_accounts": n,
            "lr_detected_at_threshold": lr_detected,
            "lr_missed_at_threshold": n - lr_detected,
            "lr_recall_at_threshold": lr_detected / n,
            "gine_detected_at_threshold": gine_detected,
            "gine_missed_at_threshold": n - gine_detected,
            "gine_recall_at_threshold": gine_detected / n,
            "both_missed": count["both_missed"],
            "lr_only_missed": count["gine_only_detected"],
            "gine_only_missed": count["lr_only_detected"],
            "lr_found_top_40": sum(row["lr_top_40"] for row in subset),
            "gine_found_top_40": sum(row["gine_top_40"] for row in subset),
            "lr_found_top_400": sum(row["lr_top_400"] for row in subset),
            "gine_found_top_400": sum(row["gine_top_400"] for row in subset),
        })
    write_csv(out / "pattern_summary.csv", list(summary[0]), summary)
    totals = Counter(row["comparison"] for row in records)
    report = {
        "scope": "test split only; 4,000 accounts, 232 SAR accounts",
        "pattern_source": "raw/test/alert_accounts.csv; post-hoc analysis only",
        "classification_rule": "predicted_sar from each model's test_predictions.csv, using its validation-selected threshold",
        "lr_threshold": json.loads((ROOT / "results/lr_base_17_features/metrics.json").read_text())["selected_validation_threshold"],
        "gine_threshold": json.loads((ROOT / "results/gine_17_features/metrics.json").read_text())["selected_validation_threshold"],
        "pattern_counts": dict(Counter(row["pattern"] for row in records)),
        "comparison_counts": dict(totals),
        "rank_tie_rule": "descending score, stable original prediction-file order",
    }
    (out / "analysis_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "comparison_counts": dict(totals), "missed_rows": len(missed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
