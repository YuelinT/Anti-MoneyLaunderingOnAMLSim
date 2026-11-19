"""Compare XGBoost misses by injected AMLSim test pattern."""
import csv
import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2]
output = root / "results" / "xgb_pattern_analysis"
data = root
models = {
    "base": root / "results/xgb_base",
    "graph": root / "results/xgb_graph",
}

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

output.mkdir(parents=True, exist_ok=True)
accounts = read_csv(data / "model_input/test_account_features.csv")
truth = {r["acct_id"]: int(r["is_sar"]) for r in accounts}
assert len(accounts) == len(truth) == 4000
positive = {account for account, label in truth.items() if label == 1}
patterns = defaultdict(set)
for row in read_csv(data / "raw/test/alert_accounts.csv"):
    if row["is_sar"].strip().lower() == "true":
        patterns[row["acct_id"]].add(row["alert_type"])
assert set(patterns) == positive
assert all(len(types) == 1 and next(iter(types)) in {"fan_in", "fan_out", "cycle"}
           for types in patterns.values())

predictions, ranks, thresholds, cutoff_ties = {}, {}, {}, {}
for name, directory in models.items():
    rows = read_csv(directory / "test_predictions.csv")
    mapping = {r["acct_id"]: r for r in rows}
    assert len(rows) == len(mapping) == len(truth) and set(mapping) == set(truth)
    assert all(r["split"] == "test" and int(r["is_sar"]) == truth[r["acct_id"]] for r in rows)
    threshold = json.loads((directory / "metrics.json").read_text())["selected_threshold"]
    assert all(int(r["predicted_sar"]) == int(float(r["risk_score"]) >= threshold) for r in rows)
    ordered = sorted(rows, key=lambda r: -float(r["risk_score"]))
    ranks[name] = {r["acct_id"]: i for i, r in enumerate(ordered, start=1)}
    cutoff_ties[name] = float(ordered[399]["risk_score"]) == float(ordered[400]["risk_score"])
    predictions[name], thresholds[name] = mapping, threshold

details = []
for account in sorted(positive, key=int):
    row = {"acct_id": account, "pattern": next(iter(patterns[account]))}
    for name in models:
        prediction = predictions[name][account]
        row[f"{name}_score"] = prediction["risk_score"]
        row[f"{name}_rank"] = ranks[name][account]
        row[f"{name}_top400_detected"] = int(ranks[name][account] <= 400)
        row[f"{name}_threshold_detected"] = int(prediction["predicted_sar"])
    for measure in ("top400", "threshold"):
        base = row[f"base_{measure}_detected"]
        graph = row[f"graph_{measure}_detected"]
        row[f"{measure}_comparison"] = (
            "both_detected" if base and graph else
            "graph_only" if graph else
            "base_only" if base else
            "missed_by_both"
        )
    details.append(row)

fields = list(details[0])
write_csv(output / "sar_account_comparison.csv", details, fields)
for measure in ("top400", "threshold"):
    for model in models:
        misses = [r for r in details if not r[f"{model}_{measure}_detected"]]
        write_csv(output / f"{model}_missed_{measure}.csv", misses, fields)

summary = []
for measure in ("top400", "threshold"):
    for pattern in ("fan_in", "fan_out", "cycle", "all"):
        subset = [r for r in details if pattern == "all" or r["pattern"] == pattern]
        total = len(subset)
        row = {"measure": measure, "pattern": pattern, "sar_accounts": total}
        for model in models:
            hits = sum(r[f"{model}_{measure}_detected"] for r in subset)
            row[f"{model}_detected"] = hits
            row[f"{model}_missed"] = total - hits
            row[f"{model}_recall"] = hits / total
        for label in ("both_detected", "graph_only", "base_only", "missed_by_both"):
            row[label] = sum(r[f"{measure}_comparison"] == label for r in subset)
        row["net_gain"] = row["graph_detected"] - row["base_detected"]
        assert row["net_gain"] == row["graph_only"] - row["base_only"]
        summary.append(row)
write_csv(output / "summary_by_pattern.csv", summary, list(summary[0]))
print("Thresholds:", thresholds, "Top-400 cutoff ties:", cutoff_ties)
for row in summary:
    print(row)
