#!/usr/bin/env python3
"""Validate raw and model-ready AMLSim split files and write a summary."""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
FORBIDDEN_FEATURES = {"alert_id", "alert_type", "transaction_is_sar"}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_split(split):
    raw = ROOT / "raw" / split
    accounts = read_rows(raw / "accounts.csv")
    transactions = read_rows(raw / "transactions.csv")
    alert_accounts = read_rows(raw / "alert_accounts.csv")
    features = read_rows(ROOT / "model_input" / (split + "_account_features.csv"))

    account_ids = {row["acct_id"] for row in accounts}
    tx_ids = [row["tran_id"] for row in transactions]
    positive_ids = {
        row["acct_id"]
        for row in alert_accounts
        if row["is_sar"].strip().lower() == "true"
    }
    timestamps = [
        datetime.fromisoformat(row["tran_timestamp"].replace("Z", "+00:00"))
        for row in transactions
    ]

    assert len(accounts) == 4000, split + ": unexpected account count"
    assert len(features) == len(accounts), split + ": feature row count mismatch"
    assert {row["acct_id"] for row in features} == account_ids, split + ": feature account mismatch"
    assert sum(int(row["is_sar"]) for row in features) == len(positive_ids), split + ": label mismatch"
    assert not (FORBIDDEN_FEATURES & set(features[0])), split + ": leakage-prone feature present"
    assert len(tx_ids) == len(set(tx_ids)), split + ": duplicate transaction ids"
    assert all(float(row["base_amt"]) > 0 for row in transactions), split + ": non-positive amount"
    assert all(
        row["orig_acct"] in account_ids and row["bene_acct"] in account_ids
        for row in transactions
    ), split + ": invalid account reference"

    return {
        "split": split,
        "accounts": len(accounts),
        "transactions": len(transactions),
        "positive_accounts": len(positive_ids),
        "positive_rate": round(len(positive_ids) / len(accounts), 6),
        "sar_transactions": sum(row["is_sar"].lower() == "true" for row in transactions),
        "alert_patterns": len({row["alert_id"] for row in alert_accounts}),
        "fan_in_members": Counter(row["alert_type"] for row in alert_accounts)["fan_in"],
        "fan_out_members": Counter(row["alert_type"] for row in alert_accounts)["fan_out"],
        "cycle_members": Counter(row["alert_type"] for row in alert_accounts)["cycle"],
        "start_date": min(timestamps).date().isoformat(),
        "end_date": max(timestamps).date().isoformat(),
    }


def write_checksums():
    lines = []
    for path in sorted(ROOT.rglob("*.csv")) + sorted(ROOT.rglob("*.json")):
        if path.name == "validation_report.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(digest + "  " + str(path.relative_to(ROOT)))
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    summaries = [validate_split(split) for split in SPLITS]
    with (ROOT / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (ROOT / "validation_report.json").write_text(
        json.dumps({"status": "passed", "splits": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_checksums()
    print(json.dumps({"status": "passed", "splits": summaries}, indent=2))
