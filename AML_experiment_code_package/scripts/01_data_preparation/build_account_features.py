#!/usr/bin/env python3
"""Build leakage-safe account-level tables from AMLSim split folders."""

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def mean(total, count):
    return total / count if count else 0.0


def stddev(total, total_sq, count):
    if not count:
        return 0.0
    variance = max(total_sq / count - (total / count) ** 2, 0.0)
    return math.sqrt(variance)


def build_split(split):
    source = ROOT / "raw" / split
    destination = ROOT / "model_input" / (split + "_account_features.csv")

    with (source / "accounts.csv").open(newline="", encoding="utf-8") as handle:
        accounts = list(csv.DictReader(handle))
    with (source / "alert_accounts.csv").open(newline="", encoding="utf-8") as handle:
        positive_accounts = {
            row["acct_id"]
            for row in csv.DictReader(handle)
            if row["is_sar"].strip().lower() == "true"
        }

    stats = defaultdict(
        lambda: {
            "in_count": 0,
            "out_count": 0,
            "in_sum": 0.0,
            "out_sum": 0.0,
            "in_sum_sq": 0.0,
            "out_sum_sq": 0.0,
            "in_max": 0.0,
            "out_max": 0.0,
            "in_partners": set(),
            "out_partners": set(),
            "days": set(),
            "first_time": None,
            "last_time": None,
        }
    )

    with (source / "transactions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Deliberately ignore transaction-level is_sar and alert_id columns.
            orig = row["orig_acct"]
            bene = row["bene_acct"]
            amount = float(row["base_amt"])
            timestamp = parse_time(row["tran_timestamp"])

            out_s = stats[orig]
            out_s["out_count"] += 1
            out_s["out_sum"] += amount
            out_s["out_sum_sq"] += amount * amount
            out_s["out_max"] = max(out_s["out_max"], amount)
            out_s["out_partners"].add(bene)

            in_s = stats[bene]
            in_s["in_count"] += 1
            in_s["in_sum"] += amount
            in_s["in_sum_sq"] += amount * amount
            in_s["in_max"] = max(in_s["in_max"], amount)
            in_s["in_partners"].add(orig)

            for account_stats in (out_s, in_s):
                account_stats["days"].add(timestamp.date())
                if account_stats["first_time"] is None or timestamp < account_stats["first_time"]:
                    account_stats["first_time"] = timestamp
                if account_stats["last_time"] is None or timestamp > account_stats["last_time"]:
                    account_stats["last_time"] = timestamp

    fieldnames = [
        "split",
        "acct_id",
        "is_individual",
        "initial_deposit",
        "total_tx_count",
        "incoming_tx_count",
        "outgoing_tx_count",
        "unique_counterparties",
        "unique_in_counterparties",
        "unique_out_counterparties",
        "total_in_amount",
        "total_out_amount",
        "mean_in_amount",
        "mean_out_amount",
        "std_in_amount",
        "std_out_amount",
        "max_in_amount",
        "max_out_amount",
        "net_flow",
        "active_days",
        "activity_span_days",
        "transactions_per_active_day",
        "is_sar",
    ]

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for account in accounts:
            account_id = account["acct_id"]
            s = stats[account_id]
            total_count = s["in_count"] + s["out_count"]
            active_days = len(s["days"])
            if s["first_time"] is None:
                span_days = 0
            else:
                span_days = (s["last_time"].date() - s["first_time"].date()).days + 1
            all_partners = s["in_partners"] | s["out_partners"]
            writer.writerow(
                {
                    "split": split,
                    "acct_id": account_id,
                    "is_individual": int(account["type"] == "I"),
                    "initial_deposit": account["initial_deposit"],
                    "total_tx_count": total_count,
                    "incoming_tx_count": s["in_count"],
                    "outgoing_tx_count": s["out_count"],
                    "unique_counterparties": len(all_partners),
                    "unique_in_counterparties": len(s["in_partners"]),
                    "unique_out_counterparties": len(s["out_partners"]),
                    "total_in_amount": round(s["in_sum"], 2),
                    "total_out_amount": round(s["out_sum"], 2),
                    "mean_in_amount": round(mean(s["in_sum"], s["in_count"]), 6),
                    "mean_out_amount": round(mean(s["out_sum"], s["out_count"]), 6),
                    "std_in_amount": round(stddev(s["in_sum"], s["in_sum_sq"], s["in_count"]), 6),
                    "std_out_amount": round(stddev(s["out_sum"], s["out_sum_sq"], s["out_count"]), 6),
                    "max_in_amount": round(s["in_max"], 2),
                    "max_out_amount": round(s["out_max"], 2),
                    "net_flow": round(s["in_sum"] - s["out_sum"], 2),
                    "active_days": active_days,
                    "activity_span_days": span_days,
                    "transactions_per_active_day": round(mean(total_count, active_days), 6),
                    "is_sar": int(account_id in positive_accounts),
                }
            )
    return destination


if __name__ == "__main__":
    for split_name in SPLITS:
        output = build_split(split_name)
        print(output.relative_to(ROOT))
