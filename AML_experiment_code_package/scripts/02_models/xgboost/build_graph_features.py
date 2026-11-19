#!/usr/bin/env python3
"""Build label-free graph/motif features from ordinary AMLSim transactions."""

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd


GRAPH_FEATURES = [
    "unique_counterparties",
    "unique_in_counterparties",
    "unique_out_counterparties",
    "reciprocal_counterparties",
    "two_hop_in_unique",
    "two_hop_out_unique",
    "directed_3cycle_count",
    "max_in_counterparties_7d",
    "max_out_counterparties_7d",
]


def max_unique_partners_in_7_days(events: list[tuple[int, str]]) -> int:
    events.sort()
    left = 0
    partners = Counter()
    best = 0
    for day, partner in events:
        partners[partner] += 1
        while day - events[left][0] >= 7:
            old_partner = events[left][1]
            partners[old_partner] -= 1
            if partners[old_partner] == 0:
                del partners[old_partner]
            left += 1
        best = max(best, len(partners))
    return best


def build_split(dataset: Path, split: str) -> pd.DataFrame:
    """Load only sender, receiver and time; exclude SAR/alert columns."""
    source = dataset / "raw" / split / "transactions.csv"
    base = pd.read_csv(dataset / "model_input" / f"{split}_account_features.csv")
    account_ids = base["acct_id"].astype(str).tolist()
    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    in_events = defaultdict(list)
    out_events = defaultdict(list)

    transactions = pd.read_csv(
        source,
        usecols=["orig_acct", "bene_acct", "tran_timestamp"],
        dtype={"orig_acct": str, "bene_acct": str},
    )
    for sender, receiver, timestamp in transactions.itertuples(index=False, name=None):
        day = date.fromisoformat(timestamp[:10]).toordinal()
        outgoing[sender].add(receiver)
        incoming[receiver].add(sender)
        out_events[sender].append((day, receiver))
        in_events[receiver].append((day, sender))

    rows = []
    for account in account_ids:
        ins = incoming[account]
        outs = outgoing[account]
        two_hop_in = set().union(*(incoming[neighbor] for neighbor in ins)) if ins else set()
        two_hop_out = set().union(*(outgoing[neighbor] for neighbor in outs)) if outs else set()
        two_hop_in.discard(account)
        two_hop_out.discard(account)
        cycles = sum(len((outgoing[neighbor] & ins) - {account, neighbor}) for neighbor in outs)
        rows.append(
            {
                "split": split,
                "acct_id": account,
                "unique_counterparties": len(ins | outs),
                "unique_in_counterparties": len(ins),
                "unique_out_counterparties": len(outs),
                "reciprocal_counterparties": len(ins & outs),
                "two_hop_in_unique": len(two_hop_in),
                "two_hop_out_unique": len(two_hop_out),
                "directed_3cycle_count": cycles,
                "max_in_counterparties_7d": max_unique_partners_in_7_days(in_events[account]),
                "max_out_counterparties_7d": max_unique_partners_in_7_days(out_events[account]),
            }
        )
    graph = pd.DataFrame(rows)
    # The three degree features already exist in the supplied table; this check
    # catches graph construction mistakes before they affect the comparison.
    for feature in GRAPH_FEATURES[:3]:
        if not (graph[feature].to_numpy() == base[feature].to_numpy()).all():
            raise ValueError(f"{split}: {feature} does not match the prepared account table")
    return graph


def build_all(dataset: Path, destination: Path) -> dict[str, pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    graphs = {}
    for split in ("train", "validation", "test"):
        graph = build_split(dataset, split)
        graph.to_csv(destination / f"{split}_graph_features.csv", index=False)
        graphs[split] = graph
    return graphs


if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parents[3]
    build_all(dataset_root, Path(__file__).resolve().parents[3] / "results" / "feature_tables" / "graph")
