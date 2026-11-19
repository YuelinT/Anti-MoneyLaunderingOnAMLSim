#!/usr/bin/env python3
"""Label-free AML motifs: timed fan patterns and directed cycles of 4–10 nodes."""

from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd


MOTIF_FEATURES = [
    "fan_in_then_out_14d",
    "in_then_fan_out_14d",
    "directed_cycle_paths_4_10",
    "shortest_directed_cycle_4_10",
]


def timed_fan_in_then_out(in_events, out_events):
    """Most distinct prior senders within 14 days before an outgoing transfer."""
    return max(
        (
            len({sender for day, sender in in_events if 0 <= out_day - day <= 14 and sender != receiver})
            for out_day, receiver in out_events
        ),
        default=0,
    )


def timed_in_then_fan_out(in_events, out_events):
    """Most distinct later receivers within 14 days after an incoming transfer."""
    return max(
        (
            len({receiver for day, receiver in out_events if 0 <= day - in_day <= 14 and receiver != sender})
            for in_day, sender in in_events
        ),
        default=0,
    )


def directed_cycles_4_10(start: str, outgoing: dict[str, set[str]]) -> tuple[int, int]:
    """Count simple directed cycles containing start; return 0 length if absent."""
    count = 0
    shortest = 0
    stack = [(start, (start,))]
    while stack:
        node, path = stack.pop()
        for neighbor in outgoing[node]:
            if neighbor == start:
                if 4 <= len(path) <= 10:
                    count += 1
                    shortest = len(path) if shortest == 0 else min(shortest, len(path))
            elif neighbor not in path and len(path) < 10:
                stack.append((neighbor, path + (neighbor,)))
    return count, shortest


def build_split(dataset: Path, split: str) -> pd.DataFrame:
    source = dataset / "raw" / split / "transactions.csv"
    base = pd.read_csv(dataset / "model_input" / f"{split}_account_features.csv")
    transactions = pd.read_csv(
        source,
        usecols=["orig_acct", "bene_acct", "tran_timestamp"],
        dtype={"orig_acct": str, "bene_acct": str},
    )
    outgoing = defaultdict(set)
    in_events = defaultdict(list)
    out_events = defaultdict(list)
    for sender, receiver, timestamp in transactions.itertuples(index=False, name=None):
        day = date.fromisoformat(timestamp[:10]).toordinal()
        outgoing[sender].add(receiver)
        in_events[receiver].append((day, sender))
        out_events[sender].append((day, receiver))

    rows = []
    for account in base["acct_id"].astype(str):
        cycles, shortest = directed_cycles_4_10(account, outgoing)
        rows.append(
            {
                "split": split,
                "acct_id": account,
                "fan_in_then_out_14d": timed_fan_in_then_out(in_events[account], out_events[account]),
                "in_then_fan_out_14d": timed_in_then_fan_out(in_events[account], out_events[account]),
                "directed_cycle_paths_4_10": cycles,
                "shortest_directed_cycle_4_10": shortest,
            }
        )
    return pd.DataFrame(rows)


def build_all(dataset: Path, destination: Path) -> dict[str, pd.DataFrame]:
    destination.mkdir(parents=True, exist_ok=True)
    motifs = {}
    for split in ("train", "validation", "test"):
        motif = build_split(dataset, split)
        motif.to_csv(destination / f"{split}_motif_features.csv", index=False)
        motifs[split] = motif
    return motifs


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build_all(root.parents[2], root.parents[2] / "results" / "feature_tables" / "motif")
