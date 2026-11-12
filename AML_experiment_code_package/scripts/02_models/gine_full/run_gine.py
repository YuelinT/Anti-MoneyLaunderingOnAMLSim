#!/usr/bin/env python3
"""Train a small, full-graph GINE for retrospective account-level SAR detection."""

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import sklearn
import torch
import torch_geometric
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.nn import GINEConv


DEFAULT_ROOT = Path(__file__).resolve().parents[3]
BASE_FEATURES = (
    "is_individual", "initial_deposit", "total_tx_count", "incoming_tx_count",
    "outgoing_tx_count", "total_in_amount", "total_out_amount", "mean_in_amount",
    "mean_out_amount", "std_in_amount", "std_out_amount", "max_in_amount",
    "max_out_amount", "net_flow", "active_days", "activity_span_days",
    "transactions_per_active_day",
)
START_DATE = datetime(2017, 1, 1)
SIMULATION_DAYS = 180
TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def read_nodes(root, split):
    path = root / "model_input" / f"{split}_account_features.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(BASE_FEATURES).issubset(reader.fieldnames):
            raise ValueError(f"Missing one or more 17 base features in {path}")
        if not {"split", "acct_id", "is_sar"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing account identifiers or label in {path}")
        rows = list(reader)
    if any(row["split"] != split for row in rows):
        raise ValueError(f"Unexpected split name in {path}")
    ids = [row["acct_id"] for row in rows]
    if len(ids) != 4000 or len(ids) != len(set(ids)):
        raise ValueError(f"Expected 4,000 unique accounts in {path}")
    features = np.asarray([[float(row[name]) for name in BASE_FEATURES] for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["is_sar"]) for row in rows], dtype=np.int64)
    if not np.isfinite(features).all() or set(labels) != {0, 1}:
        raise ValueError(f"Invalid node features or labels in {path}")
    return ids, features, labels


def read_edges(root, split, id_to_index):
    path = root / "raw" / split / "transactions.csv"
    senders, receivers, edge_values = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"orig_acct", "bene_acct", "base_amt", "tran_timestamp"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing transaction columns in {path}")
        for row in reader:
            # is_sar, alert_id, and alert-list files are never read here.
            source = id_to_index[row["orig_acct"]]
            target = id_to_index[row["bene_acct"]]
            amount = float(row["base_amt"])
            timestamp = datetime.fromisoformat(row["tran_timestamp"].replace("Z", "+00:00"))
            day = (timestamp.date() - START_DATE.date()).days
            if amount <= 0 or not 0 <= day < SIMULATION_DAYS:
                raise ValueError(f"Invalid transaction amount or date in {path}")
            senders.append(source)
            receivers.append(target)
            edge_values.append((math.log1p(amount), day / (SIMULATION_DAYS - 1)))
    if not senders:
        raise ValueError(f"No transactions in {path}")
    return np.asarray(senders, dtype=np.int64), np.asarray(receivers, dtype=np.int64), np.asarray(edge_values, dtype=np.float32)


def prepare_graph(root, split, node_scaler, amount_scaler, device):
    ids, node_values, labels = read_nodes(root, split)
    senders, receivers, edge_values = read_edges(root, split, {key: i for i, key in enumerate(ids)})
    # Add a message path in both directions, with +1/-1 encoding the original
    # transaction direction. This lets both endpoints receive information.
    forward = np.stack((senders, receivers))
    reverse = np.stack((receivers, senders))
    edge_index = np.concatenate((forward, reverse), axis=1)
    normalized_amount = amount_scaler.transform(edge_values[:, :1]).astype(np.float32)
    base_attr = np.concatenate((normalized_amount, edge_values[:, 1:2]), axis=1)
    forward_attr = np.concatenate((base_attr, np.ones((len(senders), 1), dtype=np.float32)), axis=1)
    reverse_attr = np.concatenate((base_attr, -np.ones((len(senders), 1), dtype=np.float32)), axis=1)
    edge_attr = np.concatenate((forward_attr, reverse_attr), axis=0)
    graph = {
        "ids": ids,
        "x": torch.from_numpy(node_scaler.transform(node_values).astype(np.float32)).to(device),
        "y": torch.from_numpy(labels.astype(np.float32)).to(device),
        "edge_index": torch.from_numpy(edge_index).to(device),
        "edge_attr": torch.from_numpy(edge_attr).to(device),
        "transactions": len(senders),
    }
    return graph


class AccountGINE(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(len(BASE_FEATURES), hidden), nn.ReLU())
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)),
                    edge_dim=3,
                )
                for _ in range(2)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, graph):
        x = self.encoder(graph["x"])
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, graph["edge_index"], graph["edge_attr"]))
            x = self.dropout(torch.relu(x))
        return self.classifier(x).squeeze(-1)


def predict(model, graph):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(graph)).detach().cpu().numpy().astype(np.float64)


def select_threshold(labels, scores):
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    tp, best_f1, threshold = 0, -1.0, None
    positives = int(labels.sum())
    for index, (score, label) in enumerate(zip(ranked_scores, ranked_labels)):
        tp += int(label)
        if index + 1 < len(scores) and ranked_scores[index + 1] == score:
            continue
        f1 = 2 * tp / (index + 1 + positives)
        if f1 > best_f1 + 1e-12:
            best_f1, threshold = f1, float(score)
    return threshold


def recall_at_top(labels, scores, fraction):
    k = max(1, math.ceil(len(labels) * fraction))
    ranked = np.argsort(-scores, kind="stable")[:k]
    hits = int(labels[ranked].sum())
    return k, hits, hits / int(labels.sum())


def evaluate(labels, scores, threshold):
    predicted = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    k, hits, recall_top = recall_at_top(labels, scores, 0.01)
    return {
        "accounts": int(len(labels)), "sar_accounts": int(labels.sum()),
        "sar_rate": float(labels.mean()), "average_precision": float(average_precision_score(labels, scores)),
        "sar_precision": float(precision), "sar_recall": float(recall), "sar_f1": float(f1),
        "recall_at_top_1_percent": recall_top, "top_1_percent_k": k, "top_1_percent_sar_found": hits,
        "tp": int(((predicted == 1) & (labels == 1)).sum()),
        "fp": int(((predicted == 1) & (labels == 0)).sum()),
        "tn": int(((predicted == 0) & (labels == 0)).sum()),
        "fn": int(((predicted == 0) & (labels == 1)).sum()),
    }


def write_predictions(path, split, ids, labels, scores, threshold):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "acct_id", "is_sar", "risk_score", "predicted_sar"))
        for account_id, label, score in zip(ids, labels, scores):
            writer.writerow((split, account_id, int(label), repr(float(score)), int(score >= threshold)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run two epochs on train/validation only")
    args = parser.parse_args()
    if args.hidden < 1 or args.max_epochs < 1 or args.patience < 1 or not 0 <= args.dropout < 1:
        parser.error("hidden, max-epochs, patience and dropout are out of range")

    root = args.data_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "results" / ("gine_smoke" if args.smoke else "gine_17_features")
    output.mkdir(parents=True, exist_ok=True)
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable in this process; select --device cpu")
    device = torch.device(device_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ids, train_values, _ = read_nodes(root, "train")
    _, _, train_edge_values = read_edges(root, "train", {key: i for i, key in enumerate(train_ids)})
    node_scaler = StandardScaler().fit(train_values)
    amount_scaler = StandardScaler().fit(train_edge_values[:, :1])
    train_graph = prepare_graph(root, "train", node_scaler, amount_scaler, device)
    val_graph = prepare_graph(root, "validation", node_scaler, amount_scaler, device)
    train_y = train_graph["y"].detach().cpu().numpy().astype(np.int64)
    val_y = val_graph["y"].detach().cpu().numpy().astype(np.int64)

    model = AccountGINE(args.hidden, args.dropout).to(device)
    positive_weight = (len(train_y) - int(train_y.sum())) / int(train_y.sum())
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    best_ap, best_epoch, best_state = -1.0, 0, None
    stale_epochs = 0
    start = time.perf_counter()
    max_epochs = 2 if args.smoke else args.max_epochs

    print(f"device={device_name} train_nodes={len(train_y)} train_transactions={train_graph['transactions']} validation_transactions={val_graph['transactions']}", flush=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_graph)
        loss = loss_function(logits, train_graph["y"])
        loss.backward()
        optimizer.step()
        val_scores = predict(model, val_graph)
        val_ap = float(average_precision_score(val_y, val_scores))
        elapsed = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": float(loss.detach().cpu()), "validation_ap": val_ap, "elapsed_seconds": elapsed})
        if args.smoke or epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(f"epoch={epoch:03d} loss={history[-1]['train_loss']:.5f} val_AP={val_ap:.5f} elapsed={elapsed:.1f}s", flush=True)
        if val_ap > best_ap + 1e-4:
            best_ap, best_epoch = val_ap, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if not args.smoke and stale_epochs >= args.patience:
                break

    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "validation_ap", "elapsed_seconds"))
        writer.writeheader()
        writer.writerows(history)

    if args.smoke:
        print(json.dumps({"status": "smoke_passed", "device": device_name, "epochs": len(history), "best_validation_ap": best_ap}, indent=2))
        return

    model.load_state_dict(best_state)
    val_scores = predict(model, val_graph)
    threshold = select_threshold(val_y, val_scores)
    test_graph = prepare_graph(root, "test", node_scaler, amount_scaler, device)
    test_y = test_graph["y"].detach().cpu().numpy().astype(np.int64)
    test_scores = predict(model, test_graph)
    validation = evaluate(val_y, val_scores, threshold)
    test = evaluate(test_y, test_scores, threshold)
    assert test["tp"] + test["fn"] == test["sar_accounts"]
    assert test["tp"] + test["fp"] + test["tn"] + test["fn"] == test["accounts"]

    config = {
        "model": "GINE-Full (2-layer account-node classifier)",
        "protocol": "17 fixed node features; retrospective 180-day account detection",
        "split_seeds": {"train": 42, "validation": 43, "test": 44},
        "model_seed": args.seed,
        "device": device_name,
        "versions": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "scikit_learn": sklearn.__version__},
        "node_features": list(BASE_FEATURES),
        "edge_features": ["train_standardized_log1p_amount", "day_fraction_of_180", "transaction_direction_plus_or_minus_one"],
        "edge_construction": "original directed transaction plus reversed message edge with direction indicator",
        "normalization": "node and log amount StandardScaler fitted on train only",
        "class_weight": {"positive_weight_from_train_only": positive_weight},
        "hyperparameter_search": "one hardware-constrained configuration; validation AP selects epoch",
        "hyperparameters": {"layers": 2, "hidden": args.hidden, "dropout": args.dropout, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "max_epochs": args.max_epochs, "patience": args.patience, "optimizer": "Adam"},
        "best_epoch": best_epoch,
        "trained_epochs": len(history),
        "threshold_selection": "validation SAR F1 maximum; highest score threshold breaks ties",
        "selected_validation_threshold": threshold,
        "train": {"accounts": len(train_y), "sar_accounts": int(train_y.sum()), "transactions": train_graph["transactions"]},
        "validation": validation,
        "test": test,
        "runtime_seconds": time.perf_counter() - start,
        "process_rss_bytes_at_end": psutil.Process().memory_info().rss,
        "mps_current_allocated_bytes_at_end": torch.mps.current_allocated_memory() if device_name == "mps" else None,
    }
    (output / "metrics.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (output / "report_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "accounts", "sar_accounts", "sar_rate", "AP", "SAR_precision", "SAR_recall", "SAR_F1", "Recall_at_Top_1pct", "threshold"))
        for split, metrics in (("validation", validation), ("test", test)):
            writer.writerow((split, metrics["accounts"], metrics["sar_accounts"], metrics["sar_rate"], metrics["average_precision"], metrics["sar_precision"], metrics["sar_recall"], metrics["sar_f1"], metrics["recall_at_top_1_percent"], threshold))
    with (output / "recall_at_k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "top_fraction", "k", "sar_found", "sar_total", "recall_at_k"))
        for split, labels, scores in (("validation", val_y, val_scores), ("test", test_y, test_scores)):
            for fraction in TOP_FRACTIONS:
                k, hits, recall = recall_at_top(labels, scores, fraction)
                writer.writerow((split, fraction, k, hits, int(labels.sum()), recall))
    write_predictions(output / "validation_predictions.csv", "validation", val_graph["ids"], val_y, val_scores, threshold)
    write_predictions(output / "test_predictions.csv", "test", test_graph["ids"], test_y, test_scores, threshold)
    torch.save({"state_dict": best_state, "node_mean": node_scaler.mean_.tolist(), "node_scale": node_scaler.scale_.tolist(), "amount_mean": amount_scaler.mean_.tolist(), "amount_scale": amount_scaler.scale_.tolist(), "config": config["hyperparameters"]}, output / "gine_model.pt")
    print(json.dumps({"device": device_name, "best_epoch": best_epoch, "threshold": threshold, "validation": validation, "test": test, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
