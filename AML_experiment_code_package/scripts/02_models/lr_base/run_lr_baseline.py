#!/usr/bin/env python3
"""Train and evaluate the account-level logistic-regression (LR-Base) model."""

import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path(__file__).resolve().parents[3]
EXCLUDED_COLUMNS = {"split", "acct_id", "is_sar"}
LEAKAGE_COLUMNS = {"alert_id", "alert_type", "transaction_is_sar", "model_id"}
BASE_FEATURES = (
    "is_individual",
    "initial_deposit",
    "total_tx_count",
    "incoming_tx_count",
    "outgoing_tx_count",
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
)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def read_split(root, split, feature_names=None):
    path = root / "model_input" / (split + "_account_features.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Missing or duplicate column names in " + str(path))
        if not EXCLUDED_COLUMNS.issubset(columns):
            raise ValueError("Required columns missing in " + str(path))
        if LEAKAGE_COLUMNS.intersection(columns):
            raise ValueError("Label-leaking columns present in " + str(path))
        missing = set(BASE_FEATURES) - set(columns)
        if missing:
            raise ValueError("Missing protocol features in " + str(path) + ": " + str(sorted(missing)))
        if feature_names is None:
            feature_names = list(BASE_FEATURES)
        elif feature_names != list(BASE_FEATURES):
            raise ValueError("Feature list must match the 17-feature protocol")

        account_ids = []
        feature_rows = []
        labels = []
        for row in reader:
            if row["split"] != split:
                raise ValueError("Unexpected split value in " + str(path))
            account_ids.append(row["acct_id"])
            feature_rows.append([float(row[name]) for name in feature_names])
            labels.append(int(row["is_sar"]))

    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if not len(y) or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Empty or non-finite features in " + str(path))
    if not set(y).issubset({0, 1}) or len(set(y)) < 2:
        raise ValueError("Labels must contain both binary classes in " + str(path))
    if len(set(account_ids)) != len(account_ids):
        raise ValueError("Duplicate account IDs in " + str(path))
    return account_ids, x, y, feature_names


def fit_model(x_train, y_train, c_value):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def select_threshold(y_true, scores):
    """Choose the highest score cutoff among validation F1 maxima."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    tp = 0
    best_f1 = -1.0
    best_threshold = None
    positives = int(y_true.sum())
    for index, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
        tp += int(label)
        if index + 1 < len(sorted_scores) and sorted_scores[index + 1] == score:
            continue
        predicted = index + 1
        f1 = 2 * tp / (predicted + positives)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_threshold = float(score)
    return best_threshold


def recall_at_top(y_true, scores, fraction):
    k = max(1, math.ceil(len(y_true) * fraction))
    order = np.argsort(-scores, kind="stable")[:k]
    hits = int(y_true[order].sum())
    return {"k": k, "sar_found": hits, "recall": hits / int(y_true.sum())}


def evaluate(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    top_one = recall_at_top(y_true, scores, 0.01)
    return {
        "accounts": int(len(y_true)),
        "sar_accounts": int(y_true.sum()),
        "sar_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, scores)),
        "sar_precision": float(precision),
        "sar_recall": float(recall),
        "sar_f1": float(f1),
        "recall_at_top_1_percent": top_one["recall"],
        "top_1_percent_k": top_one["k"],
        "top_1_percent_sar_found": top_one["sar_found"],
        "tp": int(((predicted == 1) & (y_true == 1)).sum()),
        "fp": int(((predicted == 1) & (y_true == 0)).sum()),
        "tn": int(((predicted == 0) & (y_true == 0)).sum()),
        "fn": int(((predicted == 0) & (y_true == 1)).sum()),
    }


def write_predictions(path, split, account_ids, labels, scores, threshold):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "acct_id", "is_sar", "risk_score", "predicted_sar"))
        for account_id, label, score in zip(account_ids, labels, scores):
            writer.writerow((split, account_id, int(label), repr(float(score)), int(score >= threshold)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "results" / "lr_base_17_features"
    output.mkdir(parents=True, exist_ok=True)

    train_ids, x_train, y_train, names = read_split(root, "train")
    val_ids, x_val, y_val, _ = read_split(root, "validation", names)
    test_ids, x_test, y_test, _ = read_split(root, "test", names)

    # Select regularization by validation AP; the test graph is not used for selection.
    trials = []
    best_model = None
    best_c = None
    best_ap = -1.0
    for c_value in C_VALUES:
        candidate = fit_model(x_train, y_train, c_value)
        val_scores = candidate.predict_proba(x_val)[:, 1]
        val_ap = float(average_precision_score(y_val, val_scores))
        trials.append({"C": c_value, "validation_average_precision": val_ap})
        if val_ap > best_ap + 1e-12:
            best_model, best_c, best_ap = candidate, c_value, val_ap

    val_scores = best_model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_scores)
    test_scores = best_model.predict_proba(x_test)[:, 1]
    validation = evaluate(y_val, val_scores, threshold)
    test = evaluate(y_test, test_scores, threshold)
    assert test["tp"] + test["fn"] == test["sar_accounts"]
    assert test["tp"] + test["fp"] + test["tn"] + test["fn"] == test["accounts"]

    results = {
        "model": "LR-Base (logistic regression)",
        "protocol": "17 fixed base features; retrospective account classification over full 180 days",
        "split_seeds": {"train": 42, "validation": 43, "test": 44},
        "model_random_state": 42,
        "scikit_learn_version": sklearn.__version__,
        "training_accounts": len(train_ids),
        "training_sar_accounts": int(y_train.sum()),
        "features": names,
        "preprocessing": "StandardScaler fitted on training graph only",
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 2000,
        "hyperparameter_selection": "select C with highest validation Average Precision; first candidate breaks ties",
        "regularization_candidates": trials,
        "selected_C": best_c,
        "threshold_selection": "maximize validation SAR F1; highest threshold breaks ties",
        "selected_validation_threshold": threshold,
        "validation": validation,
        "test": test,
        "top_k_rule": "ceil(fraction * number of accounts); sort by descending risk score",
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output / "report_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "accounts", "sar_accounts", "sar_rate", "AP", "SAR_precision", "SAR_recall", "SAR_F1", "Recall_at_Top_1pct", "threshold"))
        for split, metrics in (("validation", validation), ("test", test)):
            writer.writerow((split, metrics["accounts"], metrics["sar_accounts"], metrics["sar_rate"], metrics["average_precision"], metrics["sar_precision"], metrics["sar_recall"], metrics["sar_f1"], metrics["recall_at_top_1_percent"], threshold))
    with (output / "recall_at_k.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "top_fraction", "k", "sar_found", "sar_total", "recall_at_k"))
        for split, labels, scores in (("validation", y_val, val_scores), ("test", y_test, test_scores)):
            for fraction in TOP_FRACTIONS:
                row = recall_at_top(labels, scores, fraction)
                writer.writerow((split, fraction, row["k"], row["sar_found"], int(labels.sum()), row["recall"]))
    write_predictions(output / "validation_predictions.csv", "validation", val_ids, y_val, val_scores, threshold)
    write_predictions(output / "test_predictions.csv", "test", test_ids, y_test, test_scores, threshold)
    joblib.dump(best_model, output / "lr_base_model.joblib")
    with (output / "coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature", "coefficient_per_training_standard_deviation"))
        for name, coefficient in zip(names, best_model.named_steps["classifier"].coef_[0]):
            writer.writerow((name, float(coefficient)))

    print(json.dumps({"selected_C": best_c, "threshold": threshold, "validation": validation, "test": test, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
