#!/usr/bin/env python3
"""Account-level XGBoost baseline for the supplied AMLSim course splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier


BASE_FEATURES = [
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
]


def load_split(dataset: Path, split: str) -> pd.DataFrame:
    path = dataset / "model_input" / f"{split}_account_features.csv"
    frame = pd.read_csv(path)
    needed = {"split", "acct_id", "is_sar", *BASE_FEATURES}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not frame["split"].eq(split).all():
        raise ValueError(f"{path}: unexpected split values")
    if frame["acct_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate account IDs")
    if frame["is_sar"].isna().any() or not frame["is_sar"].isin([0, 1]).all():
        raise ValueError(f"{path}: invalid labels")
    if frame[BASE_FEATURES].isna().any().any():
        raise ValueError(f"{path}: missing feature values")
    if not np.isfinite(frame[BASE_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite feature values")
    return frame


def choose_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best = np.flatnonzero(np.isclose(f1, f1.max()))
    index = int(best[-1])  # deterministic tie-break: fewer alerts
    return float(thresholds[index]), float(f1[index])


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    k = max(1, int(np.ceil(len(labels) * 0.01)))
    top = np.argsort(-scores, kind="stable")[:k]
    top_positives = int(labels[top].sum())
    return {
        "n_accounts": int(len(labels)),
        "n_sar": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, scores)),
        "sar_precision": precision,
        "sar_recall": recall,
        "sar_f1": f1,
        "recall_at_top_1_percent": top_positives / int(labels.sum()),
        "top_1_percent_k": k,
        "top_1_percent_sar_found": top_positives,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "results" / "xgb_base")
    args = parser.parse_args()
    data = {split: load_split(args.dataset, split) for split in ("train", "validation", "test")}
    train, validation, test = (data[split] for split in ("train", "validation", "test"))
    y_train = train["is_sar"].to_numpy(dtype=int)
    y_validation = validation["is_sar"].to_numpy(dtype=int)
    y_test = test["is_sar"].to_numpy(dtype=int)
    if any(len(np.unique(y)) != 2 for y in (y_train, y_validation, y_test)):
        raise ValueError("Each split must contain both classes")

    positive_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    model_config = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "scale_pos_weight": positive_weight,
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": 4,
    }
    model = XGBClassifier(**model_config)
    model.fit(
        train[BASE_FEATURES],
        y_train,
        eval_set=[(validation[BASE_FEATURES], y_validation)],
        verbose=False,
    )
    validation_scores = model.predict_proba(validation[BASE_FEATURES])[:, 1]
    threshold, validation_best_f1 = choose_f1_threshold(y_validation, validation_scores)
    test_scores = model.predict_proba(test[BASE_FEATURES])[:, 1]
    metrics = evaluate(y_test, test_scores, threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output / "xgboost_baseline.json")
    pd.DataFrame(
        {
            "split": "validation",
            "acct_id": validation["acct_id"],
            "is_sar": y_validation,
            "risk_score": validation_scores,
        }
    ).to_csv(args.output / "validation_predictions.csv", index=False)
    k = metrics["top_1_percent_k"]
    top = np.argsort(-test_scores, kind="stable")[:k]
    in_top = np.zeros(len(test), dtype=int)
    in_top[top] = 1
    pd.DataFrame(
        {
            "split": "test",
            "acct_id": test["acct_id"],
            "is_sar": y_test,
            "risk_score": test_scores,
            "predicted_sar": (test_scores >= threshold).astype(int),
            "in_top_1_percent": in_top,
        }
    ).to_csv(args.output / "test_predictions.csv", index=False)
    report = {
        "model": "XGB-Base (account-level, no counterparty/graph features)",
        "dataset": str(args.dataset.resolve()),
        "features": BASE_FEATURES,
        "label": "is_sar",
        "hyperparameter_search": "none; one pre-specified baseline configuration",
        "hyperparameters": model_config,
        "positive_weight": positive_weight,
        "best_iteration": int(model.best_iteration),
        "threshold_selection": "maximize SAR F1 on validation; highest threshold on ties",
        "validation_f1_at_selected_threshold": validation_best_f1,
        "selected_threshold": threshold,
        "test_metrics": metrics,
        "versions": {
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
