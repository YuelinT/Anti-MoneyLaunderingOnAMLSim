#!/usr/bin/env python3
"""XGBoost-Graph on the same splits and model settings as XGBoost-Base."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier

from build_graph_features import GRAPH_FEATURES, build_all
from build_motif_features import MOTIF_FEATURES, build_all as build_motifs
from run_xgboost_baseline import BASE_FEATURES, choose_f1_threshold, evaluate, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", type=Path, default=root.parents[2])
    parser.add_argument("--with-motifs", action="store_true", help="add timed fan and 4–10-node cycle features")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = root.parents[2] / "results" / ("xgb_graph_motif" if args.with_motifs else "xgb_graph")

    baseline = json.loads((root.parents[2] / "results" / "xgb_base" / "metrics.json").read_text(encoding="utf-8"))
    if baseline["features"] != BASE_FEATURES:
        raise ValueError("Baseline feature list has changed; freeze the shared protocol first")
    # Older metrics may record a contributor's machine path that is absent here.
    recorded_dataset = Path(baseline["dataset"])
    if recorded_dataset.exists() and recorded_dataset.resolve() != args.dataset.resolve():
        raise ValueError("Graph and baseline dataset paths differ")
    # Confirm the frozen feature list and class weight against this dataset below.
    config = baseline["hyperparameters"]
    graphs = build_all(args.dataset, root.parents[2] / "results" / "feature_tables" / "graph")
    motifs = build_motifs(args.dataset, root.parents[2] / "results" / "feature_tables" / "motif") if args.with_motifs else None
    features = BASE_FEATURES + GRAPH_FEATURES + (MOTIF_FEATURES if args.with_motifs else [])
    data = {}
    for split in ("train", "validation", "test"):
        base = load_split(args.dataset, split).drop(columns=GRAPH_FEATURES[:3])
        graph = graphs[split]
        base["acct_id"] = base["acct_id"].astype(str)
        frame = base.merge(graph, on=["split", "acct_id"], how="inner", validate="one_to_one")
        if motifs is not None:
            frame = frame.merge(motifs[split], on=["split", "acct_id"], how="inner", validate="one_to_one")
        if len(frame) != len(base) or frame[features].isna().any().any():
            raise ValueError(f"{split}: failed to match every account to graph features")
        data[split] = frame

    train, validation, test = (data[split] for split in ("train", "validation", "test"))
    y_train = train["is_sar"].to_numpy(dtype=int)
    y_validation = validation["is_sar"].to_numpy(dtype=int)
    y_test = test["is_sar"].to_numpy(dtype=int)
    expected_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    if not np.isclose(config["scale_pos_weight"], expected_weight):
        raise ValueError("Baseline class weight does not match this training split")
    model = XGBClassifier(**config)
    model.fit(train[features], y_train, eval_set=[(validation[features], y_validation)], verbose=False)
    validation_scores = model.predict_proba(validation[features])[:, 1]
    threshold, validation_best_f1 = choose_f1_threshold(y_validation, validation_scores)
    test_scores = model.predict_proba(test[features])[:, 1]
    metrics = evaluate(y_test, test_scores, threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output / ("xgboost_graph_motif.json" if args.with_motifs else "xgboost_graph.json"))
    pd.DataFrame(
        {"split": "validation", "acct_id": validation["acct_id"], "is_sar": y_validation, "risk_score": validation_scores}
    ).to_csv(args.output / "validation_predictions.csv", index=False)
    top = np.argsort(-test_scores, kind="stable")[:metrics["top_1_percent_k"]]
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
        "model": "XGB-Graph+Motif" if args.with_motifs else "XGB-Graph",
        "dataset": str(args.dataset.resolve()),
        "features": features,
        "graph_feature_definitions": {
            "unique_*counterparties": "distinct incoming/outgoing transaction partners",
            "reciprocal_counterparties": "partners with transactions in both directions",
            "two_hop_*_unique": "distinct accounts reachable in two directed hops, excluding self",
            "directed_3cycle_count": "distinct directed v→u→w→v cycles touching the account",
            "max_*_counterparties_7d": "maximum distinct partners in any 7-day event-anchored window",
        },
        "feature_source": "raw/<split>/transactions.csv: orig_acct, bene_acct, tran_timestamp only",
        "motif_feature_definitions": {
            "fan_in_then_out_14d": "maximum distinct senders within 14 days before an outgoing transfer",
            "in_then_fan_out_14d": "maximum distinct receivers within 14 days after an incoming transfer",
            "directed_cycle_paths_4_10": "simple directed cycles of length 4–10 touching the account",
            "shortest_directed_cycle_4_10": "length of shortest such cycle; zero if none",
        } if args.with_motifs else {},
        "hyperparameter_search": "none; identical XGBoost configuration to XGB-Base",
        "hyperparameters": config,
        "best_iteration": int(model.best_iteration),
        "threshold_selection": "maximize SAR F1 on validation; highest threshold on ties",
        "validation_f1_at_selected_threshold": validation_best_f1,
        "selected_threshold": threshold,
        "test_metrics": metrics,
        "xgboost_version": xgboost.__version__,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
