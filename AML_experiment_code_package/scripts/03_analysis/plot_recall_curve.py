#!/usr/bin/env python3
"""Plot test-set cumulative SAR recall against the top 0–10% review budget."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aml_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "comparison"
PREDICTIONS = {
    "Rule-Based": ROOT / "results/rule_based/test_predictions.csv",
    "XGB-Base": ROOT / "results/xgb_base/test_predictions.csv",
    "XGB-Graph": ROOT / "results/xgb_graph/test_predictions.csv",
    "XGB-Graph+Motif": ROOT / "results/xgb_graph_motif/test_predictions.csv",
}
LR_GINE_CURVE = ROOT / "results/comparison/full_recall_at_k.csv"
COLORS = {
    "Rule-Based": "#465468", "LR-Base": "#C4473D",
    "XGB-Base": "#C78A00", "GINE-Full": "#7849A6",
    "XGB-Graph": "#005A9C", "XGB-Graph+Motif": "#008773",
}
ORDER = ("Rule-Based", "LR-Base", "XGB-Base", "GINE-Full", "XGB-Graph", "XGB-Graph+Motif")


def main():
    rows = []
    reference_labels = None
    for model, path in PREDICTIONS.items():
        frame = pd.read_csv(path)
        assert len(frame) == 4000 and frame["acct_id"].is_unique
        assert frame["split"].eq("test").all()
        labels = frame.set_index("acct_id")["is_sar"].sort_index()
        if reference_labels is not None:
            assert labels.equals(reference_labels), f"Label mismatch for {model}"
        reference_labels = labels
        ranked = frame.sort_values("risk_score", ascending=False, kind="stable")
        found = np.r_[0, ranked["is_sar"].to_numpy(dtype=int)[:400].cumsum()]
        rows.extend((model, k, k / 4000, int(n), 232, n / 232, "prediction file")
                    for k, n in enumerate(found))
    assert int(reference_labels.sum()) == 232
    teammate = pd.read_csv(LR_GINE_CURVE)
    teammate = teammate.loc[teammate["split"].eq("test")].sort_values("k")
    assert len(teammate) == 4000 and np.array_equal(teammate["k"], np.arange(1, 4001))
    assert teammate["sar_total"].eq(232).all()
    assert np.allclose(teammate["top_fraction"], teammate["k"] / 4000)
    for model, prefix in (("LR-Base", "lr"), ("GINE-Full", "gine")):
        found = teammate[f"{prefix}_sar_found"].to_numpy(dtype=int)
        assert np.isin(np.diff(np.r_[0, found]), [0, 1]).all()
        assert np.allclose(teammate[f"{prefix}_recall_at_k"], found / 232)
        assert {"LR-Base": 186, "GINE-Full": 200}[model] == found[399]
        rows.append((model, 0, 0.0, 0, 232, 0.0, "mathematical origin"))
        rows.extend((model, k, k / 4000, int(n), 232, n / 232, "full_recall_at_k.csv")
                    for k, n in enumerate(found[:400], start=1))
    data = pd.DataFrame(rows, columns=["model", "k", "top_fraction", "sar_found", "sar_total", "recall_at_k", "source"])
    assert data.groupby("model")["recall_at_k"].apply(lambda s: s.is_monotonic_increasing).all()
    assert data.loc[data.k.eq(400)].set_index("model").loc["XGB-Graph", "sar_found"] == 224
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "recall_at_k_curve.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.edgecolor": "#aeb9c4", "axes.labelcolor": "#253449",
                         "xtick.color": "#526173", "ytick.color": "#526173"})
    fig, ax = plt.subplots(figsize=(11.6, 7.2), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for model in ORDER:
        current = data.loc[data.model.eq(model)]
        end = current.iloc[-1]["recall_at_k"]
        ax.step(current.top_fraction * 100, current.recall_at_k * 100,
                where="post", label=f"{model}  ({end:.1%})", color=COLORS[model],
                linewidth=3.1 if model.startswith("XGB-Graph") else 2.7,
                linestyle="-", zorder=4 if model.startswith("XGB-Graph") else 3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, .5, 1, 2, 5, 10], ["0%", "0.5%", "1%", "2%", "5%", "10%"])
    ax.set_yticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.set_xlabel("Review budget (% of 4,000 test accounts)", labelpad=12)
    ax.set_ylabel("SAR recall (% of 232 SAR-labelled accounts)", labelpad=12)
    ax.grid(axis="y", color="#dce3ea", linewidth=.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(.52, .89), frameon=False, ncol=3,
               fontsize=10.2, handlelength=2.5, columnspacing=1.4)
    fig.suptitle("Cumulative SAR recall by investigation budget", x=.08, y=.968,
                 ha="left", fontsize=20, fontweight="bold", color="#162235")
    fig.text(.08, .922, "Test set | 4,000 accounts | 232 SAR-labelled accounts",
             fontsize=11, color="#576779")
    fig.subplots_adjust(left=.12, right=.975, top=.79, bottom=.19)
    fig.text(.08, .07, "Rule-Based: tied scores use stable input order; recall at 10% is 58/232 (tie range 54–62).",
             fontsize=9.7, color="#576779")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"recall_at_k_curve.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    print(data.loc[data.k.eq(400), ["model", "sar_found", "recall_at_k"]].to_string(index=False))


if __name__ == "__main__":
    main()
