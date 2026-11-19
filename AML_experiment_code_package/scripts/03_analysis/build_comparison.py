#!/usr/bin/env python3
"""Merge course-model results and draw the test Recall@Top-10% figure."""

import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "comparison"
LOCAL = {
    "Rule-Based": ("results/rule_based",),
    "XGB-Base": ("results/xgb_base",),
    "XGB-Graph": ("results/xgb_graph",),
    "XGB-Graph+Motif": ("results/xgb_graph_motif",),
}
TEAMMATE_RESULTS = {
    "LR-Base": ROOT / "results/lr_base_17_features/metrics.json",
    "GINE-Full": ROOT / "results/gine_17_features/metrics.json",
}
TABLE_ORDER = ("Rule-Based", "LR-Base", "XGB-Base", "XGB-Graph", "XGB-Graph+Motif", "GINE-Full")


def local_result(name, folder, labels):
    base = ROOT / folder
    report = json.loads((base / "metrics.json").read_text())
    predictions = pd.read_csv(base / "test_predictions.csv")
    needed = {"split", "acct_id", "is_sar", "risk_score"}
    assert needed <= set(predictions.columns)
    assert len(predictions) == 4000 and predictions["acct_id"].is_unique
    assert predictions["split"].eq("test").all()
    current_labels = predictions.set_index("acct_id")["is_sar"].sort_index()
    if labels is not None:
        assert current_labels.equals(labels), f"Label mismatch: {name}"
    ranked = predictions.sort_values("risk_score", ascending=False, kind="stable")
    k = math.ceil(len(ranked) * .10)
    found = int(ranked.head(k)["is_sar"].sum())
    m = report["test_metrics"]
    assert m["n_accounts"] == len(ranked) and m["n_sar"] == int(current_labels.sum())
    return {
        "model": name, "ap": m["average_precision"], "sar_precision": m["sar_precision"],
        "sar_recall": m["sar_recall"], "sar_f1": m["sar_f1"],
        "top_10_k": k, "sar_found_top_10": found,
        "recall_at_top_10_percent": found / m["n_sar"],
        "source": str(base.relative_to(ROOT)),
    }, current_labels


def svg_chart(rows):
    ordered = sorted(rows, key=lambda row: row["recall_at_top_10_percent"], reverse=True)
    width, height = 1200, 646
    left, plot_width, top, step, bar_height = 244, 680, 145, 66, 30
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#162235} .title{font-size:28px;font-weight:700} .sub{font-size:16px;fill:#526071} .model{font-size:18px;font-weight:600} .tick{font-size:14px;fill:#697586} .value{font-size:17px;font-weight:700} .note{font-size:13px;fill:#697586}</style>',
           '<text class="title" x="42" y="49">Recall@Top-10% — test set</text>',
           '<text class="sub" x="42" y="79">Review budget: 400 of 4,000 accounts · 232 SAR-labelled accounts</text>']
    axis_bottom = top + (len(ordered) - 1) * step + bar_height + 10
    for fraction in (0, .25, .50, .75, 1):
        x = left + fraction * plot_width
        svg.append(f'<line x1="{x:.1f}" y1="{top-15}" x2="{x:.1f}" y2="{axis_bottom}" stroke="#e4e9ef"/>')
        svg.append(f'<text class="tick" x="{x:.1f}" y="{axis_bottom+25}" text-anchor="middle">{fraction:.0%}</text>')
    for index, row in enumerate(ordered):
        y = top + index * step
        value = row["recall_at_top_10_percent"]
        bar_width = value * plot_width
        name = row["model"] + ("†" if "screenshot" in row["source"] else "")
        color = "#295f9a" if row["model"].startswith("XGB") else ("#7655a3" if row["model"].startswith("GINE") else "#8b9baa")
        svg.append(f'<text class="model" x="{left-16}" y="{y+23}" text-anchor="end">{escape(name)}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" rx="4" fill="{color}"/>')
        svg.append(f'<text class="value" x="{left+bar_width+12:.1f}" y="{y+23}">{row["sar_found_top_10"]}/232 ({value:.1%})</text>')
    svg.append('<text class="note" x="42" y="586">LR/GINE: metrics from saved JSON; Top-10% from full_recall_at_k.csv.</text>')
    svg.append('<text class="note" x="42" y="609">Rule-Based has a score tie at rank 400; stable account order gives 58 hits (possible tie range: 54–62).</text>')
    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def main():
    rows, labels = [], None
    for name, (folder,) in LOCAL.items():
        row, labels = local_result(name, folder, labels)
        rows.append(row)
    assert int(labels.sum()) == 232
    teammate = pd.read_csv(ROOT / "results/comparison/full_recall_at_k.csv")
    top_ten = teammate.loc[teammate["split"].eq("test") & teammate["k"].eq(400)]
    assert len(top_ten) == 1 and int(top_ten.iloc[0]["sar_total"]) == 232
    for name, metrics_path in TEAMMATE_RESULTS.items():
        recorded = json.loads(metrics_path.read_text(encoding="utf-8"))["test"]
        ap, precision, recall, f1 = (recorded[key] for key in
                                     ("average_precision", "sar_precision", "sar_recall", "sar_f1"))
        found = int(top_ten.iloc[0]["lr_sar_found" if name == "LR-Base" else "gine_sar_found"])
        rows.append({"model": name, "ap": ap, "sar_precision": precision,
                     "sar_recall": recall, "sar_f1": f1, "top_10_k": 400,
                     "sar_found_top_10": found, "recall_at_top_10_percent": found / 232,
                     "source": str(metrics_path.relative_to(ROOT)) + "; Top-10%: full_recall_at_k.csv"})
    rows.sort(key=lambda row: TABLE_ORDER.index(row["model"]))
    OUT.mkdir(exist_ok=True)
    with (OUT / "comparison_test.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ["# 六个模型的测试集结果", "", "测试集：4,000 个账户，其中 232 个 SAR 标注账户。Recall@Top-10% 统一按风险分数取前 400 个账户；与分类阈值无关。", "", "| 模型 | AP | SAR Precision | SAR Recall | SAR F1 | Recall@Top-10% | 检出数 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        mark = "†" if "screenshot" in row["source"] else ""
        table.append(f'| {row["model"]}{mark} | {row["ap"]:.4f} | {row["sar_precision"]:.4f} | {row["sar_recall"]:.4f} | {row["sar_f1"]:.4f} | {row["recall_at_top_10_percent"]:.4f} | {row["sar_found_top_10"]}/232 |')
    table += ["", "LR/GINE 的 AP 与分类指标来自保存的 metrics.json；Recall@Top-10% 与完整曲线来自 full_recall_at_k.csv。其余四行从本地预测文件重算 Top-10%，AP/Precision/Recall/F1 读取对应 metrics.json。", "", "人工规则在第 400 名存在同分：第 400 名同分组共有 242 个账户，其中 11 个 SAR；高于该分数的 392 个账户中有 54 个 SAR。固定原始账户顺序取余下 8 个名额，检出 58 个；不同同分顺序下为 54–62 个。", "", "![Recall@K 曲线](recall_at_k_curve.png)", "", "![Recall@Top-10% 柱形图](recall_top_10.png)", ""]
    (OUT / "comparison_test.md").write_text("\n".join(table), encoding="utf-8")
    (OUT / "recall_top_10.svg").write_text(svg_chart(rows), encoding="utf-8")
    print("Wrote", OUT / "comparison_test.csv", OUT / "comparison_test.md", OUT / "recall_top_10.svg")


if __name__ == "__main__":
    main()
