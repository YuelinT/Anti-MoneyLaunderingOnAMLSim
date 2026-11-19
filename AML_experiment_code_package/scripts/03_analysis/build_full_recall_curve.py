"""Build account-level LR/GINE comparisons and exact Recall@K curves.

Run from any directory:
  python3 scripts/build_full_recall_curve.py --output-dir OUTPUT_DIR

The input predictions are already trained model outputs. This script does not
refit either model or use test labels to select model settings.
"""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "LR": ROOT / "results/lr_base_17_features",
    "GINE": ROOT / "results/gine_17_features",
}


def read_predictions(path, split):
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                raise ValueError(f"Unexpected split in {path}: {row['split']}")
            acct_id = row["acct_id"]
            if acct_id in rows:
                raise ValueError(f"Duplicate acct_id in {path}: {acct_id}")
            label = int(row["is_sar"])
            score = float(row["risk_score"])
            if label not in (0, 1) or not 0 <= score <= 1:
                raise ValueError(f"Invalid label/score for {acct_id} in {path}")
            rows[acct_id] = (label, score)
    return rows


def account_key(acct_id):
    try:
        return (0, int(acct_id))
    except ValueError:
        return (1, acct_id)


def make_curve(rows):
    ordered = sorted(rows.items(), key=lambda item: (-item[1][1], account_key(item[0])))
    sar_total = sum(label for label, _ in rows.values())
    if not sar_total:
        raise ValueError("No SAR accounts")
    found = 0
    output = []
    for k, (acct_id, (label, score)) in enumerate(ordered, 1):
        found += label
        output.append((k, acct_id, score, found, found / sar_total))
    return output, sar_total


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def chart_svg(path, test_rows, sar_total):
    width, height = 1120, 680
    left, right, top, bottom = 95, 35, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(k):
        return left + (k / len(test_rows)) * plot_w

    def y(recall):
        return top + (1 - recall) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="95" y="35" font-family="Arial,sans-serif" font-size="24" font-weight="bold" fill="#162235">测试集 Recall@K：LR vs GINE</text>',
        f'<text x="95" y="57" font-family="Arial,sans-serif" font-size="14" fill="#526276">4,000 个测试账户 · {sar_total} 个 SAR · K=1–4,000 · 按风险分数降序</text>',
    ]
    for pct in range(0, 101, 20):
        yy = y(pct / 100)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e4e9ef"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="13" fill="#526276">{pct}%</text>')
    for k in (0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
        xx = x(k)
        parts.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#edf0f4"/>')
        parts.append(f'<text x="{xx:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#526276">{k:,}</text>')
    parts.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#162235">复核账户数 K</text>')
    parts.append(f'<text transform="translate(25 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" fill="#162235">SAR Recall@K</text>')
    for field, color, label in (("LR", "#2276c7", "LR"), ("GINE", "#d2692d", "GINE")):
        points = " ".join(f'{x(row["k"]):.2f},{y(row[field]):.2f}' for row in test_rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        legend_x = 760 if field == "LR" else 865
        parts.append(f'<line x1="{legend_x}" y1="38" x2="{legend_x+30}" y2="38" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+38}" y="43" font-family="Arial,sans-serif" font-size="15" fill="#162235">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_rows, curve_rows = [], []
    by_split = {}
    for split in ("validation", "test"):
        lr = read_predictions(MODELS["LR"] / f"{split}_predictions.csv", split)
        gine = read_predictions(MODELS["GINE"] / f"{split}_predictions.csv", split)
        if lr.keys() != gine.keys():
            raise ValueError(f"LR/GINE account sets differ in {split}")
        for acct_id in lr:
            if lr[acct_id][0] != gine[acct_id][0]:
                raise ValueError(f"LR/GINE labels differ for {split}/{acct_id}")
        for acct_id in sorted(lr, key=account_key):
            label, lr_score = lr[acct_id]
            _, gine_score = gine[acct_id]
            score_rows.append((split, acct_id, label, lr_score, gine_score))
        lr_curve, sar_total = make_curve(lr)
        gine_curve, gine_total = make_curve(gine)
        if sar_total != gine_total or len(lr_curve) != len(gine_curve):
            raise ValueError(f"Incompatible model outputs for {split}")
        by_split[split] = []
        for lr_point, gine_point in zip(lr_curve, gine_curve):
            k = lr_point[0]
            row = {
                "split": split,
                "k": k,
                "top_fraction": k / len(lr_curve),
                "sar_total": sar_total,
                "LR": lr_point[4],
                "GINE": gine_point[4],
            }
            by_split[split].append(row)
            curve_rows.append((split, k, row["top_fraction"], sar_total,
                               lr_point[3], row["LR"], gine_point[3], row["GINE"]))

    write_csv(args.output_dir / "paired_account_scores.csv",
              ("split", "acct_id", "is_sar", "lr_risk_score", "gine_risk_score"), score_rows)
    write_csv(args.output_dir / "full_recall_at_k.csv",
              ("split", "k", "top_fraction", "sar_total", "lr_sar_found", "lr_recall_at_k", "gine_sar_found", "gine_recall_at_k"), curve_rows)
    chart_svg(args.output_dir / "test_recall_at_k_lr_vs_gine.svg", by_split["test"], by_split["test"][0]["sar_total"])
    for split in ("validation", "test"):
        rows = by_split[split]
        print(split, "accounts", len(rows), "SAR", rows[0]["sar_total"])
        for k in (20, 40, 80, 200, 400, 4000):
            row = rows[k - 1]
            print(f"  K={k}: LR={row['LR']:.6f}, GINE={row['GINE']:.6f}")


if __name__ == "__main__":
    main()
