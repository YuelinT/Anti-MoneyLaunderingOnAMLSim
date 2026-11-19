# 03 · 结果分析

| 脚本 | 内容 | 默认输出 |
| --- | --- | --- |
| `build_full_recall_curve.py` | LR/GINE 的完整 Recall@K | 指定的 `--output-dir` |
| `analyze_sar_misses_by_pattern.py` | LR/GINE 的 SAR 模式漏检分析 | 指定的 `--output-dir` |
| `build_comparison.py` | 合并六组测试结果并画 Top-10% 对比 | `results/comparison/` |
| `plot_recall_curve.py` | 六模型 0–10% 审查预算召回曲线 | `results/comparison/` |
| `run_xgb_pattern_analysis.py` | XGB-Base 与 XGB-Graph 的模式漏检对比 | `results/xgb_pattern_analysis/` |

从代码包根目录运行。六模型分析读取 `results/` 的已保存预测；警报模式只用于事后分组，不进入模型输入。
