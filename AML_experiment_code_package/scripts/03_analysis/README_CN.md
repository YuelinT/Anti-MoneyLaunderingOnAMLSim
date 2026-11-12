# 03 · 结果分析

- `build_full_recall_curve.py`：从保存的 LR/GINE 风险分数重算 Recall@K，输出逐账户对照和曲线数据。
- `analyze_sar_misses_by_pattern.py`：将测试集预测与注入的 fan-in、fan-out、cycle 模式连接，检查漏检账户。模式字段只用于事后分析。

两项脚本都需要 `--output-dir` 指定输出目录；详细命令见包根目录 `README_CN.md`。
