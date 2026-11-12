# LR-Base：统一 17 特征协议结果

## 实验口径

数据取自本数据包中的 train、validation、test 独立模拟图，随机种子分别为 42、43、44。每份数据有 4,000 个账户；一个账户为一个样本，`is_sar=1` 为正类。观察完整 180 天，属于回顾性账户检测。不同图中相同的 `acct_id` 是不同账户。

LR-Base 使用以下且仅以下 17 项数值特征：`is_individual`、`initial_deposit`、`total_tx_count`、`incoming_tx_count`、`outgoing_tx_count`、`total_in_amount`、`total_out_amount`、`mean_in_amount`、`mean_out_amount`、`std_in_amount`、`std_out_amount`、`max_in_amount`、`max_out_amount`、`net_flow`、`active_days`、`activity_span_days`、`transactions_per_active_day`。`split`、`acct_id` 只用于识别，`is_sar` 只用于监督和评估；SAR/alert 字段和名单没有参与建模。数据表里额外存在的三个对手方数量列未输入该模型。

先在 train 拟合 `StandardScaler` 和带 `class_weight="balanced"` 的逻辑回归（`solver="lbfgs"`、`max_iter=2000`、`random_state=42`）。在 validation 比较 `C ∈ {0.01, 0.1, 1, 10}` 的 AP，选出 `C=10`；然后选择使 validation SAR F1 最高的风险分数阈值 `0.8575020453051394`。配置与阈值固定后评价 test。测试集没有用于调参。

## 报告主表

| 数据 | 账户数 | SAR 账户数（比例） | AP | SAR Precision | SAR Recall | SAR F1 | Recall@Top-1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 4,000 | 211（5.275%） | 0.7692 | 0.9139 | 0.6540 | 0.7624 | 0.1896 |
| test | 4,000 | 232（5.8%） | 0.8015 | 0.8798 | 0.6940 | 0.7759 | 0.1724 |

测试集 Top-1% 为按风险分数排序的前 40 个账户，其中 40 个为 SAR，因此 Recall@Top-1% = 40 / 232 = 0.1724。固定阈值下的测试集混淆矩阵：TP=161、FP=22、TN=3,746、FN=71；模型标记 183 个账户。

## 交付文件与复现

- `validation_predictions.csv`、`test_predictions.csv`：逐账户标签、风险分数及固定阈值预测。
- `report_table.csv`：五项主指标和样本数量。
- `metrics.json`：完整精度指标、阈值、特征、搜索范围和随机种子。
- `recall_at_k.csv`：不同调查预算下的召回率。
- `coefficients.csv`、`lr_base_model.joblib`：标准化后系数和保存的模型；仅加载可信来源的模型文件。

从数据集根目录运行：

```bash
python3 -m pip install -r requirements-lr.txt
python3 scripts/run_lr_baseline.py
```

本次输入和评价流程符合组内统一协议，没有模型结构导致的口径差异。旧版 `../lr_base/` 使用了额外的对手方数量特征，只留作追溯，不作为统一比较结果。此次仅有一套合成图实验；结果不能说明多随机种子稳定性或真实银行场景的效果。
