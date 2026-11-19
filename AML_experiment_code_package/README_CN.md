# AMLSim 六组模型实验代码包

此目录按数据生成、特征处理、模型训练、结果分析分类。根目录的 [项目 README](../README.md) 介绍实验设计和六组结果。数据均为 AMLSim 合成数据。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`configs/`](configs/) 和 [`parameters/`](parameters/) | 三次独立 AMLSim 运行的配置与共享参数，随机种子为 42、43、44 |
| [`raw/`](raw/) | train、validation、test 三个交易图的公开版导出；账户表已去除身份字段 |
| [`scripts/01_data_preparation/`](scripts/01_data_preparation/) | 账户特征构建与数据校验 |
| [`model_input/`](model_input/) | 每个账户一行的建模表 |
| [`scripts/02_models/`](scripts/02_models/) | 六组模型的运行脚本与各自依赖 |
| [`scripts/03_analysis/`](scripts/03_analysis/) | Recall@K、六模型比较和模式错误分析 |
| [`results/`](results/) | 六组保存结果、逐账户预测、特征表及分析图表 |
| [`validation_report.json`](validation_report.json)、[`SHA256SUMS.txt`](SHA256SUMS.txt) | 数据核对与包内文件校验 |

## 运行顺序

在此目录执行以下命令。各模型依赖不同，请按 [`模型说明`](scripts/02_models/README_CN.md) 分别安装。

```bash
python3 scripts/01_data_preparation/build_account_features.py
python3 scripts/01_data_preparation/validate_dataset.py
python3 scripts/02_models/rule_based/run_rule_baseline.py
python3 scripts/02_models/lr_base/run_lr_baseline.py
python3 scripts/02_models/xgboost/run_xgboost_baseline.py
python3 scripts/02_models/xgboost/run_xgboost_graph.py
python3 scripts/02_models/xgboost/run_xgboost_graph.py --with-motifs
python3 scripts/02_models/gine_full/run_gine.py --device auto
```

前三项 XGBoost 依次运行：图与 motif 版本读取 XGB-Base 保存的模型配置。上述命令默认覆盖 `results/` 中的原有结果；需要保留原始结果时，先复制本包，或给支持输出参数的脚本指定新目录。GINE 完整运行可能耗时较长。

```bash
python3 scripts/03_analysis/build_comparison.py
python3 scripts/03_analysis/plot_recall_curve.py
python3 scripts/03_analysis/run_xgb_pattern_analysis.py
```

**公开数据处理：** `raw/<split>/accounts.csv` 仅保留构建特征需要的账户 ID、类型和初始存款；账户姓名、地址、出生日期、证件号、地理坐标等身份字段已移除。警报和 SAR 表中未用于建模的名称/客户 ID 字段也已移除。交易、标签和实验指标保持原样。

**数据生成边界：** 此包保存 AMLSim 输入配置和已导出的三套图，但不包含上游模拟器源码。`build_account_features.py` 从现有 `raw/` 重建账户特征；从头重建交易图需要单独安装相应 AMLSim 版本并运行三份配置。不同 split 的相同 `acct_id` 不代表同一账户。

**结果边界：** `results/` 中的文件是原实验保存的证据。XGB 和规则指标中的 `dataset` 字段保留了贡献者机器的绝对路径，仅作为来源记录；本包脚本改用当前包路径。包装过程核对了账户、标签与 Top-400 检出数，但没有对六组模型全部重新训练。
