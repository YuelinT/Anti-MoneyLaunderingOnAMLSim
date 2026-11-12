# AMLSim 实验代码包

本包按实验步骤整理，可从配置、模拟输出、特征处理、模型训练到结果分析逐层查看。数据为 AMLSim 生成的合成记录，不是真实账户或监管 SAR 申报。

## 目录导航

| 位置 | 内容 | 用途 |
| --- | --- | --- |
| `configs/`、`parameters/` | train、validation、test 的 AMLSim 配置和共享参数 | 记录三个独立模拟图的生成设置；种子分别为 42、43、44 |
| `raw/<split>/` | 账户、交易、警报和 SAR 导出 | 模拟器的原始输出；`split` 为 `train`、`validation` 或 `test` |
| `scripts/01_data_preparation/` | 账户特征构建、数据校验 | 从 `raw/` 生成 `model_input/` 并核对数据 |
| `model_input/` | 每个 split 一份账户特征 CSV | 每行对应一个账户；含标签 `is_sar` |
| `scripts/02_models/lr_base/` | LR-Base 训练脚本及依赖 | 17 项基础特征的线性基线 |
| `scripts/02_models/gine_full/` | GINE-Full 训练脚本及依赖 | 账户节点及带属性的交易边 |
| `results/lr_base_17_features/`、`results/gine_17_features/` | 已保存的模型、指标、逐账户预测 | 核对报告中的 LR/GINE 结果 |
| `scripts/03_analysis/` | Recall@K 与 SAR 模式错误分析 | 对保存的预测做事后分析 |
| `dataset_summary.csv`、`validation_report.json`、`SHA256SUMS.txt` | 数据规模、校验结果、文件校验值 | 检查交付完整性 |

**尚未包含：**Rule-Based、XGB-Base、XGB-Graph、XGB-Graph+Motif 的原始运行脚本及逐账户预测文件。这些部分留待后续补充；不要把本包理解为六个模型均可重新训练。

## 生成与处理流程

1. AMLSim 使用 `configs/` 中三份配置分别生成 180 天交易图；除随机种子和模拟名称外，配置一致。每个图含 4,000 个账户和 30 个注入的可疑模式。
2. 原始导出保存在 `raw/train/`、`raw/validation/`、`raw/test/`。同一个 `acct_id` 在不同 split 中不是同一个账户。
3. 特征脚本把交易聚合为每账户一行，写入 `model_input/`。基础模型只使用其中指定的 17 项特征；账户 ID、标签和警报字段不得作为输入特征。
4. 训练只使用 train；validation 用来选择配置或训练轮次和分类阈值；test 用于最终报告。

本包保存了 AMLSim 的配置和输出，**没有复制上游 AMLSim 模拟器源码**。以下特征脚本不能重新生成原始交易图；若要从头生成图，需要安装相应版本的 [IBM AMLSim](https://github.com/IBM/AMLSim) 及其依赖，分别运行三份配置。

## 运行已提供的代码

在解压后的包根目录运行：

```bash
python3 scripts/01_data_preparation/build_account_features.py
python3 scripts/01_data_preparation/validate_dataset.py
```

LR 与 GINE 使用不同依赖，建议分别创建 Python 环境。从包根目录运行 LR：

```bash
python3 -m pip install -r scripts/02_models/lr_base/requirements.txt
python3 scripts/02_models/lr_base/run_lr_baseline.py
```

在另一个环境中运行 GINE：

```bash
python3 -m pip install -r scripts/02_models/gine_full/requirements.txt
python3 scripts/02_models/gine_full/run_gine.py --device auto
```

两项训练脚本默认将新结果写入 `results/` 中对应目录。若想保留包内原始结果，请使用脚本的 `--output-dir` 参数指定新的输出位置。GINE 完整训练可能需要较长时间；`--smoke` 可用于快速检查运行环境，但其输出不是报告结果。

事后分析可在预测文件存在时运行：

```bash
python3 scripts/03_analysis/build_full_recall_curve.py --output-dir analysis_output/recall
python3 scripts/03_analysis/analyze_sar_misses_by_pattern.py --output-dir analysis_output/patterns
```

`analyze_sar_misses_by_pattern.py` 使用警报模式字段仅作**事后分组分析**，不把这些字段送入 LR 或 GINE。

## 数据规模与报告口径

| Split | 账户 | 交易 | SAR 账户 |
| --- | ---: | ---: | ---: |
| Train | 4,000 | 38,206 | 232 |
| Validation | 4,000 | 37,790 | 211 |
| Test | 4,000 | 38,301 | 232 |

报告中的 AP、Precision、Recall 和 F1 与 Recall@Top-10% 使用不同口径：前三项分类指标使用验证集选定的阈值；AP 和 Top-400 召回基于风险分数排序。已保存的预测文件可复核 LR/GINE 指标，但并不等于所有六个模型都已在本包重新训练。
