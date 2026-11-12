# GINE-Full：Apple Silicon 本机实验

## 运行方式与数据口径

在本机 Apple Silicon 上使用 PyTorch 2.14.1、PyTorch Geometric 2.8.1 和 MPS 后端完成训练。train、validation、test 是随机种子 42、43、44 生成的独立图；每图 4,000 个账户。任务为完整 180 天的回顾性账户级 SAR 检测。

节点仅使用组内统一的 17 项基础特征。边取自各图自己的 `transactions.csv`，只读取发起账户、收款账户、金额和时间。边属性是用 train 标准化的 `log1p(金额)`、在 180 天窗口中的日期位置，以及原交易方向标记。每笔原交易附加一条带相反方向标记的反向消息边，使收付款双方都能聚合信息；这不表示多生成了一笔真实交易。`is_sar`、`alert_id`、警报类型及名单都没有进入节点或边特征。节点和金额标准化参数仅从 train 拟合。

模型为两层 GINEConv、隐藏维度 32、dropout 0.25。使用 Adam，学习率 0.001，weight decay 0.0001；正类权重由 train 的 232 个 SAR 与 3,768 个非 SAR 账户计算。模型随机种子 42。预设一种适合本机的小模型配置；验证集 AP 选择最佳轮次，最多 300 轮，连续 20 轮无改善时早停。共训练 240 轮，最佳为第 220 轮。验证集 F1 选出的固定阈值是 `0.9760556817054749`。

## 五项主指标

| 数据 | 账户数 | SAR 账户数（比例） | AP | SAR Precision | SAR Recall | SAR F1 | Recall@Top-1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 4,000 | 211（5.275%） | 0.8305 | 0.9732 | 0.6872 | 0.8056 | 0.1896 |
| test | 4,000 | 232（5.8%） | 0.8851 | 0.9733 | 0.7845 | 0.8687 | 0.1724 |

测试集前 40 个高风险账户均为 SAR，因此 Recall@Top-1% = 40 / 232。固定阈值下，TP=182、FP=5、TN=3,763、FN=50。相同 17 项节点基础输入的 LR-Base 测试集 AP 为 0.8015、F1 为 0.7759；GINE 还使用了交易图和边属性，不能把性能差异单独归因于网络架构。

## 复现与文件

在项目根目录执行本机现成环境：

```bash
work/gine_venv/bin/python deliverables/amlsim_course_dataset_v1/scripts/run_gine.py --device mps --smoke
work/gine_venv/bin/python deliverables/amlsim_course_dataset_v1/scripts/run_gine.py --device mps
```

若 MPS 不可用，可改为 `--device cpu`；脚本的 `--device auto` 会自动选择。新机器可按数据集根目录的 `README.md` 和 `requirements-gine.txt` 建立环境。

`report_table.csv` 是报告主表，`metrics.json` 记录完整配置和混淆矩阵，`training_history.csv` 记录每轮验证 AP，`validation_predictions.csv` 与 `test_predictions.csv` 含逐账户风险分数，`gine_model.pt` 为可用 `torch.load(..., weights_only=True)` 加载的模型。保存模型在 CPU 上重算测试分数，与 MPS 保存分数最大差值约 0.000003，AP 相同。

这次先做过 60 轮试跑并查看其测试指标；由于当时验证 AP 至第 60 轮仍在上升，仅据验证曲线把训练上限扩到 300 后重训。最终配置的选择依据是验证集，但测试集已被查看过，因此它不是严格意义上的首次盲测。结果也只代表这一组合成数据与一次模型随机种子，尚不能说明多次运行的波动或真实银行效果。
