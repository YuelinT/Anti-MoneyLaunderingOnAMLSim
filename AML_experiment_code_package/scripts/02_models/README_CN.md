# 02 · 模型代码

| 模型 | 脚本 | 保存结果 | 依赖 |
| --- | --- | --- | --- |
| Rule-Based | [`rule_based/run_rule_baseline.py`](rule_based/run_rule_baseline.py) | [`results/rule_based/`](../../results/rule_based/) | [`rule_based/requirements.txt`](rule_based/requirements.txt) |
| LR-Base | [`lr_base/run_lr_baseline.py`](lr_base/run_lr_baseline.py) | [`results/lr_base_17_features/`](../../results/lr_base_17_features/) | [`lr_base/requirements.txt`](lr_base/requirements.txt) |
| XGB-Base | [`xgboost/run_xgboost_baseline.py`](xgboost/run_xgboost_baseline.py) | [`results/xgb_base/`](../../results/xgb_base/) | [`xgboost/requirements.txt`](xgboost/requirements.txt) |
| XGB-Graph | [`xgboost/run_xgboost_graph.py`](xgboost/run_xgboost_graph.py) | [`results/xgb_graph/`](../../results/xgb_graph/) | 同上 |
| XGB-Graph+Motif | `run_xgboost_graph.py --with-motifs` | [`results/xgb_graph_motif/`](../../results/xgb_graph_motif/) | 同上 |
| GINE-Full | [`gine_full/run_gine.py`](gine_full/run_gine.py) | [`results/gine_17_features/`](../../results/gine_17_features/) | [`gine_full/requirements.txt`](gine_full/requirements.txt) |

基础账户特征共 17 项；XGB-Graph 增加 9 项图特征；Motif 再增加 4 项。图特征构建脚本、motif 构建脚本和小型算法测试均在 `xgboost/`。图模型依赖 XGB-Base 的 `metrics.json` 配置，请先运行基础版或保留随包保存的指标文件。

运行规则与 XGBoost 前可分别安装依赖：

```bash
python3 -m pip install -r scripts/02_models/rule_based/requirements.txt
python3 -m pip install -r scripts/02_models/xgboost/requirements.txt
```

训练和重建结果会覆盖同名保存文件。若只想查阅原始实验，请直接浏览 `results/`。XGB 原指标保留贡献者本机的绝对数据路径作为来源记录，运行脚本默认使用当前代码包根目录。
