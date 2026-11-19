# 01 · 数据准备

- `build_account_features.py`：读取 `raw/<split>/`，输出 `model_input/<split>_account_features.csv`。标签只从模拟器的 SAR 账户名单形成；交易级 `is_sar` 和 `alert_id` 不进入特征。
- `validate_dataset.py`：核对账户、标签、交易 ID、金额和交易双方，并更新包根目录的汇总、校验报告与 SHA256 清单。

两个脚本从包根目录运行，详细命令见根目录 `README_CN.md`。
