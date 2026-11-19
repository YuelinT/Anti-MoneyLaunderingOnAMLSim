# XGBoost baseline: verified protocol rerun

Data: AMLSim course dataset v1; independent train/validation/test graphs with 4,000 accounts each. The test graph has 232 SAR-positive accounts (5.8%). The model uses the 17 `BASE_FEATURES` listed in `run_xgboost_baseline.py`, excluding counterparty and motif features.

| Metric | Test result |
|---|---:|
| Average Precision (AP) | 0.8879 |
| SAR Precision | 0.9683 |
| SAR Recall | 0.7888 |
| SAR F1 | 0.8694 |
| Recall@Top-1% | 0.1724 (40 / 232) |

The F1 threshold (0.7027) was selected on the validation graph and fixed for test. At that threshold, the test confusion matrix is TP 183, FP 6, FN 49, TN 3,762. The 40 highest-scoring test accounts are all SAR-positive. Because Top-1% contains only 40 accounts, 0.1724 is the *maximum possible* Recall@Top-1% for this test set; this metric may not distinguish strong models here.

This is one run with one fixed dataset configuration and XGBoost seed 42. It is a result on fully synthetic retrospective data, not a claim about real bank performance. Reproduce it with `python run_xgboost_baseline.py`; the exact output, feature list, package versions, saved model, and account scores are in this directory.

On 2026-10-27, the same pre-specified model was rerun under the shared experiment protocol. The five test metrics, validation-selected threshold, and best iteration matched the earlier run exactly. The rerun also exported `validation_predictions.csv` and added `split` to `test_predictions.csv`; both files contain the common `split,acct_id,is_sar,risk_score` fields. This is a reproducibility check on the **same** generated graphs, not an additional independent dataset or a second statistical replicate. No hyperparameter search was performed.
