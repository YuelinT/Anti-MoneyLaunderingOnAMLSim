# XGBoost + graph features: first run

Both models use the same AMLSim train/validation/test graphs, account labels, 17 base features, XGBoost hyperparameters, class weight, validation early stopping, and validation-selected F1 threshold rule. Model 3 adds nine features computed separately within each graph from ordinary transaction sender, receiver, and timestamp columns. Its score files use the group's `split,acct_id,is_sar,risk_score` format.

| Test metric | XGB-Base | XGB-Graph |
|---|---:|---:|
| Average Precision (AP) | 0.8879 | 0.9648 |
| SAR Precision | 0.9683 | 0.9761 |
| SAR Recall | 0.7888 | 0.8793 |
| SAR F1 | 0.8694 | 0.9252 |
| Recall@Top-1% | 0.1724 (40/232) | 0.1724 (40/232) |

The graph model's threshold was 0.6303, selected on validation. On test it found 204 of 232 SAR accounts with 5 false positives (TP 204, FP 5, FN 28, TN 3,763). Both models fill the 40-account Top-1% budget with SAR accounts, so this metric is at its theoretical ceiling and cannot separate them on this dataset.

The added features measure distinct incoming/outgoing partners, reciprocal partners, two-hop reach, short-window fan-in/fan-out, and directed three-node cycles. The three-cycle feature is nearly always zero in this generated dataset, including for SAR accounts in test. Thus the result supports the value of the **combined added graph feature set**, but does not establish that explicit cycle detection caused the gain. A degree-only ablation or typology-specific analysis would be needed for that narrower claim.

This is one fixed synthetic dataset and one model seed. No test-driven hyperparameter search or independent repeat was performed. The same 180-day window is observed for all models, so the task is retrospective account detection rather than online prediction.
