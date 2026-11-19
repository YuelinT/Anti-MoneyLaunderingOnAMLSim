# Exploratory graph + motif extension

Four explicit motif features were added to the existing nine graph features: the largest number of distinct senders in the 14 days before an outgoing transfer, the largest number of distinct receivers in the 14 days after an incoming transfer, the number of simple directed cycles of length 4–10 touching the account, and the shortest such cycle length. They use only ordinary transaction sender, receiver, and timestamp fields. All other XGBoost settings and split rules match the earlier runs.

| Test metric | XGB-Base | XGB-Graph | XGB-Graph+Motif |
|---|---:|---:|---:|
| Average Precision (AP) | 0.8879 | 0.9648 | 0.9657 |
| SAR Precision | 0.9683 | 0.9761 | 0.9758 |
| SAR Recall | 0.7888 | 0.8793 | 0.8707 |
| SAR F1 | 0.8694 | 0.9252 | 0.9203 |
| Recall@Top-1% | 0.1724 | 0.1724 | 0.1724 |

The motif extension increased AP by only 0.0009 relative to XGB-Graph, while F1 fell by 0.0049. Under each model's validation-selected threshold, both models found 77 of 79 test accounts belonging to the injected cycle pattern; the extra cycle feature did not increase cycle recall. Top-1% is saturated at 40/232 for all three models.

The generated graph makes directed cycles a particularly strong synthetic signature: the new 4–10-cycle feature is nonzero for all 79 test cycle-pattern accounts and for zero test non-SAR accounts. It is computed from transactions, not the SAR/alert labels, but this near-perfect alignment is specific to this simulator configuration. The earlier three-node cycle feature was almost always zero and did not describe these injected 5–10-account cycles.

This extension was designed after earlier test results had been inspected. Its test comparison is **exploratory**, not an unbiased confirmation. Keep the original XGB-Graph as the frozen model-3 result unless a new independent test graph is generated and evaluated with the motif design fixed in advance.
