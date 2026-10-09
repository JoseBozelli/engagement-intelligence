# Model comparison (validation)

Synthetic data for methodology evaluation only. The held-out test split was not used.

- modeling config fingerprint: `5399367fbd7ec9ce`
- feature spec fingerprint: `d177b052d82e477d`
- labeling config fingerprint: `64cfdf882d3bef12`
- source run fingerprint: `8918c959fdd5f9fa`

## 1. Tuning (purged walk-forward CV within train)

| candidate | features | selected parameters | CV log loss |
|---|---|---|---|
| baseline | 1 | - | n/a |
| logistic[all] | 19 | C=0.01 | 0.2280 |
| xgboost[all] | 19 | max_depth=5, min_child_weight=50, rounds=128 | 0.2230 |
| logistic[no_policy] | 15 | C=0.01 | 0.2295 |
| xgboost[no_policy] | 15 | max_depth=5, min_child_weight=50, rounds=134 | 0.2233 |
| logistic[level_only] | 15 | C=0.01 | 0.2282 |
| xgboost[level_only] | 15 | max_depth=3, min_child_weight=10, rounds=119 | 0.2261 |

## 2. Validation metrics

| candidate | log loss [95% CI] | PR-AUC [95% CI] | ROC-AUC | Brier | ECE | precision@5% | precision@10% | precision@20% | recall@10% |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2755 [0.2559, 0.2946] | 0.426 [0.380, 0.465] | 0.819 | 0.0789 | 0.0167 | 0.573 | 0.488 | 0.356 | 0.441 |
| logistic[all] | 0.2406 [0.2230, 0.2592] | 0.514 [0.464, 0.559] | 0.873 | 0.0706 | 0.0028 | 0.640 | 0.544 | 0.400 | 0.492 |
| xgboost[all] | 0.2393 [0.2196, 0.2602] | 0.521 [0.462, 0.570] | 0.875 | 0.0700 | 0.0068 | 0.654 | 0.558 | 0.401 | 0.505 |
| logistic[no_policy] | 0.2415 [0.2250, 0.2602] | 0.509 [0.457, 0.553] | 0.872 | 0.0709 | 0.0080 | 0.626 | 0.547 | 0.401 | 0.495 |
| xgboost[no_policy] | 0.2389 [0.2195, 0.2594] | 0.522 [0.463, 0.575] | 0.875 | 0.0698 | 0.0059 | 0.661 | 0.557 | 0.403 | 0.504 |
| logistic[level_only] | 0.2418 [0.2249, 0.2604] | 0.503 [0.453, 0.545] | 0.871 | 0.0710 | 0.0036 | 0.632 | 0.544 | 0.399 | 0.492 |
| xgboost[level_only] | 0.2399 [0.2215, 0.2597] | 0.516 [0.464, 0.560] | 0.873 | 0.0702 | 0.0044 | 0.652 | 0.560 | 0.401 | 0.506 |

## 3. Cold-start vs warm (validation)

| candidate | warm log loss | cold log loss | warm PR-AUC | cold PR-AUC |
|---|---|---|---|---|
| baseline | 0.2779 | 0.2189 | 0.427 | 0.428 |
| logistic[all] | 0.2418 | 0.2109 | 0.518 | 0.372 |
| xgboost[all] | 0.2404 | 0.2144 | 0.525 | 0.395 |
| logistic[no_policy] | 0.2428 | 0.2128 | 0.514 | 0.339 |
| xgboost[no_policy] | 0.2399 | 0.2156 | 0.527 | 0.395 |
| logistic[level_only] | 0.2431 | 0.2119 | 0.507 | 0.368 |
| xgboost[level_only] | 0.2410 | 0.2138 | 0.521 | 0.391 |

## 4. Paired comparisons (log loss difference a - b; negative favors a)

| question | a | b | difference [95% CI] | reading |
|---|---|---|---|---|
| vs recency baseline | logistic[all] | baseline | -0.0349 [-0.0430, -0.0257] | better |
| vs recency baseline | xgboost[all] | baseline | -0.0362 [-0.0460, -0.0259] | better |
| vs recency baseline | logistic[no_policy] | baseline | -0.0340 [-0.0426, -0.0248] | better |
| vs recency baseline | xgboost[no_policy] | baseline | -0.0366 [-0.0465, -0.0265] | better |
| vs recency baseline | logistic[level_only] | baseline | -0.0336 [-0.0421, -0.0246] | better |
| vs recency baseline | xgboost[level_only] | baseline | -0.0356 [-0.0444, -0.0260] | better |
| xgboost vs logistic | xgboost[all] | logistic[all] | -0.0013 [-0.0053, 0.0024] | equivalent |
| xgboost vs logistic | xgboost[no_policy] | logistic[no_policy] | -0.0026 [-0.0071, 0.0017] | equivalent |
| xgboost vs logistic | xgboost[level_only] | logistic[level_only] | -0.0020 [-0.0044, 0.0004] | equivalent |
| no_policy vs all | logistic[no_policy] | logistic[all] | 0.0010 [-0.0005, 0.0023] | equivalent |
| level_only vs all | logistic[level_only] | logistic[all] | 0.0013 [0.0007, 0.0019] | worse |
| no_policy vs all | xgboost[no_policy] | xgboost[all] | -0.0004 [-0.0017, 0.0008] | equivalent |
| level_only vs all | xgboost[level_only] | xgboost[all] | 0.0006 [-0.0017, 0.0031] | equivalent |

## 5. Provisional selection

- Best by validation log loss: `xgboost[no_policy]`
- Selected: `logistic[no_policy]`
- Difference vs best [95% CI]: [-0.0017, 0.0071]
- Rule: simplest candidate not significantly worse than the best by validation log loss (paired, account-clustered bootstrap)
- Calibration and the operating threshold are finalized in the next step.
