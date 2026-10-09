# Held-out evaluation (test split)

Synthetic data for methodology evaluation only. The held-out split was not used for any modeling decision; this evaluation runs once (reruns require a recorded reason).

- Bundle: `lapse-60d-logistic-no-policy-198a1617fa`
- Run: first run (previous runs: 0)
- feature spec fingerprint: `d177b052d82e477d`
- labeling config fingerprint: `64cfdf882d3bef12`
- modeling config fingerprint: `0aef6dd85b1168bf`
- operating config fingerprint: `3af96ef0e1049b53`
- roles fingerprint: `77741094123c1254`
- source run fingerprint: `8918c959fdd5f9fa`

## 1. Selected model vs recency baseline

| model | log loss [95% CI] | PR-AUC | ROC-AUC | Brier | ECE | precision@10% | recall@10% |
|---|---|---|---|---|---|---|---|
| model | 0.2438 [0.2300, 0.2623] | 0.549 | 0.882 | 0.0721 | 0.0150 | 0.584 | 0.498 |
| baseline | 0.2815 [0.2637, 0.2999] | 0.466 | 0.825 | 0.0809 | 0.0205 | 0.531 | 0.453 |

- Log loss difference (model - baseline): -0.0377 [-0.0448, -0.0295]

## 2. Calibration on the held-out split

- ECE: 0.0150; slope / intercept: 1.043 / -0.147 (ideal 1 / 0); calibration applied: none

| week | rows | mean predicted | observed |
|---|---|---|---|
| 2025-12-01 | 1560 | 0.128 | 0.124 |
| 2025-12-08 | 1564 | 0.130 | 0.121 |
| 2025-12-15 | 1573 | 0.127 | 0.117 |
| 2025-12-22 | 1572 | 0.129 | 0.109 |
| 2025-12-29 | 1584 | 0.130 | 0.111 |
| 2026-01-05 | 1595 | 0.132 | 0.112 |
| 2026-01-12 | 1596 | 0.131 | 0.108 |
| 2026-01-19 | 1597 | 0.128 | 0.107 |
| 2026-01-26 | 1602 | 0.130 | 0.114 |
| 2026-02-02 | 1596 | 0.128 | 0.116 |
| 2026-02-09 | 1591 | 0.127 | 0.112 |
| 2026-02-16 | 1590 | 0.130 | 0.117 |
| 2026-02-23 | 1586 | 0.130 | 0.117 |
| 2026-03-02 | 1587 | 0.134 | 0.125 |
| 2026-03-09 | 1594 | 0.137 | 0.124 |
| 2026-03-16 | 1588 | 0.138 | 0.127 |
| 2026-03-23 | 1590 | 0.140 | 0.123 |
| 2026-03-30 | 1586 | 0.140 | 0.124 |
| 2026-04-06 | 1583 | 0.141 | 0.121 |
| 2026-04-13 | 1581 | 0.140 | 0.123 |

## 3. Validation-derived operating point applied unchanged (demo)

- HIGH: score >= 0.3765 (set for 10% weekly review capacity on validation)
- Realized weekly HIGH share: mean 0.118, range 0.109-0.136
- At HIGH: precision 0.552, recall 0.554, lift 4.70

Per 1,000 eligible accounts per week:

| flagged HIGH | lapses caught | false alarms | lapses missed |
|---|---|---|---|
| 117.8 | 65.1 | 52.8 | 52.5 |

## 4. Observable slices

| slice | value | rows | prevalence | mean predicted | gap | PR-AUC | HIGH share | precision | recall |
|---|---|---|---|---|---|---|---|---|---|
| cold_start | warm | 26427 | 0.121 | 0.141 | 0.020 | 0.570 | 0.131 | 0.555 | 0.598 |
| cold_start | cold | 5288 | 0.098 | 0.090 | -0.008 | 0.402 | 0.053 | 0.520 | 0.280 |
| tenure_days | <90 | 1428 | 0.039 | 0.068 | 0.029 | 0.173 | 0.019 | 0.333 | 0.161 |
| tenure_days | 90-180 | 2684 | 0.127 | 0.093 | -0.034 | 0.456 | 0.059 | 0.557 | 0.258 |
| tenure_days | 180-365 | 5243 | 0.101 | 0.104 | 0.003 | 0.553 | 0.083 | 0.573 | 0.471 |
| tenure_days | >=365 | 22360 | 0.125 | 0.148 | 0.023 | 0.573 | 0.139 | 0.551 | 0.613 |
