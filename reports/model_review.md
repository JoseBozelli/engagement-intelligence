# Model review (validation)

Synthetic data for methodology evaluation only. Thresholds are demo values, not operating thresholds for any real deployment. The held-out test split was not used.

- Selected model: `logistic[no_policy]`
- Operating config fingerprint: `3af96ef0e1049b53`
- modeling config fingerprint: `5399367fbd7ec9ce`
- feature spec fingerprint: `d177b052d82e477d`
- labeling config fingerprint: `64cfdf882d3bef12`
- source run fingerprint: `8918c959fdd5f9fa`

## 1. Calibration

- ECE (raw): 0.0080; tolerance: 0.02
- Calibration slope / intercept (raw): 1.020 / -0.070 (ideal 1 / 0)
- Decision: **none**
- ECE (after decision): 0.0080

| bin | mean predicted | observed | count |
|---|---|---|---|
| 1 | 0.005 | 0.003 | 1893 |
| 2 | 0.011 | 0.012 | 1893 |
| 3 | 0.017 | 0.013 | 1892 |
| 4 | 0.024 | 0.027 | 1893 |
| 5 | 0.035 | 0.030 | 1892 |
| 6 | 0.052 | 0.043 | 1893 |
| 7 | 0.080 | 0.068 | 1893 |
| 8 | 0.134 | 0.112 | 1892 |
| 9 | 0.264 | 0.251 | 1893 |
| 10 | 0.560 | 0.552 | 1893 |

Weekly calibration (mean predicted vs observed):

| week | rows | mean predicted | observed |
|---|---|---|---|
| 2025-07-07 | 1391 | 0.113 | 0.100 |
| 2025-07-14 | 1404 | 0.112 | 0.104 |
| 2025-07-21 | 1406 | 0.110 | 0.098 |
| 2025-07-28 | 1423 | 0.111 | 0.103 |
| 2025-08-04 | 1434 | 0.114 | 0.110 |
| 2025-08-11 | 1451 | 0.118 | 0.109 |
| 2025-08-18 | 1455 | 0.119 | 0.111 |
| 2025-08-25 | 1464 | 0.121 | 0.115 |
| 2025-09-01 | 1476 | 0.122 | 0.113 |
| 2025-09-08 | 1495 | 0.125 | 0.114 |
| 2025-09-15 | 1503 | 0.124 | 0.117 |
| 2025-09-22 | 1509 | 0.123 | 0.121 |
| 2025-09-29 | 1516 | 0.123 | 0.125 |

## 2. Operating point (demo)

- HIGH: score >= 0.3765 (review capacity 10% of eligible accounts per week)
- MEDIUM: score >= 0.1318 (watch capacity 25%)
- Realized weekly HIGH share: mean 0.100, range 0.088-0.110
- Realized weekly HIGH+MEDIUM share: mean 0.250, range 0.231-0.268
- At HIGH: precision 0.552, recall 0.497, lift 4.97; PR-AUC 0.509

Per 1,000 eligible accounts per week:

| flagged HIGH | lapses caught | false alarms | lapses missed |
|---|---|---|---|
| 100.0 | 55.2 | 44.9 | 55.8 |

False alarms consume review time and send outreach to accounts that would have stayed engaged; missed lapses are accounts that disengage without timely contact. The capacity setting trades one for the other.

## 3. Error analysis by slice

| slice | value | rows | prevalence | mean predicted | gap | PR-AUC | HIGH share | precision | recall |
|---|---|---|---|---|---|---|---|---|---|
| cold_start | warm | 18156 | 0.113 | 0.120 | 0.007 | 0.514 | 0.104 | 0.552 | 0.508 |
| cold_start | cold | 771 | 0.074 | 0.075 | 0.001 | 0.339 | 0.017 | 0.462 | 0.105 |
| tenure_days | <90 | 1628 | 0.076 | 0.075 | -0.001 | 0.285 | 0.025 | 0.400 | 0.129 |
| tenure_days | 90-180 | 1819 | 0.114 | 0.096 | -0.018 | 0.504 | 0.079 | 0.653 | 0.452 |
| tenure_days | 180-365 | 3281 | 0.114 | 0.104 | -0.010 | 0.604 | 0.084 | 0.642 | 0.469 |
| tenure_days | >=365 | 12199 | 0.114 | 0.131 | 0.017 | 0.515 | 0.118 | 0.528 | 0.544 |
| archetype (truth) | chronic_low | 1521 | 0.377 | 0.298 | -0.079 | 0.640 | 0.351 | 0.646 | 0.602 |
| archetype (truth) | disruption_lapse | 1505 | 0.160 | 0.165 | 0.005 | 0.606 | 0.146 | 0.621 | 0.564 |
| archetype (truth) | disruption_recovery | 2428 | 0.084 | 0.091 | 0.006 | 0.419 | 0.059 | 0.524 | 0.366 |
| archetype (truth) | false_alarm | 2125 | 0.092 | 0.104 | 0.012 | 0.587 | 0.081 | 0.630 | 0.559 |
| archetype (truth) | gradual_decline | 2390 | 0.197 | 0.183 | -0.015 | 0.550 | 0.164 | 0.608 | 0.506 |
| archetype (truth) | sparse_stable | 3157 | 0.088 | 0.134 | 0.046 | 0.243 | 0.097 | 0.274 | 0.302 |
| archetype (truth) | stable_high | 5801 | 0.024 | 0.040 | 0.016 | 0.368 | 0.021 | 0.452 | 0.409 |

## 4. HIGH flags vs latent truth (evaluation only)

- Label precision among HIGH: 0.552
- Latent precision among HIGH (latent-disengaged or onset within horizon): 0.658
- Label false alarms that are latent-positive: 0.313 (266 of 849)

| archetype | share of false alarms | share of missed lapses |
|---|---|---|
| chronic_low | 0.223 | 0.216 |
| disruption_lapse | 0.098 | 0.099 |
| disruption_recovery | 0.080 | 0.123 |
| false_alarm | 0.075 | 0.081 |
| gradual_decline | 0.181 | 0.220 |
| sparse_stable | 0.263 | 0.184 |
| stable_high | 0.080 | 0.077 |
