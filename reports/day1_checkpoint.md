# Day-1 checkpoint report

Synthetic data for methodology evaluation only; values are not estimates of any real population.

- Simulation config fingerprint: `8918c959fdd5f9fa`
- Labeling config fingerprint: `64cfdf882d3bef12`
- Snapshot (as_of): 2026-07-01T00:00:00+00:00
- Assessments: 102294 rows, 1982 accounts

## 1. Observable lapse prevalence (diagnostic, D-007)

| split | rows | prevalence | warm prevalence | cold-start rows | cold-start prevalence |
|---|---|---|---|---|---|
| train | 51652 | 0.099 | 0.099 | 0 | n/a |
| validation | 18927 | 0.111 | 0.113 | 771 | 0.074 |
| test | 31715 | 0.118 | 0.121 | 5288 | 0.098 |

By archetype (truth; evaluation only):

| archetype | train | validation | test |
|---|---|---|---|
| stable_high | 0.032 | 0.024 | 0.034 |
| sparse_stable | 0.061 | 0.088 | 0.081 |
| gradual_decline | 0.137 | 0.197 | 0.212 |
| disruption_recovery | 0.058 | 0.084 | 0.093 |
| disruption_lapse | 0.196 | 0.160 | 0.217 |
| chronic_low | 0.312 | 0.377 | 0.342 |
| false_alarm | 0.081 | 0.092 | 0.107 |

## 2. Heuristic diagnostic: days since last known engagement (D-008)

| split | AUC all | AUC warm | AUC cold-start | reading |
|---|---|---|---|---|
| train | 0.823 | 0.823 | n/a | plausible |
| validation | 0.819 | 0.819 | 0.791 | plausible |
| test | 0.825 | 0.827 | 0.810 | plausible |

## 3. Observable label vs latent truth

Latent positive: latent-disengaged at assessment, or latent onset within the horizon.

| group | rows | label rate | latent+ rate | P(label \| latent+) | P(label \| latent-) | P(latent+ \| label) |
|---|---|---|---|---|---|---|
| all | 102294 | 0.107 | 0.152 | 0.596 | 0.019 | 0.846 |
| stable_high | 31291 | 0.031 | 0.069 | 0.439 | 0.001 | 0.974 |
| sparse_stable | 16755 | 0.072 | 0.096 | 0.523 | 0.024 | 0.697 |
| gradual_decline | 13395 | 0.170 | 0.246 | 0.608 | 0.027 | 0.881 |
| disruption_recovery | 12673 | 0.074 | 0.137 | 0.510 | 0.005 | 0.945 |
| disruption_lapse | 8098 | 0.196 | 0.280 | 0.641 | 0.023 | 0.916 |
| chronic_low | 8848 | 0.331 | 0.327 | 0.743 | 0.131 | 0.734 |
| false_alarm | 11234 | 0.092 | 0.139 | 0.620 | 0.006 | 0.942 |

## 4. Observation noise: measured vs configured

| mechanism | configured | measured |
|---|---|---|
| drop | 0.030 | 0.0307 |
| delay | 0.100 | 0.1004 |
| duplicate | 0.005 | 0.0050 |

## 5. Day-1 stop criteria

| criterion | status |
|---|---|
| All archetypes present in assessments | PASS |
| Overall prevalence within 10-25% (diagnostic, D-007) | PASS |
| Heuristic AUC on 'validation' within 0.60-0.90 (diagnostic, D-008) | PASS |
| Label noise measurable: P(label \| latent-) > 0 and P(label \| latent+) < 1 | PASS |
| Noise rates within tolerance of configuration | PASS |
| Determinism and leakage controls | covered by test suite |
