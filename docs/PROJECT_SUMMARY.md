# Project summary

**Objective.** Predict, each week, which accounts are likely to disengage within 60 days, from
irregular and noisy behavioral event streams, and serve calibrated, explainable assessments
through a typed API for downstream decision systems.

**Approach.** A deterministic synthetic world (2,000 accounts, 30 months, seven behavioral
archetypes, recording delays and data-quality noise, separately stored latent truth); point-in-time
labels with settlement and purged temporal splits; nineteen declarative features verified by a
truncation audit; recency baseline, logistic regression, and XGBoost compared with purged
walk-forward tuning and paired account-clustered bootstrap; capacity-based risk bands; a
self-describing model bundle; a batch-first scoring engine, HTTP API, typed client, and demo UI.

**Results (held-out, evaluated once).** Log loss 0.244 vs 0.282 for the recency baseline
(difference -0.038, 95% CI [-0.045, -0.030]); PR-AUC 0.549 vs 0.466; ROC-AUC 0.882 vs 0.825;
precision at 10% weekly review capacity 0.584 vs 0.531; ECE 0.015. XGBoost was equivalent to
logistic regression; the simpler model was selected. Live scoring reproduces offline predictions
on all 31,715 held-out rows.

**Limitations.** Synthetic data only; weak ranking for early-lifecycle accounts; slight uniform
over-prediction under population drift; single-process demo storage.

**Engineering.** 205 tests; CI on Linux and Windows; every artifact fingerprinted to its
configuration; decision log of 47 entries including rejected hypotheses.