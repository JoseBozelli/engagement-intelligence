# Synthetic Engagement World

The simulator generates a deterministic, longitudinal population of accounts, their latent
states, and the observable events those states produce. It exists to evaluate system
architecture and modeling methodology. It is not intended to estimate real-world
prevalence, real behavior, or production intervention thresholds.

## Pipeline

```text
configuration (layered YAML, validated, fingerprinted)
  -> population      archetype, arrival time, static latent traits            [truth]
  -> dynamics        daily satisfaction, disruption, dips, propensity,
                     escapable disengagement                                   [truth]
  -> events          sessions, follow-ups, tasks, messages; observables-only
                     operator policy; experience feedback into satisfaction    [truth: clean events]
  -> noise           drops, recording delay, timestamp jitter, exact duplicates [observable + truth log]
  -> labeling        separate, vocabulary-agnostic package                     [labels]
```

## Data layout

| Path | Content | Readers |
|---|---|---|
| `data/<run>/manifest.json` | Full configuration and its fingerprint | Provenance checks |
| `data/<run>/truth/accounts.parquet` | Archetype, arrival time, `latent_*` traits | Evaluation only |
| `data/<run>/truth/state_daily.parquet` | Daily satisfaction, disruption, dip, propensity, disengaged | Evaluation only |
| `data/<run>/truth/disengagement_episodes.parquet` | Latent disengagement onsets and returns | Evaluation only |
| `data/<run>/truth/events_clean.parquet` | Noise-free event stream | Evaluation only |
| `data/<run>/truth/noise_log.parquet` | Per-event drop, delay, jitter, duplication | Evaluation only |
| `data/<run>/observable/events.parquet` | Event stream after observation noise | Features, models, service |
| `data/<run>/labels/assessments.parquet` | Weekly point-in-time assessments and labels | Model training and evaluation |

Nothing on the modeling path reads `truth/`. An architecture test prevents the labeling
package from importing the simulator.

## Components

**Population.** Each account draws an archetype from a configured mixture, an arrival time
(an established population plus later arrivals), and six static latent traits on a
standardized scale (0 = population-typical; units of one standard deviation).

**Dynamics.** A daily stepper evolves satisfaction (mean-reverting toward a latent anchor,
with optional per-archetype drift and experience shocks), disruption episodes, dip
episodes (activity drops that do not raise risk), engagement propensity (a linear
combination of traits and state), and an escapable disengaged state driven by a daily
hazard and recovery probability. Activity frequency and commitment are separate concepts:
frequency drives booking rates; commitment drives the hazard (D-021).

**Events.** A per-account generator observes daily state and emits reference-vocabulary
events. Each booked session resolves to exactly one outcome (completed, no-show, account
cancellation, operator cancellation). Completed sessions may produce linked follow-ups;
tasks recur on a fixed interval; messages form sent, opened, and replied threads. The
operator policy (reminders, nudges after observed silence, follow-ups, tasks) reads only
the account's past events, never latent state.

**Noise.** Configured drops, lognormal recording delay, bounded timestamp jitter, and exact
duplicates (identical rows, same `event_id`) transform the clean stream into the
observable store. Events recorded at or after the snapshot are excluded as not yet
observable. System-of-record events are exempt.

## Determinism and isolation

A single master seed derives an independent random stream per account and per process
(population, dynamics, events, noise). Draws are made for every opportunity with one
column per mechanism, so outputs are invariant to generation order and population size,
and changing one mechanism's parameters cannot change another mechanism's draws. Every
output file carries the configuration fingerprint in its metadata.

## Configuration

- `configs/sim_reference.yaml`: reference world (2,000 accounts, 30 months).
- `configs/sim_small.yaml`: overlay for tests and fast iteration (300 accounts).
- `configs/labeling_reference.yaml`: target definition and split plan.

Simulator pathways use modeling units (logits) because they only generate test data.
Parameters a deployment tunes (target horizon, eligibility, settlement, splits, later
thresholds) use business units (D-025).

## Validation evidence

The reference run is evaluated in `reports/day1_checkpoint.md`. Summary:

| Check | Result |
|---|---|
| Observable lapse prevalence (train / validation / test) | 0.099 / 0.111 / 0.118 |
| Recency heuristic ROC-AUC (all splits) | 0.82 to 0.83 |
| Label precision against latent truth, P(latent+ \| label) | 0.85 |
| Label sensitivity against latent truth, P(label \| latent+) | 0.60 |
| Measured noise (drop / delay / duplicate) | 0.031 / 0.100 / 0.005 (configured 0.03 / 0.10 / 0.005) |

## Known characteristics and limitations

- Label noise concentrates in low-activity accounts: latently engaged `chronic_low` accounts
  receive lapse labels 13% of the time because their natural gaps exceed the horizon.
- Observable labels capture about 60% of latent-positive assessments; onsets late in the
  horizon follow real activity, and newly disengaged accounts leave residual events.
- Prevalence rises across splits through configured decline drift and population ageing;
  the shift is retained, so calibration must be checked per split.
- Cold-start accounts show lower prevalence, consistent with an earlier lifecycle stage.
- 18 of 2,000 reference accounts never meet the eligibility rule and are never assessed.
- Out of scope: seasonality, multiple tenants, multiple operator policies, near-duplicate
  events, and any calibration to real populations.

## Running

```bash
uv run python -m engagement_intelligence.simulation.cli --config configs/sim_reference.yaml --out data/reference
uv run python -m engagement_intelligence.labels.cli --events data/reference/observable/events.parquet --config configs/labeling_reference.yaml --as-of 2026-07-01 --out data/reference/labels/assessments.parquet
uv run python -m engagement_intelligence.simulation.diagnostics --run data/reference --labeling-config configs/labeling_reference.yaml --out reports/day1_checkpoint.md
```