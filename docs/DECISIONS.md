# Decision Log: Engagement Intelligence

Format: Observation / Decision / Alternatives / Evidence / Consequence.
Append-only; superseded entries are marked SUPERSEDED, never deleted.
Status: LOCKED = approved at a checkpoint; ACTIVE = made during implementation, open to review.

This repository is a reference implementation. The event vocabulary, target definition,
windows, and thresholds are illustrative defaults supplied as data or configuration;
other deployments define their own.

---

## Day 1 design checkpoint (approved 2026-09-28, amendments A-E)

### D-001 Prediction target: observable lapse [LOCKED]
- Observation: operational systems observe behavior; latent disengagement is never observed.
- Decision: for an eligible account assessed at t, y(t)=1 iff no account-initiated engagement
  event is recorded in (t, t+H]. Eligible iff >= 1 such event in (t-90d, t]. Illustrative
  defaults: H=60d, eligibility window 90d. Engagement events (actor=account): session.booked,
  session.completed, message.replied, followup.completed, task.completed. Excluded:
  message.opened (passive, noisy tracking) and account cancellations (contact, not
  engagement). Rows with t+H beyond the data end are dropped (censored), never labeled 0.
  Latent disengagement onset is stored only in simulator truth, for evaluation and diagnosis.
- Alternatives: latent onset as label (not observable in practice); separate horizon and
  lapse parameters (more knobs, no demonstrated gain).
- Evidence: design review; label noise vs latent truth measured at the Day-1 checkpoint.
- Consequence: realistic, noisy labels for sparse and false-alarm trajectories. Noise is
  measured and reported, not removed.

### D-002 Weekly assessment cadence [LOCKED]
- Decision: score every eligible account once per week on a fixed weekday.
- Alternatives: daily (about 7x the rows, strongly autocorrelated); event-triggered
  (irregular intervals complicate risk_change).
- Consequence: rows correlated within account; CIs use account-clustered bootstrap.

### D-003 Temporal split, purged label windows, cold-start slice [LOCKED]
- Observation: scoring runs forward in time over an existing population of accounts.
- Decision: 30 simulated months. 1-6 burn-in (no assessments); 7-18 train; 19-23
  validation; 24-28 test; 29-30 label horizon only. Rows whose (t, t+H] crosses the next
  split's start are purged. Fitted transforms use train only. The test set is evaluated once
  via a separate command. Accounts created after the train period form a separately
  reported cold-start slice.
- Alternatives: random row split (temporal and entity leakage); account-only split (does not
  reflect forward-in-time scoring).
- Consequence: fewer rows near boundaries; honest forward-looking estimates.

### D-004 Configuration boundary [LOCKED]
- Decision: vocabulary, target definition, windows, and thresholds are inputs (data/config),
  not code. The repository ships one reference vocabulary and illustrative defaults, and
  documents them as such. Code in core/ never depends on a specific vocabulary.
- Consequence: the same code runs standalone, alongside SignalGuard, or under a different
  vocabulary and parameters without modification.

### D-005 SignalGuard reuse by faithful copy [LOCKED]
- Decision: small generic SignalGuard components are copied into core/ with a provenance
  header and extended by subclassing or composition, never edited in place. No shared
  package yet; extraction is deferred until integration across repositories.
- Not carried over: LLM provider and embedding interfaces (no LLM in this project);
  module-level mutable state.

### D-006 Dataset scale [LOCKED, Amendment A]
- Decision: default profile 2,000 accounts; small/test profile 300. A 4,000-account
  benchmark only if runtime is trivial AND it materially improves evaluation stability
  (e.g., cold-start CIs too wide). Dataset size is not an objective.

### D-007 Prevalence is a reported diagnostic [LOCKED, Amendment B]
- Decision: aim for roughly 10-25% overall observable-lapse prevalence. Report it by train,
  validation, test, cold-start, and archetype. Temporal differences are investigated, not
  tuned away; the simulator is never tuned to make splits look IID.
- Supersedes: draft criterion "10-20% positives in every split".

### D-008 Heuristic AUC is a diagnostic, not a simulator target [LOCKED, Amendment C]
- Decision: report validation ROC-AUC of days-since-last-engagement. > ~0.90: investigate
  triviality. ~0.5-0.6: check other longitudinal features before concluding the problem is
  unlearnable. The simulator is judged on mechanism coherence, noise, longitudinal
  structure, absence of leakage, and trajectory heterogeneity. Simulator changes are
  justified by mechanism, never by downstream metrics, and logged as
  Question/Hypothesis/Experiment/Result/Decision.
- Supersedes: draft pass band "AUC 0.65-0.80".

### D-009 Duplicate semantics [LOCKED, Amendment D]
- Decision: an exact duplicate is the same underlying event ingested or stored more than
  once; it preserves event_id and all fields. Primary exact dedup key: event_id.
  (ref_id, event_type, actor) is not a dedup key, because distinct events legitimately share
  it (e.g., a message opened twice). Near-duplicates (distinct event_id, same real-world
  action) would be a separate, explicitly labeled noise mechanism. The raw store may hold
  exact duplicates; dedup happens in the feature layer.
- Status: near-duplicate simulation deferred.

### D-010 Scalability-preserving decisions [LOCKED, Amendment E]
- Decision: tenant_id in the envelope (constant "demo" here); Parquet storage; Pydantic
  validation at boundaries only, not in vectorized paths; occurrence time (`timestamp`)
  vs recorded_at, with point-in-time features filtering on recorded_at <= t;
  config-driven simulation; batch weekly assessment; no LLM dependency. Multi-tenancy
  itself is not implemented.

---

## Step 1 implementation decisions

### D-011 Independent seeded RNG streams [ACTIVE]
- Observation: one shared generator makes each account's draws depend on generation order
  and on unrelated random calls, so changing a noise parameter would reshuffle latents.
- Decision: rng_for(master_seed, stream, index) via numpy SeedSequence spawn keys, with
  named streams (e.g., latents, events, noise); names hashed with blake2b.
- Alternatives: single global generator (order-dependent).
- Evidence: tests/test_core_seeding.py.
- Consequence: one mechanism can change without perturbing others (supports D-008).

### D-012 Attributes stored as canonical JSON [ACTIVE]
- Decision: attributes_json column, sorted keys, compact separators (byte-stable files).
  Attributes that become frequent feature inputs are promoted to typed columns by a later,
  logged decision.
- Alternatives: map<string,string> (loses types); struct of all keys (sparse, schema churn);
  a column per attribute now (premature).
- Consequence: JSON parse cost if attributes are used heavily; monitor.

### D-013 Identifier conventions [ACTIVE]
- Decision: synthetic event and ref IDs are keyed blake2b hashes (opaque). Synthetic
  account IDs are readable indices assigned independently of any latent property. Incoming
  IDs are validated only against a permissive generic pattern, so external formats
  (UUIDs, prefixed keys) are accepted unchanged.
- Consequence: the envelope ingests events from other systems without ID rewriting.

### D-014 Alignment with SignalGuard's event base [ACTIVE]
- Decision: core/base_event.py is a verbatim copy of SignalGuard's BaseCorrelatedEvent and
  validate_correlation_ids. The envelope extends it: `timestamp` keeps its meaning
  (occurrence time); recorded_at, tenant_id, actor, attributes, schema_version are added;
  event_id is overridden to be required because the base's random default breaks
  reproducibility. Correlation uses ref_id / related_ref_id. The reference vocabulary reuses
  SignalGuard's eight type names and ref-id rules (including followup.required as the only
  type allowing related_ref_id) and adds session.cancelled and message.*.
- Supersedes: draft names occurred_at, correlation_id, *.requested.
- Consequence: projection into SignalGuard = filter to the shared types and drop added fields.

### D-015 Vocabulary as data [ACTIVE]
- Decision: EventVocabulary (frozen) declares event types, allowed actors per type, and
  ref-id rule sets; validation delegates the ref-id rules to validate_correlation_ids.
  Construction rejects incomplete or inconsistent vocabularies. Round-trips to a plain dict
  for config files. Actor values in the reference vocabulary: account, operator, system.
- Alternatives: rule sets hardcoded in the typed Event validator (couples mechanism to one
  vocabulary).
- Evidence: tests/test_core_vocabulary.py uses an unrelated toy vocabulary.

### D-016 Arrow boundary hardening and timezone data [ACTIVE]
- Observation: pyarrow's from_pylist fills missing keys with nulls even in non-nullable
  columns (caught only at Parquet write). On Windows, converting UTC timestamps back to
  Python requires the tzdata package (no OS timezone database).
- Decision: check_schema() rejects nulls in non-nullable columns and runs inside both
  to_arrow() and from_arrow(). tzdata is a declared dependency.
- Evidence: Step 1 test run (misspelled column passed to_arrow; zoneinfo lookup failed on
  Windows); regression test test_nulls_in_required_columns_are_rejected.
- Consequence: malformed tables fail at the boundary that produced them, on every platform.

## Step 2 implementation decisions

### D-017 Layered, validated configuration with fingerprints [ACTIVE]
- Observation: simulator, target, and model parameters must be swappable without code
  changes, and every output must be traceable to the exact parameters that produced it.
- Decision: YAML layers deep-merged in order (mappings merge; lists and scalars are
  replaced), validated by frozen Pydantic models with unknown keys rejected. A config
  fingerprint (sha256 of canonical JSON, 16 hex chars) is written into Parquet metadata
  and a run manifest.
- Alternatives: CLI flags only (unwieldy for nested parameters); Python modules as config
  (couples parameters to code); full copies per profile (duplication, drift).
- Evidence: tests/test_core_config.py, tests/test_sim_config.py.
- Consequence: a new benchmark or profile is a new YAML layer; typos fail at load time.

### D-018 Population model [ACTIVE]
- Decision: latent traits and archetypes are declared in configuration. Latents use a
  standardized scale (0 = population-typical, units of one SD) with per-archetype normal
  priors and a shared per-archetype SD. Archetype mixture weights are normalized. Arrivals:
  an established population created during an initial window, plus uniform later arrivals
  until arrivals_end. Archetype, arrival, and latents use separate random streams per
  account index. Truth-table latent columns carry the prefix `latent_`.
- Alternatives: Beta-distributed traits on [0,1] (awkward in additive linear predictors);
  explicit latent covariance (not needed; archetype means already induce correlation).
- Evidence: tests/test_population.py (determinism, size invariance, stream isolation,
  mixture and prior recovery).
- Consequence: archetypes defined mainly by dynamics (gradual_decline, false_alarm,
  disruption_*) differ only weakly in static traits until time-varying state is added in
  the event-generation step.

## Step 3 design (approved 2026-10-08)

### D-019 Daily latent dynamics [LOCKED]
- Decision: daily discrete-time simulation per account from creation to calendar end.
  State: satisfaction (mean-reverting toward a latent anchor with optional per-archetype
  drift, daily noise, and externally supplied experience shocks); disruption episodes
  (daily onset logit from latents, geometric duration); dip episodes (per-archetype rate
  and duration; affect observable activity only, never the hazard); engagement propensity
  E(t) (linear in latents and state); disengaged (escapable state with daily hazard and
  recovery logits). Every coefficient is configuration; the state variables a pathway may
  reference are fixed by the mechanics and validated. Implemented as a stepper so the event
  generator can drive it day by day and feed back shocks. Work is split into dynamics (3a),
  observable events with an observables-only operator policy (3b), and observation noise (3c).
- Alternatives: continuous-time event simulation (exact but harder to read and test);
  trajectories computed independently of events (no experience feedback).
- Evidence: tests/test_dynamics.py.
- Consequence: about 1.4M account-days at the reference size; pure-Python loop runtime of
  tens of seconds, accepted to preserve per-account stream determinism (D-011).

### D-020 Stable account index and per-process random columns [ACTIVE]
- Decision: the account truth table stores account_index, which keys every downstream
  random stream (rather than row position). Within the dynamics stream, random numbers are
  pre-drawn with one column per process.
- Evidence: test_disengagement_parameters_do_not_perturb_other_processes.
- Consequence: filtering or reordering accounts, or changing one process's parameters,
  cannot alter another account's or another process's trajectory.

### D-021 Separate activity frequency from commitment in propensity [ACTIVE]
- Question: why did sparse_stable disengage at ~50% (reference run, Step 3a)?
- Hypothesis: baseline_engagement in E(t) treats infrequent activity as low commitment.
- Experiment: hand calculation with configured coefficients reproduced the simulated rate
  (~47% predicted vs 49.5% observed).
- Result: mechanism confirmed; behavior contradicts the archetype's intent (sparse but stable).
- Decision: remove baseline_engagement from the propensity predictor. It drives activity
  frequency only (event generation); commitment comes from satisfaction, friction,
  disruption, and persistence. No other parameter changed.
- Consequence: sparse_stable becomes a low-activity, moderate-risk population, a deliberate
  source of false warnings for observable-only models.

## Step 3b implementation decisions

### D-022 Event generation and operator policy [ACTIVE]
- Decision: a per-account EventGenerator observes daily state via a generic observer hook
  in simulate() and returns experience shocks (satisfaction only). One open booking at a
  time; each booked session resolves to exactly one outcome (operator cancellation, account
  cancellation, no-show, completed). Completed sessions may produce follow-ups linked by
  related_ref_id. Tasks are issued periodically. Messages form sent -> opened -> replied
  threads. Operator policy: reminders before sessions and nudges after observed silence with
  a cooldown, computed only from the account's events whose time has passed. Every parameter
  is configuration; predictor inputs are validated.
- Alternatives: precomputed trajectories without feedback (loses realism); a policy driven
  by latent state (leaks truth into observables).
- Evidence: tests/test_events.py (lifecycle integrity, determinism, behavior, feedback).
- Consequence: the operator policy is the seam where a later intervention layer plugs in.

### D-023 Clean events are simulator truth [ACTIVE]
- Decision: the noise-free stream is written to truth/events_clean.parquet. The observable
  store is produced only by the observation-noise step.
- Consequence: no model can train on, or be evaluated against, noise-free observations by
  accident; noise is auditable against the clean stream.

### D-024 Bulk event validation [ACTIVE]
- Decision: EventVocabulary.check_table validates whole tables using the same rule logic as
  validate() (no per-row objects). check_schema also enforces recorded_at >= timestamp.
- Consequence: bulk producers and ingestion get full rule checks at Pydantic-free cost.

### D-025 Business-facing parameters use business units [LOCKED]
- Decision: every parameter a deployment is expected to tune (target horizon and
  eligibility, feature windows, risk bands, thresholds, change-alert rules) is expressed
  in business units (days, probabilities, counts, rates) with a stated operational meaning.
  Simulator internals may use modeling units (logits) because they only generate test data.
- Alternatives: convert simulator predictors to base probabilities and odds ratios now
  (deferred: readability gain does not change any business-facing interface).
- Consequence: integration requires configuring meanings, not reverse-engineering numbers.