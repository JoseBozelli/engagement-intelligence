"""Generate simulator outputs from layered config files.

Usage:
  uv run python -m engagement_intelligence.simulation.cli \
      --config configs/sim_reference.yaml [--config configs/sim_small.yaml] --out data/reference
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import write_parquet
from engagement_intelligence.simulation.config import SimConfig, load_sim_config
from engagement_intelligence.simulation.events import simulate_world
from engagement_intelligence.simulation.population import LATENT_PREFIX, sample_population
from engagement_intelligence.simulation.noise import apply_noise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic engagement world.")
    parser.add_argument("--config", action="append", required=True,
                        help="YAML layer; later layers override earlier ones")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)

    cfg = load_sim_config(*args.config)
    accounts = sample_population(cfg)
    states, episodes, events = simulate_world(cfg, accounts)

    metadata = {"sim.config_fingerprint": cfg.fingerprint, "sim.seed": str(cfg.seed)}
    truth = args.out / "truth"
    write_parquet(accounts, truth / "accounts.parquet", metadata)
    write_parquet(states, truth / "state_daily.parquet", metadata)
    write_parquet(episodes, truth / "disengagement_episodes.parquet", metadata)
    write_parquet(events, truth / "events_clean.parquet", metadata)
    observable, noise_log = apply_noise(cfg, accounts, events)
    write_parquet(noise_log, truth / "noise_log.parquet", metadata)
    write_parquet(observable, args.out / "observable" / "events.parquet", metadata)
    manifest = {
        "config_fingerprint": cfg.fingerprint,
        "config_layers": args.config,
        "config": cfg.model_dump(mode="json"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archetype_of = dict(zip(accounts.column("account_id").to_pylist(),
                            accounts.column("archetype").to_pylist()))
    state_labels = np.array([archetype_of[a] for a in states.column("account_id").to_pylist()])

    _summarize_population(cfg, accounts)
    print()
    _summarize_dynamics(cfg, archetype_of, state_labels, states, episodes)
    print()
    _summarize_events(cfg, archetype_of, state_labels, events)
    print()
    _summarize_noise(cfg, events, observable, noise_log)


def _summarize_population(cfg: SimConfig, accounts: pa.Table) -> None:
    archetypes = np.array(accounts.column("archetype").to_pylist())
    print(f"config fingerprint: {cfg.fingerprint}")
    print(f"accounts: {accounts.num_rows}")
    header = "  ".join(f"{n[:10]:>10}" for n in cfg.latents)
    print(f"{'archetype':<20} {'share':>5}  {header}")
    for name in cfg.archetypes:
        mask = archetypes == name
        means = [accounts.column(LATENT_PREFIX + n).to_numpy()[mask].mean() for n in cfg.latents]
        print(f"{name:<20} {mask.mean():5.3f}  " + "  ".join(f"{m:10.2f}" for m in means))


def _summarize_dynamics(cfg: SimConfig, archetype_of: dict, labels: np.ndarray,
                        states: pa.Table, episodes: pa.Table) -> None:
    disrupted = np.array(states.column("disrupted").to_pylist())
    dip = np.array(states.column("dip").to_pylist())
    ep_accounts = episodes.column("account_id").to_pylist()
    ep_ended = [e is not None for e in episodes.column("end").to_pylist()]
    ever = set(ep_accounts)
    ongoing = {a for a, ended in zip(ep_accounts, ep_ended) if not ended}

    print(f"account-days: {states.num_rows}   disengagement episodes: {episodes.num_rows}")
    print(f"{'archetype':<20} {'disrupted':>9} {'dip':>6} {'ever_dis':>9} {'dis_at_end':>10} {'returned':>9}")
    for name in cfg.archetypes:
        members = [a for a, n in archetype_of.items() if n == name]
        ended = [e for a, e in zip(ep_accounts, ep_ended) if archetype_of[a] == name]
        mask = labels == name
        returned = np.mean(ended) if ended else float("nan")
        print(
            f"{name:<20} {disrupted[mask].mean():9.3f} {dip[mask].mean():6.3f} "
            f"{np.mean([a in ever for a in members]):9.3f} "
            f"{np.mean([a in ongoing for a in members]):10.3f} {returned:9.3f}"
        )


def _summarize_events(cfg: SimConfig, archetype_of: dict, labels: np.ndarray, events: pa.Table) -> None:
    exposure_days = Counter(labels.tolist())
    engagement_types = set(cfg.events.messages.nudge_reset_by)
    c: Counter = Counter()
    for account, event_type, actor, attrs in zip(
        events.column("account_id").to_pylist(),
        events.column("event_type").to_pylist(),
        events.column("actor").to_pylist(),
        events.column("attributes_json").to_pylist(),
    ):
        arch = archetype_of[account]
        c[arch, event_type, actor] += 1
        if actor == "account" and event_type in engagement_types:
            c[arch, "engagement"] += 1
        if event_type == "message.sent" and json.loads(attrs).get("kind") == "nudge":
            c[arch, "nudge"] += 1

    def pct(n: int, d: int) -> float:
        return 100.0 * n / d if d else float("nan")

    print(f"clean events: {events.num_rows}")
    print(f"{'archetype':<20} {'eng/yr':>7} {'sess/yr':>8} {'compl%':>7} {'noshow%':>8} "
          f"{'cancel%':>8} {'nudge/yr':>9} {'reply%':>7}")
    for name in cfg.archetypes:
        years = exposure_days[name] / 365.25
        booked = c[name, "session.booked", "account"] + c[name, "session.booked", "operator"]
        completed = c[name, "session.completed", "account"]
        no_show = c[name, "session.no_show", "account"]
        acct_cancel = c[name, "session.cancelled", "account"]
        resolved = completed + no_show + acct_cancel + c[name, "session.cancelled", "operator"]
        print(
            f"{name:<20} {c[name, 'engagement'] / years:7.1f} {booked / years:8.1f} "
            f"{pct(completed, resolved):7.1f} {pct(no_show, resolved):8.1f} "
            f"{pct(acct_cancel, resolved):8.1f} {c[name, 'nudge'] / years:9.1f} "
            f"{pct(c[name, 'message.replied', 'account'], c[name, 'message.sent', 'operator']):7.1f}"
        )

def _summarize_noise(cfg: SimConfig, clean: pa.Table, observable: pa.Table, log: pa.Table) -> None:
    nz = cfg.noise
    exempt = np.isin(np.array(clean.column("event_type").to_pylist()), list(nz.exempt_types))
    dropped = np.array(log.column("dropped").to_pylist())
    duplicated = np.array(log.column("duplicated").to_pylist())
    observed = np.array(log.column("observed").to_pylist())
    delay = log.column("delay_hours").to_numpy()
    delayed = delay > 0

    print(f"observable events: {observable.num_rows}   (clean: {clean.num_rows})")
    print(f"dropped     {dropped[~exempt].mean():.4f}   (config {nz.drop_prob})")
    print(f"delayed     {delayed[~exempt].mean():.4f}   (config {nz.delay_prob})   "
          f"median {np.median(delay[delayed]):.1f} h   p95 {np.percentile(delay[delayed], 95):.1f} h   "
          f"(config median {nz.delay_median_hours} h)")
    print(f"duplicated  {duplicated[~exempt].mean():.4f}   (config {nz.duplicate_prob})")
    print(f"recorded at/after horizon (not yet observable): {int((~dropped & ~observed).sum())}")

if __name__ == "__main__":
    main()