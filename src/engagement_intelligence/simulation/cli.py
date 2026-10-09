"""Generate simulator outputs from layered config files.

Usage:
  uv run python -m engagement_intelligence.simulation.cli \
      --config configs/sim_reference.yaml [--config configs/sim_small.yaml] --out data/reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import write_parquet
from engagement_intelligence.simulation.config import SimConfig, load_sim_config
from engagement_intelligence.simulation.dynamics import simulate_dynamics
from engagement_intelligence.simulation.population import LATENT_PREFIX, sample_population


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic engagement world.")
    parser.add_argument("--config", action="append", required=True,
                        help="YAML layer; later layers override earlier ones")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)

    cfg = load_sim_config(*args.config)
    accounts = sample_population(cfg)
    states, episodes = simulate_dynamics(cfg, accounts)

    metadata = {"sim.config_fingerprint": cfg.fingerprint, "sim.seed": str(cfg.seed)}
    truth = args.out / "truth"
    write_parquet(accounts, truth / "accounts.parquet", metadata)
    write_parquet(states, truth / "state_daily.parquet", metadata)
    write_parquet(episodes, truth / "disengagement_episodes.parquet", metadata)
    manifest = {
        "config_fingerprint": cfg.fingerprint,
        "config_layers": args.config,
        "config": cfg.model_dump(mode="json"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _summarize_population(cfg, accounts)
    print()
    _summarize_dynamics(cfg, accounts, states, episodes)


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


def _summarize_dynamics(cfg: SimConfig, accounts: pa.Table, states: pa.Table, episodes: pa.Table) -> None:
    archetype_of = dict(zip(accounts.column("account_id").to_pylist(),
                            accounts.column("archetype").to_pylist()))
    labels = np.array([archetype_of[a] for a in states.column("account_id").to_pylist()])
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


if __name__ == "__main__":
    main()