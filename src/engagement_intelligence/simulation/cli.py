"""Generate simulator outputs from layered config files.

Usage:
    uv run python -m engagement_intelligence.simulation.cli \
        -- config configs/sim_reference.yaml [--config configs/sim_small.yaml] --out data/reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

from engagement_intelligence.core.arrow_io import write_parquet
from engagement_intelligence.simulation.config import SimConfig, load_sim_config
from engagement_intelligence.simulation.population import LATENT_PREFIX, sample_population

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic engagement world.")
    parser.add_argument("--config", action="append", required=True,
                        help="YAML layer; later layers override earlier ones")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    args = parser.parse_args(argv)

    cfg = load_sim_config(*args.config)
    accounts = sample_population(cfg)

    metadata = {"sim.config_fingerprint": cfg.fingerprint, "sim.seed": str(cfg.seed)}
    write_parquet(accounts, args.out / "truth" / "accounts.parquet", metadata)
    manifest = {
        "config_fingerprint": cfg.fingerprint,
        "config_layers": args.config,
        "config": cfg.model_dump(mode="json")
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _summarize(cfg, accounts)

def _summarize(cfg: SimConfig, accounts: pa.Table) -> None:
    archetypes = np.array(accounts.column("archetype").to_pylist())
    print(f"config fingerprint: {cfg.fingerprint}")
    print(f"accounts: {accounts.num_rows}")
    header = " ".join(f"{n[:10]:>10}" for n in cfg.latents)
    print(f"{'archetype':<20} {'share':>5} {header}")
    for name in cfg.archetypes:
        mask = archetypes == name
        means = [accounts.column(LATENT_PREFIX + n).to_numpy()[mask].mean() for n in cfg.latents]
        print(f"{name:<20} {mask.mean():5.3f} " + " ".join(f"{m:10.2f}" for m in means))

if __name__ == "__main__":
    main()