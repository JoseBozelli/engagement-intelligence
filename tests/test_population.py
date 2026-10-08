from datetime import timedelta

import numpy as np
import pytest

from engagement_intelligence.core.arrow_io import read_parquet_metadata, write_parquet
from engagement_intelligence.simulation.population import (
    LATENT_PREFIX,
    account_schema,
    sample_population,
    utc_midnight,
)


def _with_population(cfg, **changes):
    return cfg.model_copy(update={"population": cfg.population.model_copy(update=changes)})


@pytest.fixture(scope="module")
def ref_pop(reference_cfg):
    return sample_population(reference_cfg)


def test_population_is_deterministic(small_cfg):
    assert sample_population(small_cfg).equals(sample_population(small_cfg))


def test_accounts_are_invariant_to_population_size(small_cfg):
    small = sample_population(small_cfg)
    larger = sample_population(_with_population(small_cfg, n_accounts=400))
    assert larger.slice(0, small.num_rows).equals(small)


def test_seed_changes_population(small_cfg):
    other = small_cfg.model_copy(update={"seed": small_cfg.seed + 1})
    assert not sample_population(other).equals(sample_population(small_cfg))


def test_arrival_parameters_do_not_perturb_archetypes_or_latents(small_cfg):
    base = sample_population(small_cfg)
    shifted = sample_population(_with_population(small_cfg, initial_fraction=0.2))
    cols = ["archetype", *[LATENT_PREFIX + n for n in small_cfg.latents]]
    assert shifted.select(cols).equals(base.select(cols))
    assert not shifted.column("created_at").equals(base.column("created_at"))


def test_schema_and_unique_account_ids(ref_pop, reference_cfg):
    assert ref_pop.schema.equals(account_schema(reference_cfg))
    account_ids = ref_pop.column("account_id").to_pylist()
    assert len(set(account_ids)) == len(account_ids) == reference_cfg.population.n_accounts


def test_created_at_within_arrival_window(ref_pop, reference_cfg):
    start = utc_midnight(reference_cfg.calendar.start)
    end = utc_midnight(reference_cfg.population.arrivals_end)
    assert all(start <= c < end for c in ref_pop.column("created_at").to_pylist())


def test_mixture_and_arrival_proportions_match_config(ref_pop, reference_cfg):
    archetypes = np.array(ref_pop.column("archetype").to_pylist())
    total = sum(a.weight for a in reference_cfg.archetypes.values())
    for name, archetype in reference_cfg.archetypes.items():
        assert abs((archetypes == name).mean() - archetype.weight / total) < 0.03

    initial_end = utc_midnight(reference_cfg.calendar.start) + timedelta(
        days=reference_cfg.population.initial_window_days
    )
    established = np.mean([c < initial_end for c in ref_pop.column("created_at").to_pylist()])
    assert abs(established - reference_cfg.population.initial_fraction) < 0.05


def test_latent_means_match_archetype_priors(ref_pop, reference_cfg):
    archetypes = np.array(ref_pop.column("archetype").to_pylist())
    for name, archetype in reference_cfg.archetypes.items():
        mask = archetypes == name
        for latent, mean in archetype.means.items():
            values = ref_pop.column(LATENT_PREFIX + latent).to_numpy()[mask]
            assert abs(values.mean() - mean) < 0.15


def test_parquet_metadata_records_config_fingerprint(tmp_path, small_cfg):
    path = tmp_path / "truth" / "accounts.parquet"
    write_parquet(sample_population(small_cfg), path, {"sim.config_fingerprint": small_cfg.fingerprint})
    assert read_parquet_metadata(path)["sim.config_fingerprint"] == small_cfg.fingerprint