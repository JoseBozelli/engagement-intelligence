"""Order-invariant deterministic randomness.

Pattern: seeded synthetic-data methodology from SignalGuard (github.com/JoseBozelli/signalguard).
Extension: each (stream, index) pair gets an independent child stream via SeedSequence spawn keys,
so an entity's draws do not depend on generation order, parallelism, or on other name streams."""

from __future__ import annotations

import hashlib

import numpy as np

def stream_key(name: str) -> int:
    """Stable 32-bit key for a stream name (Python's hash() is salted per process)."""
    if not name:
        raise ValueError("stream name must be non-empty")
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")

def rng_for(master_seed: int, stream: str, index: int = 0) -> np.random.Generator:
    if master_seed < 0 or index < 0:
        raise ValueError("master_seed and index must be non-negative")
    seq = np.random.SeedSequence(entropy=master_seed, spawn_key=(stream_key(stream), index))
    return np.random.default_rng(seq)