"""
Deterministic, opaque identifiers.

Generated IDs are keyed hashes: stable for (seed, account, sequence) but encoding no ordering, timing, or outcoome
information. Validation of incoming IDs is deliberately permissive (is_valid_id) so events from other systems --
UUIDs, prefixed keys -- are accepted without reformatting.
"""

from __future__ import annotations

import hashlib
import re

_DIGEST_BYTES = 10  # 20 hex chars = 80 bits; negligible collision risk at < 1e7 IDs
_SEP = b"\x1f"

GENERIC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
EVENT_ID_RE = re.compile(r"ev_[0-9a-f]{20}")
REF_ID_RE = re.compile(r"ref_[0-9a-f]{20}")
ACCOUNT_ID_RE = re.compile(r"acc_[0-9]{6}")

def _digest(namespace: str, *parts: object) -> str:
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    h.update(namespace.encode())
    for part in parts:
        h.update(_SEP)
        h.update(str(part).encode())
    return h.hexdigest()

def make_event_id(seed: int, account_id: str, seq: int) -> str:
    return "ev_" + _digest("event", seed, account_id, seq)

def make_ref_id(seed: int, account_id: str, seq: int) -> str:
    return "ref_" + _digest("ref", seed, account_id, seq)

def make_account_id(index: int) -> str:
    """Readable synthetic account ID. The index must carry no information: generators assing latent properties
    independently of it."""
    if not 0 <= index < 1_000_000:
        raise ValueError(f"account index out of range: {index}")
    return f"acc_{index:06d}"

def is_valid_id(value: str) -> bool:
    return GENERIC_ID_RE.fullmatch(value) is not None