"""
Layered YAML configuration and stable fingerprints (domain-neutral).

Layers are deep-merged in order: mappings merge key by key, every other value (including lists) is replaced by the
later layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out

def load_yaml_layers(*paths: str | Path) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one config path is required")
    merged: dict[str, Any] = {}
    for path in paths:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        merged = deep_merge(merged, data)
    return merged

def fingerprint(data: Any) -> str:
    """Short, stable hash of JSON-serialization data (key order does not matter)."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]