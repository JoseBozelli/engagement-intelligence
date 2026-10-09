"""Roles: named groups of (event_type, actor) selectors (vocabulary-agnostic).

Targets and features refer to roles rather than raw event types, so a deployment maps its
own vocabulary onto roles and reuses target and feature definitions unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from engagement_intelligence.core.config import fingerprint, load_yaml_layers
from engagement_intelligence.core.vocabulary import EventVocabulary

Selector = tuple[str, str]


@dataclass(frozen=True)
class RoleMap:
    selectors: Mapping[str, frozenset[Selector]]

    def __post_init__(self) -> None:
        normalized = {
            str(role): frozenset((str(e), str(a)) for e, a in sels)
            for role, sels in self.selectors.items()
        }
        if not normalized:
            raise ValueError("at least one role is required")
        empty = sorted(r for r, s in normalized.items() if not s)
        if empty:
            raise ValueError(f"roles without selectors: {empty}")
        object.__setattr__(self, "selectors", MappingProxyType(normalized))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoleMap:
        return cls({role: [(s["event_type"], s["actor"]) for s in entries] for role, entries in data.items()})

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {
            role: [{"event_type": e, "actor": a} for e, a in sorted(sels)]
            for role, sels in sorted(self.selectors.items())
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def validate_against(self, vocabulary: EventVocabulary) -> None:
        for role, sels in self.selectors.items():
            for event_type, actor in sels:
                if event_type not in vocabulary.allowed_actors:
                    raise ValueError(f"role {role!r}: unknown event_type {event_type!r}")
                if actor not in vocabulary.allowed_actors[event_type]:
                    raise ValueError(f"role {role!r}: actor {actor!r} cannot emit {event_type!r}")

    def mask(self, event_types: np.ndarray, actors: np.ndarray, role: str) -> np.ndarray:
        """Boolean mask of rows whose (event_type, actor) belongs to `role`."""
        sels = self.selectors[role]
        return np.fromiter(((e, a) in sels for e, a in zip(event_types, actors)),
                           dtype=bool, count=len(event_types))


def load_roles(*paths: str | Path, vocabulary: EventVocabulary | None = None) -> RoleMap:
    roles = RoleMap.from_dict(load_yaml_layers(*paths))
    if vocabulary is not None:
        roles.validate_against(vocabulary)
    return roles