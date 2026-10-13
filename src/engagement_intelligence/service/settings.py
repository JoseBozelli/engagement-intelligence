"""Service settings, loadable from environment variables (prefix EI_)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class ServiceSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    bundle_dir: Path
    events_path: Path
    store_path: Path
    trace_path: Path
    default_as_of: datetime | None = None  # None: current time (production); fixed value for replayed data

    @field_validator("default_as_of")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and (v.tzinfo is None or v.utcoffset() is None):
            raise ValueError("default_as_of must be timezone-aware")
        return v

    @classmethod
    def from_env(cls, prefix: str = "EI_") -> ServiceSettings:
        raw_as_of = os.environ.get(f"{prefix}DEFAULT_AS_OF")
        as_of = None
        if raw_as_of:
            as_of = datetime.fromisoformat(raw_as_of)
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
        return cls(
            bundle_dir=os.environ[f"{prefix}BUNDLE_DIR"],
            events_path=os.environ[f"{prefix}EVENTS_PATH"],
            store_path=os.environ[f"{prefix}STORE_PATH"],
            trace_path=os.environ[f"{prefix}TRACE_PATH"],
            default_as_of=as_of,
        )