"""Shared test utilities (importable from tests because tests/ is on sys.path)."""

import itertools
from datetime import datetime, timedelta, timezone

from engagement_intelligence.schemas.events import Event

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)  # a Monday
_seq = itertools.count()


def replace_config(model, path: str, value):
    """Copy of a frozen config with the nested attribute at dotted `path` replaced."""
    head, *rest = path.split(".")
    new = value if not rest else replace_config(getattr(model, head), ".".join(rest), value)
    return model.model_copy(update={head: new})


def ev(account: str, event_type: str, actor: str, day: float,
       recorded_day: float | None = None, ref: str | None = None) -> Event:
    """Reference-vocabulary event at T0 + day (12:00 UTC), optionally recorded later."""
    n = next(_seq)
    ts = T0 + timedelta(days=day, hours=12)
    recorded = T0 + timedelta(days=day if recorded_day is None else recorded_day, hours=12)
    ref_id = None if event_type == "account.created" else (ref or f"r{n}")
    return Event(event_id=f"e{n}", tenant_id="demo", account_id=account, event_type=event_type,
                 actor=actor, timestamp=ts, recorded_at=recorded, ref_id=ref_id)