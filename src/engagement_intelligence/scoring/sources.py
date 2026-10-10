"""Event sources. Every source returns only events recorded at or before `as_of`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from engagement_intelligence.core.arrow_io import check_schema


class EventSource(Protocol):
    def events(self, as_of: datetime, account_ids: Sequence[str] | None = None) -> pa.Table: ...


class TableEventSource:
    """Event source over an in-memory Arrow table (e.g., an observable Parquet store)."""

    def __init__(self, table: pa.Table) -> None:
        check_schema(table)
        self._table = table

    @classmethod
    def from_parquet(cls, path: str | Path) -> TableEventSource:
        return cls(pq.read_table(path))

    def events(self, as_of: datetime, account_ids: Sequence[str] | None = None) -> pa.Table:
        mask = pc.less_equal(self._table["recorded_at"], pa.scalar(as_of, type=pa.timestamp("us", tz="UTC")))
        if account_ids is not None:
            wanted = pa.array(list(account_ids), pa.string())
            mask = pc.and_(mask, pc.is_in(self._table["account_id"], value_set=wanted))
        return self._table.filter(mask)