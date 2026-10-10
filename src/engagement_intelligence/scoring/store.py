"""Assessment stores. The JSON Lines store is an append-only log, one assessment per line."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from engagement_intelligence.scoring.schemas import EngagementAssessment


class AssessmentStore(Protocol):
    def append(self, assessments: Iterable[EngagementAssessment]) -> None: ...
    def history(self, account_id: str) -> list[EngagementAssessment]: ...
    def latest_scored_before(self, account_ids: Sequence[str],
                             before: datetime) -> dict[str, EngagementAssessment]: ...


class JsonlAssessmentStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._by_account: dict[str, list[EngagementAssessment]] = defaultdict(list)
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._index(EngagementAssessment.model_validate_json(line))

    def _index(self, assessment: EngagementAssessment) -> None:
        self._by_account[assessment.account_id].append(assessment)

    def append(self, assessments: Iterable[EngagementAssessment]) -> None:
        assessments = list(assessments)
        if not assessments:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for a in assessments:
                handle.write(a.model_dump_json() + "\n")
                self._index(a)

    def history(self, account_id: str) -> list[EngagementAssessment]:
        return sorted(self._by_account.get(account_id, []), key=lambda a: a.assessed_at)

    def latest_scored_before(self, account_ids: Sequence[str],
                             before: datetime) -> dict[str, EngagementAssessment]:
        latest = {}
        for account in account_ids:
            prior = [a for a in self._by_account.get(account, ())
                     if a.status == "scored" and a.assessed_at < before]
            if prior:
                latest[account] = max(prior, key=lambda a: a.assessed_at)
        return latest