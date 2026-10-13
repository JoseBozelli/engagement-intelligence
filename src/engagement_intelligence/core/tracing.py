# Provenance: copied verbatim from SignalGuard (github.com/JoseBozelli/signalguard),
# module signalguard/observability/tracing.py. Do not modify here; extend by composition
# (see service/observability.py).
"""
JSONL execution tracing for SignalGuard.

Persists one JSON line per investigation, capturing enough detail to
reconstruct what happened without re-running the pipeline: which
documentation chunks were retrieved, what was extracted (or why the
model abstained), which tool was selected, and the final verdict. This
is deliberately a thin, pure-I/O layer -- it takes already-computed
trace data (e.g. from pipeline.orchestrator.InvestigationTrace) and
writes it; it never calls an LLM or embedding API itself, so it is fully
testable without network access.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Union


def new_run_id() -> str:
    """A short, unique identifier for one evaluation/pipeline run."""
    return uuid.uuid4().hex[:12]


class JSONLTracer:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        """Append one trace record as a JSON line, with a logging timestamp added."""
        entry = {"logged_at": time.time(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every logged record back, in the order they were written."""
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]