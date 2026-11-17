"""Portable, atomic run artifacts. No parallel 'latest' memory representation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .memory import Memory


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def storyline(graph: dict) -> dict:
    """A graph projection, not an LLM-generated narrative."""
    return {
        "snapshot_id": graph["snapshot_id"],
        "events": [
            {
                "fact_id": f["fact_id"],
                "predicate": f["predicate"],
                "roles": f["resolved_roles"],
                "observed_times": f["observed_times"],
                "support": f["support"],
                "evidence_by_slot": f["evidence_by_slot"],
                "dependencies": f["dependencies"],
            }
            for f in sorted(
                graph["facts"], key=lambda f: (f["observed_times"] or [float("inf")], f["fact_id"])
            )
            if f["kind"] == "event"
        ],
        "relations": graph["relations"],
    }
