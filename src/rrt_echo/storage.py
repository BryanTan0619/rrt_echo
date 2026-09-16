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


def export_run(memory: Memory, out: Path, *, status="partial", gaps=()):
    graph = memory.snapshot()
    # All journal data precedes the atomic reader-facing snapshot publication.
    atomic_json(out / "evidence" / "observations.json", [p.to_dict() for p in memory.packets])
    atomic_json(out / "ledger" / "identity.json", memory.identity.export())
    atomic_json(out / "views" / "storyline.json", storyline(graph))
    atomic_json(out / "hypergraph.json", graph)
    atomic_json(
        out / "status.json",
        {
            "status": status,
            "snapshot_id": graph["snapshot_id"],
            "gaps": list(gaps),
            "quality_passed": None,
        },
    )
    if "multimodal" in graph:
        from .rrt.hypergraph import validate_multimodal

        validate_multimodal(graph)
    return graph


def read_graph(path: Path) -> dict:
    """Load a frozen reader snapshot and reject changed/dangling references."""
    from .schema import SCHEMA_VERSION, digest

    graph = json.loads(Path(path).read_text())
    if graph.get("schema") != SCHEMA_VERSION:
        raise ValueError("unsupported_graph_schema")
    if graph.get("snapshot_id") != digest({k: v for k, v in graph.items() if k != "snapshot_id"}):
        raise ValueError("snapshot_digest_mismatch")
    instances = {i["instance_id"] for i in graph["instances"]}
    if len(instances) != len(graph["instances"]):
        raise ValueError("duplicate_graph_instance")
    fids = set()
    for fact in graph["facts"]:
        if fact["fact_id"] in fids:
            raise ValueError("duplicate_graph_fact")
        fids.add(fact["fact_id"])
        if not set(fact["roles"].values()) <= instances:
            raise ValueError("dangling_graph_role")
        mids = set(fact["joint_evidence"])
        mids.update(m for ids in fact["evidence_by_slot"].values() for m in ids)
        if not mids <= graph["evidence"].keys():
            raise ValueError("dangling_graph_evidence")
    if "multimodal" in graph:
        from .rrt.hypergraph import validate_multimodal

        validate_multimodal(graph)
    return graph
