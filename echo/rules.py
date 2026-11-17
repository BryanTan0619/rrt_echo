"""Explicit, conservative temporal inference. No automatic state persistence."""

from __future__ import annotations


def temporal_relations(facts: list[dict]) -> list[dict]:
    """Sparse adjacent order only when both event boundaries are supported.

    Observed-point extents are never substituted for duration. More rules can
    be added with tests; sharing an object or time does not imply causation.
    """
    bounded = sorted(
        (
            f
            for f in facts
            if f["kind"] == "event"
            and f["time_bounds"]
            and f["support"]["local_joint"]["status"] == "supported"
        ),
        key=lambda f: (f["time_bounds"][0], f["fact_id"]),
    )
    result = []
    for left, right in zip(bounded, bounded[1:]):
        if left["time_bounds"][1] <= right["time_bounds"][0]:
            result.append(
                {
                    "relation_id": f"before:{left['fact_id']}:{right['fact_id']}",
                    "type": "before",
                    "source": left["fact_id"],
                    "target": right["fact_id"],
                    "rule": "T01_supported_boundaries",
                    "evidence_ids": sorted(
                        set(left["boundary_evidence"]) | set(right["boundary_evidence"])
                    ),
                    "assessment": "structural",
                    "media_audit_required": True,
                }
            )
    return result
