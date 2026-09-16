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


def state_at(graph: dict, owner: str, property_name: str, seconds: float) -> list[dict]:
    """Return actual observations at this source time, not inferred persistence."""
    return [
        f
        for f in graph["facts"]
        if f["kind"] in {"state", "attribute", "text"}
        and f["predicate"] == property_name
        and f["roles"].get("owner") == owner
        and seconds in f["observed_times"]
    ]


def change_between(before: dict, after: dict) -> dict:
    """At least one change; no claim about exact time, count, cause or process."""
    if before["predicate"] != after["predicate"] or before["value"] == after["value"]:
        raise ValueError("different explicit values of the same property are required")
    a, b = before["resolved_roles"].get("owner"), after["resolved_roles"].get("owner")
    if (
        not a
        or not b
        or a["entity_id"] != b["entity_id"]
        or a["entity_version"] != b["entity_version"]
    ):
        raise ValueError("owner correspondence unresolved")
    for fact in (before, after):
        if fact["support"]["local_joint"]["status"] != "supported" or not fact["observed_times"]:
            raise ValueError("explicit state evidence required")
    start, end = max(before["observed_times"]), min(after["observed_times"])
    if start >= end:
        raise ValueError("unordered state observations")
    return {
        "kind": "change_between_observations",
        "bounds": [start, end],
        "rule": "S02_explicit_different_states",
        "premises": [before["fact_id"], after["fact_id"]],
        "identity_dependencies": sorted(
            set(before["dependencies"]["identity"] + after["dependencies"]["identity"])
        ),
        "exact_time": None,
        "cause": None,
        "media_audit_required": True,
    }
