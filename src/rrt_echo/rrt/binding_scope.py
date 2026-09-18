"""Evidence-preserving scope primitives; never infer identity from descriptions."""

from __future__ import annotations

import re

CLOCK = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})"


def video_time_ranges(stem, tolerance=5):
    def seconds(groups):
        h, m, s = groups
        return int(h or 0) * 3600 + int(m) * 60 + int(s)

    # A stated playback interval takes precedence over individual clock readings
    # inside that interval's content (e.g. CCTV times narrated by a documentary).
    spans = list(re.finditer(CLOCK + r"\s*(?:-|–|—|to|through)\s*" + CLOCK, stem))
    if spans:
        return tuple(
            (max(0, seconds(m.groups()[:3]) - tolerance), seconds(m.groups()[3:]) + tolerance)
            for m in spans
            if seconds(m.groups()[:3]) <= seconds(m.groups()[3:])
        )
    return tuple(
        (max(0, seconds(m.groups()) - tolerance), seconds(m.groups()) + tolerance)
        for m in re.finditer(r"(?<!\d)" + CLOCK + r"(?!\d)", stem)
    )


def text_companions(fact, available):
    """Co-observed text on a locally established owner; never concatenate values.

    An identity shared across distant events is insufficient for this closure.
    Each fragment keeps its original surface, evidence and observation.
    """
    if fact["kind"] != "text":
        return []

    def owner(f):
        projection = f.get("owner_projections", {}).get("owner")
        return (
            tuple(sorted(projection["local_owner_endpoints"]))
            if projection
            else (f["roles"].get("owner"),)
        )

    times = fact["observed_times"]
    if not times:
        return []
    return [
        f
        for f in available
        if f["kind"] == "text"
        and f["dependencies"]["observation"] == fact["dependencies"]["observation"]
        and owner(f) == owner(fact)
        and f["observed_times"]
        and min(f["observed_times"]) <= max(times) + 2
        and max(f["observed_times"]) >= min(times) - 2
    ]


def binding_view(payload):
    """Keep descriptions next to values, without changing IDs or owner claims."""
    instances = {i["instance_id"]: i for i in payload.get("instances", [])}
    return [
        {
            "fact_id": f["fact_id"],
            "value": f.get("value"),
            "local_owner": f["roles"].get("owner"),
            "observed_surface": instances.get(f["roles"].get("owner"), {}).get("description", ""),
            "owner_projection": f.get("owner_projections", {}).get("owner"),
            "observed_times": f["observed_times"],
        }
        for f in payload.get("facts", [])
        if f["kind"] in {"text", "attribute", "state"}
    ]


def state_sequence_view(payload, graph):
    """Per-entity temporal chains of observed state/attribute facts for payload entities.

    Each sequence is an ordered list of observations, never an inferred transition.
    """
    seq_by_entity = {s["entity_id"]: s for s in graph.get("state_sequences", [])}
    entities = {
        r["entity_id"]
        for f in payload.get("facts", [])
        for r in f.get("resolved_roles", {}).values()
    }
    entities.update(
        p["owner_entity"]
        for f in payload.get("facts", [])
        for p in f.get("owner_projections", {}).values()
    )
    return [seq_by_entity[e] for e in sorted(entities) if e in seq_by_entity]
