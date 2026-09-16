"""Explicit, revisable event-content evidence; proximity alone is never a binding."""

from __future__ import annotations

import copy

from ..schema import digest


def attach_event_contents(graph, reviews):
    """Materialize reviewed writing hyperedges without changing any local fact.

    A co-observed inscription is NOT a claim that this action created that text.
    Surface correspondence needs its own reviewed endpoint evidence. The caller
    must preserve review provenance; reference checks do not verify video truth.
    """
    out = copy.deepcopy(graph)
    facts = {f["fact_id"]: f for f in out["facts"]}
    instances = {i["instance_id"]: i for i in out["instances"]}
    edges, seen = [], set()
    for review in reviews:
        rid = review["review_id"]
        if rid in seen:
            raise ValueError("duplicate_content_review")
        seen.add(rid)
        if review.get("revoked") or review.get("verdict") != "supported":
            continue
        if not review.get("origin") or not review.get("basis"):
            raise ValueError("missing_content_review_provenance")
        event, text = facts[review["event_id"]], facts[review["text_id"]]
        if event["kind"] != "event" or text["kind"] != "text":
            raise ValueError("invalid_event_content_kinds")
        role_map = review["event_roles"]
        if set(role_map) != {"writer", "instrument", "surface"}:
            raise ValueError("incomplete_writing_contract")
        roles = {k: event["roles"][v] for k, v in role_map.items()}
        if set(role_map.values()) & set(event.get("unresolved_slots", [])):
            raise ValueError("unresolved_writing_role")
        if len(set(roles.values())) != 3 or instances[roles["instrument"]]["kind"] != "object":
            raise ValueError("invalid_writing_endpoints")
        if instances[roles["surface"]]["kind"] not in {"object", "region"}:
            raise ValueError("surface_must_be_local_object_or_region")
        owner = text["roles"].get("owner")
        if not owner or "owner" in text.get("unresolved_slots", []):
            raise ValueError("unresolved_text_surface")
        mids = set(review["evidence_ids"])
        if not mids or not mids <= out["evidence"].keys():
            raise ValueError("invalid_content_evidence")
        if not mids.intersection(event["joint_evidence"]) or not mids.intersection(
            text["joint_evidence"]
        ):
            raise ValueError("missing_event_or_text_evidence")
        correspondence = review.get("surface_correspondence")
        if owner != roles["surface"]:
            if (
                not correspondence
                or correspondence.get("verdict") != "supported"
                or correspondence.get("source") != owner
                or correspondence.get("target") != roles["surface"]
                or not correspondence.get("basis")
            ):
                raise ValueError("unproved_surface_correspondence")
            cmids = set(correspondence["evidence_ids"])
            if not cmids or not cmids <= mids:
                raise ValueError("invalid_surface_evidence")
            if any(
                not cmids.intersection(r["media_id"] for r in instances[i]["regions"])
                for i in (owner, roles["surface"])
            ):
                raise ValueError("surface_endpoints_not_grounded")
        relation = review["content_relation"]
        if relation not in {"co_observed_inscription", "written_content"}:
            raise ValueError("unknown_content_relation")
        if relation == "written_content":
            process = review.get("process_evidence_ids", [])
            if not process or not set(process) <= mids or not review.get("process_basis"):
                raise ValueError("creation_requires_process_evidence")
        edges.append(
            {
                "hyperedge_id": "writing:" + rid,
                "predicate": "write",
                "roles": roles,
                "content": {
                    "text_fact_id": text["fact_id"],
                    "value": text["value"],
                    "observed_surface": owner,
                    "relation": relation,
                },
                "source_fact_ids": [event["fact_id"], text["fact_id"]],
                "evidence_ids": sorted(mids),
                "observed_times": sorted({out["evidence"][m]["seconds"] for m in mids}),
                "surface_correspondence": copy.deepcopy(correspondence),
                "surface_owner": copy.deepcopy(text.get("owner_projections", {}).get("owner")),
                "review_id": rid,
                "origin": review["origin"],
                "support_scope": "explicit_review_with_structural_validation",
            }
        )
    out["event_content_reviews"] = copy.deepcopy(reviews)
    out["event_hyperedges"] = edges
    out.pop("snapshot_id", None)
    out["snapshot_id"] = digest(out)
    return out


def event_content_closure(facts, available, graph):
    """Complete seed occurrences once; shared attributes never trigger a graph walk."""
    chosen = {f["fact_id"]: f for f in facts}
    seeds = set(chosen)
    known = {f["fact_id"]: f for f in available}
    for edge in graph.get("event_hyperedges", []):
        ids = set(edge["source_fact_ids"])
        if ids & seeds and ids <= known.keys():
            for fid in ids - chosen.keys():
                chosen[fid] = known[fid]
    return list(chosen.values())


def attach_coobserved_attributes(graph):
    """Automatic, query-blind role/attribute bundles with exact local endpoints.

    This is a derived index over existing evidence, not a new VLM observation.
    Same global identity, similar descriptions or nearby times are insufficient.
    A shared evidence frame is required, and no text-creation claim is made.
    """
    out = copy.deepcopy(graph)
    attributes = {}
    for fact in graph["facts"]:
        owner = fact["roles"].get("owner")
        if (
            fact["kind"] in {"text", "state", "attribute"}
            and owner
            and not fact.get("unresolved_slots")
            and not fact.get("retrieval_quarantined")
        ):
            attributes.setdefault(owner, []).append(fact)
    edges = []
    for event in graph["facts"]:
        if event["kind"] != "event" or event.get("retrieval_quarantined"):
            continue
        attrs = []
        for role, endpoint in event["roles"].items():
            if role in event.get("unresolved_slots", []):
                continue
            for fact in attributes.get(endpoint, []):
                shared = sorted(set(event["joint_evidence"]) & set(fact["joint_evidence"]))
                if not shared:
                    continue
                attrs.append(
                    {
                        "event_role": role,
                        "local_owner": endpoint,
                        "attribute_fact_id": fact["fact_id"],
                        "predicate": fact["predicate"],
                        "value": fact["value"],
                        "shared_evidence_ids": shared,
                        "relation": "co_observed_attribute",
                    }
                )
        if not attrs:
            continue
        fids = sorted({event["fact_id"], *(a["attribute_fact_id"] for a in attrs)})
        edges.append(
            {
                "hyperedge_id": "coobserved:" + digest(fids)[:20],
                "predicate": event["predicate"],
                "roles": copy.deepcopy(event["roles"]),
                "role_attributes": attrs,
                "source_fact_ids": fids,
                "evidence_ids": sorted({m for a in attrs for m in a["shared_evidence_ids"]}),
                "origin": "exact_local_endpoint_and_shared_frame",
                "support_scope": "structural_coobservation_not_media_semantic_acceptance",
            }
        )
    # Idempotent regeneration preserves explicitly reviewed writing bindings.
    out["event_hyperedges"] = [
        e
        for e in graph.get("event_hyperedges", [])
        if e.get("origin") != "exact_local_endpoint_and_shared_frame"
    ] + edges
    out.pop("snapshot_id", None)
    out["snapshot_id"] = digest(out)
    return out
