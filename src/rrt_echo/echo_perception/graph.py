"""Compile detailed observations and explicit links without narrative completion."""

from __future__ import annotations

import copy
import hashlib


def compile_graph(results, index):
    nodes, evidence, facts, proposals = {}, {}, [], []
    for result in results:
        obs = result["observation"]
        for m in [*obs["media"], *result.get("context_media", [])]:
            evidence[m["media_id"]] = m
        for instance in obs["instances"]:
            key = instance["instance_id"]
            if key in nodes:
                raise ValueError("duplicate_local_endpoint")
            nodes[key] = {
                **copy.deepcopy(instance),
                "origin": "local_observation",
                "observation_id": obs["observation_id"],
            }
        facts.extend(
            {
                **copy.deepcopy(f),
                "observation_id": obs["observation_id"],
                "evidence_family": obs["observation_id"],
            }
            for f in obs["facts"]
        )
        proposals.extend(copy.deepcopy(result["link_proposals"]))
        for ref in result.get("reference_evidence", []):
            evidence[ref["media_id"]] = ref
            source = ref["source_region"]
            endpoint = source["instance_id"]
            if source.get("origin") == "local_observation":
                # The reference is a rendered view of an existing observation, not
                # a fresh identity endpoint. Validate provenance before adding its
                # presentation alias to this derived graph (raw observations stay intact).
                if endpoint not in nodes:
                    raise ValueError("missing_reference_observation")
                existing = nodes[endpoint]
                original = evidence.get(source["source_media_id"])
                if (
                    existing["kind"] != ref["kind"]
                    or original is None
                    or original["sha256"] != source["source_sha256"]
                    or {"media_id": source["source_media_id"], "box": source["box"]}
                    not in existing["regions"]
                ):
                    raise ValueError("reference_source_mismatch")
                alias = {"media_id": ref["media_id"], "box": source["box"]}
                if alias not in existing["regions"]:
                    existing["regions"].append(alias)
            else:
                node = {
                    "instance_id": endpoint,
                    "kind": ref["kind"],
                    "origin": "coarse_reference_anchor",
                    "regions": [{"media_id": ref["media_id"], "box": source["box"]}],
                    "candidate_id": source["candidate_id"],
                }
                if endpoint in nodes:
                    if nodes[endpoint] != node:
                        raise ValueError("reference_endpoint_changed")
                else:
                    nodes[endpoint] = node
    # All IDs in this artifact must resolve. Coarse episode assertions are not facts.
    seen = set()
    for f in facts:
        if f["fact_id"] in seen:
            raise ValueError("duplicate_fact_id")
        seen.add(f["fact_id"])
        if not set(f["roles"].values()) <= nodes.keys():
            raise ValueError("dangling_fact_endpoint")
        if not set(f["joint_evidence"]) <= evidence.keys():
            raise ValueError("dangling_fact_evidence")
    parent = {x: x for x in nodes}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    valid, invalid = [], []
    for p in proposals:
        if (
            not {p["source"], p["target"]} <= nodes.keys()
            or not set(p["evidence_ids"]) <= evidence.keys()
        ):
            invalid.append({**p, "status": "unresolved", "gap_reason": "dangling_link_reference"})
            continue
        left, right = nodes[p["source"]], nodes[p["target"]]
        review = p.get("focused_review", {})
        if (
            p["relation"] == "part_of"
            and review.get("source_category") == "body_part"
            and review.get("target_category") in {"object", "background"}
        ):
            invalid.append(
                {**p, "status": "unresolved", "gap_reason": "body_part_owner_type_mismatch"}
            )
            continue
        ids = set(p["evidence_ids"])
        kinds_ok = (
            left["kind"] in {"person", "object"} and left["kind"] == right["kind"]
            if p["relation"] in {"same_identity", "continues"}
            else p["relation"] == "part_of"
            and left["kind"] == "region"
            and right["kind"] in {"person", "object"}
        )
        anchors_ok = all(
            ids & {r["media_id"] for r in endpoint["regions"]} for endpoint in (left, right)
        )
        if p["source"] == p["target"] or not kinds_ok or not anchors_ok:
            invalid.append({**p, "status": "unresolved", "gap_reason": "invalid_link_endpoints"})
            continue
        valid.append(p)
        if p["relation"] == "same_identity" and p["verdict"] == "supported":
            parent[root(p["target"])] = root(p["source"])
    # Detect conflict over entire proposed components BEFORE accepting any merge.
    # No order-dependent "first edge wins", no coarse candidate transitive closure.
    bad_components = {
        root(p["source"])
        for p in valid
        if p["relation"] == "same_identity"
        and p["verdict"] == "contradicted"
        and root(p["source"]) == root(p["target"])
    }
    # A transitive identity path must not collapse two independently grounded
    # whole-person roles of a literal carry/hold occurrence. This is a veto on
    # acceptance, not a new claim of visually verified different identities.
    for constraint in index.get("separation_constraints", []):
        a, b = constraint["left"], constraint["right"]
        if a in nodes and b in nodes and root(a) == root(b):
            bad_components.add(root(a))
    links = []
    accepted = []
    for p in valid:
        if p["verdict"] == "unresolved":
            status, reason = "unresolved", "model_evidence_insufficient"
        elif p["relation"] == "same_identity" and root(p["source"]) in bad_components:
            status, reason = "disputed", "identity_component_conflict"
        elif p["verdict"] == "contradicted":
            status, reason = "contradicted", "explicit_model_counterevidence"
        else:
            status, reason = "accepted", "explicit_link_and_reference_checks"
        row = {**p, "status": status, "acceptance_basis": reason, "semantic_support_audited": False}
        links.append(row)
        if status == "accepted" and p["relation"] == "same_identity":
            accepted.append(row)
    # Rebuild only nonconflicting accepted components.
    parent = {x: x for x, n in nodes.items() if n["kind"] in {"person", "object"}}
    for p in accepted:
        parent[root(p["target"])] = root(p["source"])
    groups = {}
    for x in parent:
        groups.setdefault(root(x), []).append(x)
    entities, resolved = [], {}
    for members in sorted(groups.values(), key=lambda g: sorted(g)):
        members.sort()
        if len(members) < 2:
            continue  # A disconnected local instance is not a confirmed new person.
        entity_id = "entity:" + hashlib.sha256("|".join(members).encode()).hexdigest()[:16] + ":v1"
        deps = [p["proposal_id"] for p in accepted if p["source"] in members]
        entities.append(
            {
                "entity_id": entity_id,
                "members": members,
                "kind": nodes[members[0]]["kind"],
                "identity_links": deps,
                "status": "linked",
                "semantic_support_audited": False,
            }
        )
        resolved.update({m: entity_id for m in members})
    # Ownership is separate from identity. Conflicting owners never create merges.
    owners = {}
    for p in links:
        if p["relation"] == "part_of" and p["status"] == "accepted":
            owners.setdefault(p["source"], set()).add(resolved.get(p["target"], p["target"]))
    for p in links:
        if p["relation"] == "part_of" and len(owners.get(p["source"], ())) > 1:
            p.update(status="disputed", acceptance_basis="conflicting_region_owners")
    owner_links = {}
    for p in links:
        if p["relation"] == "part_of" and p["status"] == "accepted":
            owner_links.setdefault(p["source"], []).append(p)
    identity_deps = {e["entity_id"]: e["identity_links"] for e in entities}
    projections = []
    for fact in facts:
        role_view = {}
        for role, endpoint in fact["roles"].items():
            role_view[role] = {
                "local_endpoint": endpoint,
                "entity_version": resolved.get(endpoint),
                "identity_links": identity_deps.get(resolved.get(endpoint), []),
                "identity_status": "linked" if endpoint in resolved else "unresolved",
                "owner_links": [p["proposal_id"] for p in owner_links.get(endpoint, [])],
                "evidence_ids": fact["evidence_by_slot"][role],
            }
        projections.append(
            {
                "fact_id": fact["fact_id"],
                "kind": fact["kind"],
                "predicate": fact["predicate"],
                "value": fact["value"],
                "participants": role_view,
                "evidence_family": fact["evidence_family"],
                "evidence_by_slot": fact["evidence_by_slot"],
                "observed_media_ids": fact["observed_media_ids"],
                "joint_evidence": fact["joint_evidence"],
                "unresolved_slots": fact["unresolved_slots"],
                "local_joint_support": "model_asserted; structure_checked",
                "semantic_support_audited": False,
            }
        )
    links.extend(invalid)
    gaps = [g for r in results for g in r.get("coverage_gaps", [])]
    graph = {
        "schema_version": "rrt-context-graph-v1",
        "video_id": index["video_id"],
        "nodes": list(nodes.values()),
        "facts": facts,
        "links": links,
        "entities": entities,
        "fact_views": projections,
        "evidence": list(evidence.values()),
        "coverage_gaps": gaps,
        "coarse_index_version": index["version"],
        "coarse_assertions_committed": False,
        "semantic_support_audited": False,
    }
    return graph


def storyline(graph):
    """A deterministic graph projection. No coarse summary or generated transitions."""
    media = {m["media_id"]: m for m in graph["evidence"]}

    def seconds(mid):
        m = media[mid]
        return m["pts"] * m["time_base"][0] / m["time_base"][1]

    rows = []
    for f in graph["fact_views"]:
        times = sorted({seconds(mid) for mid in f["observed_media_ids"]})
        rows.append(
            {
                **f,
                "observed_times": times,
                "extent": [times[0], times[-1]] if times else None,
                "continuous_validity_inferred": False,
            }
        )
    return sorted(
        rows, key=lambda f: (f["extent"][0] if f["extent"] else float("inf"), f["fact_id"])
    )
