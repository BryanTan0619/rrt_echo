"""Query-blind observation loss audit and bounded repair candidates.

Missing events are not reconstructed from descriptions or QA. Diagnostics request
new visual evidence and do not mutate raw facts or certify semantic correctness.
"""

from __future__ import annotations

from itertools import combinations


def audit_observations(results):
    issues = []
    for result in results:
        obs = result["observation"]
        instances = {i["instance_id"]: i for i in obs["instances"]}
        for fact in obs["facts"]:
            if fact.get("unresolved_slots"):
                issues.append(
                    {
                        "observation_id": obs["observation_id"],
                        "fact_id": fact["fact_id"],
                        "reason": "unresolved_roles",
                        "slots": fact["unresolved_slots"],
                    }
                )
            if fact["kind"] != "event":
                continue
            for (ra, ia), (rb, ib) in combinations(fact["roles"].items(), 2):
                if (
                    ia == ib
                    or instances[ia]["kind"] not in {"person", "object"}
                    or instances[ib]["kind"] != instances[ia]["kind"]
                ):
                    continue
                a = {(r["media_id"], tuple(r["box"])) for r in instances[ia]["regions"]}
                b = {(r["media_id"], tuple(r["box"])) for r in instances[ib]["regions"]}
                if a & b:
                    issues.append(
                        {
                            "observation_id": obs["observation_id"],
                            "fact_id": fact["fact_id"],
                            "reason": "distinct_roles_share_same_box",
                            "roles": [ra, rb],
                            "endpoints": [ia, ib],
                            "requires_visual_review": True,
                        }
                    )
    return issues


def replacement_constraints(result):
    """Explicit replacement roles imply distinct physical objects, not new states.

    This depends on the observed replacement fact being correct. The derived link
    keeps its source fact and can be revoked through the normal identity journal.
    """
    import hashlib

    obs = result["observation"]
    instances = {i["instance_id"]: i for i in obs["instances"]}
    links = []
    for fact in obs["facts"]:
        if fact["kind"] != "event" or fact["predicate"] != "replace":
            continue
        old, new = fact["roles"].get("patient"), fact["roles"].get("theme")
        if (
            old == new
            or not old
            or not new
            or {"patient", "theme"} & set(fact.get("unresolved_slots", []))
        ):
            continue
        if any(instances[i]["kind"] != "object" for i in (old, new)):
            continue
        evidence = set(fact["joint_evidence"])
        for role in ("patient", "theme"):
            evidence.update(fact["evidence_by_slot"].get(role, []))
        if any(
            not evidence.intersection(r["media_id"] for r in instances[i]["regions"])
            for i in (old, new)
        ):
            continue
        key = hashlib.sha256((fact["fact_id"] + old + new).encode()).hexdigest()[:20]
        links.append(
            {
                "proposal_id": "replacement:" + key,
                "relation": "same_identity",
                "source": old,
                "target": new,
                "verdict": "contradicted",
                "evidence_ids": sorted(evidence),
                "basis": "The explicit replace event identifies patient as the removed item and theme as its replacement.",
                "origin": "event_role_constraint",
                "source_fact_ids": [fact["fact_id"]],
                "semantic_support_audited": False,
            }
        )
    return links


def independent_person_anchors(results):
    """Find person-role endpoints grounded only by a copied composite box.

    This is an insufficiency check, not evidence that two people are different.
    An independently grounded sighting can still support a later correspondence.
    """
    instances = {i["instance_id"]: i for r in results for i in r["observation"]["instances"]}
    duplicated = {}
    issues = audit_observations(results)
    for issue in issues:
        if issue["reason"] != "distinct_roles_share_same_box":
            continue
        left, right = (instances[key] for key in issue["endpoints"])
        if left["kind"] != "person" or right["kind"] != "person":
            continue
        a = {(r["media_id"], tuple(r["box"])) for r in left["regions"]}
        b = {(r["media_id"], tuple(r["box"])) for r in right["regions"]}
        for key in issue["endpoints"]:
            duplicated.setdefault(key, set()).update(a & b)
    return {
        key: {
            r["media_id"]
            for r in instances[key]["regions"]
            if (r["media_id"], tuple(r["box"])) not in copied
        }
        for key, copied in duplicated.items()
    }


def whole_person_separations(results):
    """Return hard same-identity vetoes from co-observed irreflexive roles.

    A veto prevents a false merge. It is not exported as verified evidence that
    two unresolved endpoints are globally different people.
    """
    instances = {i["instance_id"]: i for r in results for i in r["observation"]["instances"]}
    irreflexive = {
        frozenset(("actor", "recipient")), frozenset(("agent", "patient")),
        frozenset(("attacker", "victim")), frozenset(("giver", "recipient")),
        frozenset(("speaker", "listener")), frozenset(("carrier", "carried")),
        frozenset(("holder", "held")),
    }
    rows, seen = [], set()
    # A shared window does not prove two sightings differ. Two separately
    # localized whole people in the same frame do provide a merge veto.
    from collections import defaultdict

    coframe = defaultdict(list)
    for instance in instances.values():
        if instance.get("kind") != "person":
            continue
        for region in instance.get("regions", []):
            box = region.get("box", [])
            if len(box) == 4 and (box[2] - box[0]) * (box[3] - box[1]) >= 0.01:
                coframe[region["media_id"]].append((instance["instance_id"], box))

    def overlap(a, b):
        width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        intersection = width * height
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1e-9)

    for media_id, sightings in coframe.items():
        for index, (left, left_box) in enumerate(sightings):
            for right, right_box in sightings[index + 1:]:
                if left == right or overlap(left_box, right_box) >= 0.5:
                    continue
                key = (frozenset((left, right)), media_id)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "left": left, "right": right, "source_media_id": media_id,
                    "evidence_ids": [media_id],
                    "rule": "distinct_whole_person_regions_in_same_frame_cannot_merge",
                    "semantic_support_audited": False,
                })
    for result in results:
        for fact in result["observation"]["facts"]:
            if fact["kind"] != "event":
                continue
            roles = fact["roles"]
            pairs = [(a, b) for a in roles for b in roles if a < b and frozenset((a, b)) in irreflexive]
            if fact["predicate"].lower().strip() in {"carry", "hold"} and {"agent", "theme"} <= roles.keys():
                pairs.append(("agent", "theme"))
            for left_role, right_role in pairs:
                left, right = roles[left_role], roles[right_role]
                if left == right or {left_role, right_role} & set(fact.get("unresolved_slots", [])):
                    continue
                if any(instances[key]["kind"] != "person" for key in (left, right)):
                    continue
                key = (frozenset((left, right)), fact["fact_id"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "left": left, "right": right, "source_fact_id": fact["fact_id"],
                    "roles": [left_role, right_role], "evidence_ids": fact["joint_evidence"],
                    "rule": "coobserved_irreflexive_person_roles_cannot_merge",
                    "semantic_support_audited": False,
                })
    return rows
