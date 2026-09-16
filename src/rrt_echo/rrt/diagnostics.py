"""Query-blind observation loss audit and bounded repair candidates.

Missing events are not reconstructed from descriptions or QA. Diagnostics request
new visual evidence and do not mutate raw facts or certify semantic correctness.
"""

from __future__ import annotations

from itertools import combinations


def observation_clip_id(result):
    return result.get(
        "input_clip_id", result["observation"]["observation_id"].removesuffix(":verified-v7")
    )


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


def repair_plan(planned, results, limit=4):
    """Prioritize unobserved/empty windows, then unresolved endpoint bindings.

    The plan is bounded, deterministic and uses no question or answer input.
    Source intervals with no sampled frames need preprocessing, not hallucination.
    """
    by_obs = {observation_clip_id(r): r for r in results}
    issue_obs = {x["observation_id"] for x in audit_observations(results)}
    issue_ids = {
        observation_clip_id(r) for r in results if r["observation"]["observation_id"] in issue_obs
    }
    rows = []
    for clip in planned:
        result = by_obs.get(clip.clip_id)
        reason = (
            "not_processed"
            if result is None
            else "no_local_facts"
            if not result["observation"]["facts"]
            else "endpoint_or_role_gap"
            if clip.clip_id in issue_ids
            else None
        )
        if reason:
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "start": clip.target_start,
                    "end": clip.target_end,
                    "reason": reason,
                    "priority": 0
                    if result is None
                    else 1
                    if not result["observation"]["facts"]
                    else 2,
                }
            )
    return sorted(rows, key=lambda r: (r["priority"], r["start"]))[:limit]


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
    """Conditional consistency constraints from literal carry/hold occurrences.

    These do not independently certify video truth. Unresolved or copied-box roles
    cannot supply a constraint, and person/region pairs are not whole-person pairs.
    """
    instances = {i["instance_id"]: i for r in results for i in r["observation"]["instances"]}
    rows = []
    for result in results:
        for fact in result["observation"]["facts"]:
            if fact["kind"] != "event" or fact["predicate"].lower().strip() not in {
                "carry",
                "hold",
            }:
                continue
            roles = fact["roles"]
            for left_role, right_role in (
                ("carrier", "carried"),
                ("holder", "held"),
                ("agent", "patient"),
                ("agent", "theme"),
            ):
                if left_role not in roles or right_role not in roles:
                    continue
                left, right = roles[left_role], roles[right_role]
                if left == right or {left_role, right_role} & set(fact.get("unresolved_slots", [])):
                    continue
                if any(instances[key]["kind"] != "person" for key in (left, right)):
                    continue
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "source_fact_id": fact["fact_id"],
                        "roles": [left_role, right_role],
                        "evidence_ids": fact["joint_evidence"],
                        "rule": "literal_whole_person_carry_hold_is_irreflexive",
                        "semantic_support_audited": False,
                    }
                )
    return rows
