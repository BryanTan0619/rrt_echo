"""Dynamic output schema + cross-field checks; structural validity is not visual truth."""

import copy
from dataclasses import asdict

from jsonschema import Draft202012Validator


def arr(item, maximum=64, minimum=0):
    return {
        "type": "array",
        "items": item,
        "minItems": minimum,
        "maxItems": maximum,
        "uniqueItems": True,
    }


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def string(maximum=160):
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def enum(values):
    return {"type": "string", "enum": list(values)}


def joint_schema(target_ids, all_ids, reference_ids=(), max_instances=16, max_facts=24):
    local = [f"i{k}" for k in range(max_instances)]
    region = obj(
        {
            "media_id": enum(target_ids),
            "box": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
        }
    )
    evidence = arr(enum(target_ids), 64, 1)
    instance = obj(
        {
            "instance_id": enum(local),
            "kind": enum(["person", "object", "region"]),
            "description": string(),
            "regions": arr(region, 4, 1),
        }
    )
    fact = obj(
        {
            "fact_id": string(40),
            "kind": enum(["event", "state", "attribute", "text"]),
            "predicate": string(64),
            "roles": {"type": "object", "additionalProperties": enum(local), "minProperties": 1},
            "value": {"type": ["string", "null"], "maxLength": 256},
            "evidence_by_slot": {"type": "object", "additionalProperties": evidence},
            "joint_evidence": evidence,
            "observed_media_ids": evidence,
            "unresolved_slots": arr(string(40), 16),
        }
    )
    link = obj(
        {
            "source": enum(local),
            "target": enum(local + list(reference_ids)),
            "relation": enum(["same_identity", "part_of", "continues"]),
            "verdict": enum(["supported", "contradicted", "unresolved"]),
            "evidence_ids": arr(enum(all_ids), 12, 1),
            "basis": string(240),
        }
    )
    gap = obj({"reason": string(160), "media_ids": arr(enum(target_ids), 16)})
    return obj(
        {
            "instances": arr(instance, max_instances),
            "facts": arr(fact, max_facts),
            "links": arr(link, 24),
            "coverage_gaps": arr(gap, 12),
        }
    )


def validate(raw, schema, refs_by_id=None):
    Draft202012Validator(schema).validate(raw)
    instances = {i["instance_id"]: i for i in raw["instances"]}
    if len(instances) != len(raw["instances"]):
        raise ValueError("duplicate_instance_id")
    if len({f["fact_id"] for f in raw["facts"]}) != len(raw["facts"]):
        raise ValueError("duplicate_fact_id")
    regions = {k: {r["media_id"] for r in i["regions"]} for k, i in instances.items()}
    for i in instances.values():
        for r in i["regions"]:
            a, b, c, d = r["box"]
            if not a < c or not b < d:
                raise ValueError("invalid_box_extent")
    for f in raw["facts"]:
        slots = {"predicate", *f["roles"]}
        if f["kind"] != "event":
            slots.add("value")
            if "owner" not in f["roles"]:
                raise ValueError("missing_property_owner")
            if f["value"] is None:
                raise ValueError("missing_property_value")
        if set(f["evidence_by_slot"]) != slots:
            raise ValueError("slot_evidence_mismatch")
        if set(f["unresolved_slots"]) & set(f["roles"]):
            raise ValueError("resolved_slot_marked_unresolved")
        joint = set(f["joint_evidence"])
        if not set(f["observed_media_ids"]) <= joint:
            raise ValueError("observed_outside_joint_evidence")
        for slot, ids in f["evidence_by_slot"].items():
            if not set(ids) <= joint:
                raise ValueError("slot_outside_joint_evidence")
        for role, ref in f["roles"].items():
            if ref not in instances:
                raise ValueError("undeclared_role_endpoint")
            if not set(f["evidence_by_slot"][role]) & regions[ref]:
                raise ValueError("role_missing_endpoint_anchor")
    refs = refs_by_id or {}
    for link in raw["links"]:
        a, b = link["source"], link["target"]
        ids = set(link["evidence_ids"])
        if a not in instances or a == b:
            raise ValueError("invalid_link_source")
        if b not in instances and b not in refs:
            raise ValueError("invalid_link_target")
        other_kind = instances[b]["kind"] if b in instances else refs[b]["kind"]
        if link["relation"] in {"same_identity", "continues"}:
            if instances[a]["kind"] == "region" or instances[a]["kind"] != other_kind:
                raise ValueError("identity_kind_mismatch")
        elif instances[a]["kind"] != "region" or other_kind not in {"person", "object"}:
            raise ValueError("invalid_part_of_types")
        other_media = regions[b] if b in instances else set(refs[b]["media_ids"])
        if not ids & regions[a] or not ids & other_media:
            raise ValueError("link_missing_endpoint_evidence")
    return raw


def materialize(raw, clip, alias_to_source, refs_by_id):
    """New transport envelope; existing packet receives only target-local facts."""
    d = copy.deepcopy(raw)
    prefix = clip.clip_id + ":"
    mapping = {i["instance_id"]: prefix + i["instance_id"] for i in d["instances"]}
    for i in d["instances"]:
        i["instance_id"] = mapping[i["instance_id"]]
        for r in i["regions"]:
            r["media_id"] = alias_to_source[r["media_id"]]
            r["box"] = [v / 1000 for v in r["box"]]
    for f in d["facts"]:
        f["fact_id"] = prefix + f["fact_id"]
        f["roles"] = {k: mapping[v] for k, v in f["roles"].items()}
        f["evidence_by_slot"] = {
            k: [alias_to_source[x] for x in v] for k, v in f["evidence_by_slot"].items()
        }
        for k in ["joint_evidence", "observed_media_ids"]:
            f[k] = [alias_to_source[x] for x in f[k]]
    links = d.pop("links")
    gaps = d.pop("coverage_gaps")
    for n, link in enumerate(links):
        link["source"] = mapping[link["source"]]
        target = link["target"]
        link["target"] = mapping[target] if target in mapping else refs_by_id[target]["instance_id"]
        link["evidence_ids"] = [alias_to_source[x] for x in link["evidence_ids"]]
        link["proposal_id"] = prefix + f"link:{n}"
        link["status"] = "proposed"  # model verdict never grants commitment
    for g in gaps:
        g["media_ids"] = [alias_to_source[x] for x in g["media_ids"]]
    return {
        "observation": {
            "observation_id": clip.clip_id,
            "video_id": clip.video_id,
            "shot_id": str(clip.shot_id),
            "media": [asdict(m) for m in clip.media if m.media_id in clip.target_ids],
            **d,
        },
        "link_proposals": links,
        "coverage_gaps": gaps,
        "context_media": [asdict(m) for m in clip.media if m.media_id not in clip.target_ids],
        "global_identity_commitment": False,
        "semantic_support_audited": False,
    }
