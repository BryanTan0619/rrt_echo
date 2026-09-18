"""Bounded transport schemas; limits are exported and do not certify coverage."""


def array(items, maximum):
    return {"type": "array", "items": items, "maxItems": maximum}


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def instance_schema(media_ids):
    region = obj(
        {
            "media_id": {"type": "string", "enum": list(media_ids)},
            "box": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
        }
    )
    instance = obj(
        {
            "instance_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["person", "object", "region"]},
            "description": {"type": "string"},
            "regions": array(region, 4),
        }
    )
    return obj({"instances": array(instance, 12)})


def fact_schema(media_ids, instance_ids):
    ids = array({"type": "string", "enum": list(media_ids)}, len(media_ids))
    fact = obj(
        {
            "fact_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["event", "state", "attribute", "text"]},
            "predicate": {"type": "string"},
            "roles": {
                "type": "object",
                "additionalProperties": (
                    {"type": "string", "enum": list(instance_ids)}
                    if instance_ids is not None
                    else {"type": "string"}
                ),
            },
            "value": {"type": ["string", "null"]},
            "evidence_by_slot": {"type": "object", "additionalProperties": ids},
            "joint_evidence": ids,
            "observed_media_ids": ids,
            "unresolved_slots": array({"type": "string"}, 12),
        }
    )
    return obj({"facts": array(fact, 20)})


def joint_schema(media_ids):
    """One observation request; cross-field participant references validate after decode."""
    instances = instance_schema(media_ids)["properties"]["instances"]
    local_ids = [f"i{k}" for k in range(12)]
    instances["items"]["properties"]["instance_id"] = {"type": "string", "enum": local_ids}
    instances["items"]["properties"]["description"]["maxLength"] = 160
    facts = fact_schema(media_ids, local_ids)["properties"]["facts"]
    return obj({"instances": instances, "facts": facts})
