from copy import deepcopy

from conftest import make_packet

from perception.reader_vlm import VisionClient
from rrt_echo.runtime import Deadline


def response():
    return {
        "instances": [
            {
                "instance_id": "p",
                "kind": "person",
                "description": "visible person",
                "regions": [{"media_id": "f0", "box": [100, 100, 400, 900]}],
            }
        ],
        "facts": [
            {
                "fact_id": "e",
                "kind": "event",
                "predicate": "reach",
                "roles": {"agent": "p"},
                "value": None,
                "evidence_by_slot": {"predicate": ["f0"], "agent": ["f0"]},
                "joint_evidence": ["f0"],
                "observed_media_ids": ["f0"],
                "unresolved_slots": [],
            }
        ],
    }


def test_joint_single_call_keeps_local_binding_and_sources(monkeypatch):
    client = VisionClient(model="test", joint=True, staged=True)
    seen = []

    def complete(prompt, media, **kwargs):
        seen.append((prompt, kwargs["output_schema"]))
        return deepcopy(response())

    monkeypatch.setattr(client, "complete", complete)
    p = client.observe("a", "v", "s", make_packet("a").media, deadline=Deadline(20))
    assert len(seen) == 1
    assert set(seen[0][1]["properties"]) == {"instances", "facts"}
    assert dict(p.facts[0].roles)["agent"] == p.instances[0].instance_id == "a:p"
    assert p.instances[0].regions[0].box == (0.1, 0.1, 0.4, 0.9)
    assert p.facts[0].joint_evidence == ("a:frame",)


def test_joint_bad_role_gets_one_repair_with_error_context(monkeypatch):
    client = VisionClient(model="test", joint=True)
    prompts = []

    def complete(prompt, *args, **kwargs):
        prompts.append(prompt)
        r = response()
        if len(prompts) == 1:
            r["facts"][0]["roles"]["agent"] = "invented"
        return r

    monkeypatch.setattr(client, "complete", complete)
    p = client.observe("a", "v", "s", make_packet("a").media, deadline=Deadline(20))
    assert len(prompts) == 2
    assert "unknown local participant" in prompts[1]
    assert dict(p.facts[0].roles)["agent"] == "a:p"


def test_joint_schema_excludes_literal_attributes_from_roles():
    from perception.reader_wire import joint_schema

    schema = joint_schema(["f0"])
    ids = schema["properties"]["instances"]["items"]["properties"]["instance_id"]["enum"]
    roles = schema["properties"]["facts"]["items"]["properties"]["roles"]["additionalProperties"][
        "enum"
    ]
    assert ids == roles == [f"i{k}" for k in range(12)]
    assert "black" not in roles
