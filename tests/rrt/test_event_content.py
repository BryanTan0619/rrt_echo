import copy

import pytest

from rrt_echo.rrt.event_content import attach_event_contents, event_content_closure
from rrt_echo.rrt.link_review import endpoint_verdict


def fixture():
    instances = [
        {"instance_id": i, "kind": k, "regions": [{"media_id": "m", "box": [0, 0, 1, 1]}]}
        for i, k in [("p", "person"), ("pen", "object"), ("map", "object"), ("arm", "region")]
    ]
    e = {
        "fact_id": "e",
        "kind": "event",
        "roles": {"writer": "p", "instrument": "pen", "surface": "map"},
        "joint_evidence": ["m"],
    }
    t = {
        "fact_id": "t",
        "kind": "text",
        "roles": {"owner": "map"},
        "value": "SAFE",
        "joint_evidence": ["m"],
    }
    g = {"instances": instances, "facts": [e, t], "evidence": {"m": {"seconds": 1}}}
    r = {
        "review_id": "r",
        "event_id": "e",
        "text_id": "t",
        "event_roles": {k: k for k in e["roles"]},
        "verdict": "supported",
        "origin": "test",
        "basis": "review",
        "content_relation": "co_observed_inscription",
        "evidence_ids": ["m"],
    }
    return g, r


def test_explicit_content_closure_and_revocation_preserve_raw_facts():
    g, r = fixture()
    before = copy.deepcopy(g)
    linked = attach_event_contents(g, [r])
    assert g == before
    assert len(event_content_closure([g["facts"][1]], g["facts"], linked)) == 2
    assert attach_event_contents(g, [{**r, "revoked": True}])["event_hyperedges"] == []
    assert linked["event_hyperedges"][0]["content"]["relation"] == "co_observed_inscription"


def test_text_on_other_surface_does_not_bind_by_cooccurrence():
    g, r = fixture()
    g["facts"][1]["roles"]["owner"] = "arm"
    with pytest.raises(ValueError, match="unproved_surface"):
        attach_event_contents(g, [r])


def test_inscription_does_not_prove_writing_created_it():
    g, r = fixture()
    r["content_relation"] = "written_content"
    with pytest.raises(ValueError, match="creation_requires"):
        attach_event_contents(g, [r])


def test_body_part_contact_with_tool_is_not_ownership():
    answer = {
        "source_category": "body_part",
        "target_category": "object",
        "source_localized": True,
        "target_localized": True,
        "verdict": "supported",
    }
    assert endpoint_verdict("part_of", "region", "object", answer) == "unresolved"


def test_automatic_bundles_require_same_local_owner_and_shared_evidence():
    from rrt_echo.rrt.event_content import attach_coobserved_attributes

    g, _ = fixture()
    g["facts"][0]["predicate"] = "write"
    g["facts"][1]["predicate"] = "text"
    linked = attach_coobserved_attributes(g)
    assert len(linked["event_hyperedges"]) == 1
    assert linked == attach_coobserved_attributes(linked)
    assert linked["facts"] == g["facts"]
    assert linked["event_hyperedges"][0]["role_attributes"][0]["event_role"] == "surface"
    g["facts"][1]["roles"]["owner"] = "arm"
    assert not attach_coobserved_attributes(g)["event_hyperedges"]
    g["facts"][1]["roles"]["owner"] = "map"
    g["facts"][1]["joint_evidence"] = ["later_frame"]
    assert not attach_coobserved_attributes(g)["event_hyperedges"]


def test_shared_attribute_does_not_expand_into_another_occurrence():
    facts = [{"fact_id": x} for x in ["write", "shirt", "hold", "bag"]]
    graph = {
        "event_hyperedges": [
            {"source_fact_ids": ["write", "shirt"]},
            {"source_fact_ids": ["hold", "shirt", "bag"]},
        ]
    }
    assert {f["fact_id"] for f in event_content_closure([facts[0]], facts, graph)} == {
        "write",
        "shirt",
    }
    # An attribute can independently retrieve either event; adding it as a
    # companion to write must not recursively promote it into a retrieval seed.
    assert {f["fact_id"] for f in event_content_closure([facts[1]], facts, graph)} == {
        "write",
        "shirt",
        "hold",
        "bag",
    }
