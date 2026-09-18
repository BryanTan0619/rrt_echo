import copy

from rrt_echo.rrt.memory import RRTMemory


def result(oid, kind="person"):
    m = {
        "media_id": oid + ":f",
        "uri": "synthetic://" + oid,
        "pts": 0,
        "time_base": [1, 1],
        "sha256": "a" * 64,
        "index": 0,
    }
    return {
        "observation": {
            "observation_id": oid,
            "video_id": "v",
            "shot_id": "0",
            "media": [m],
            "instances": [
                {
                    "instance_id": oid + ":i",
                    "kind": kind,
                    "description": "subject",
                    "regions": [{"media_id": m["media_id"], "box": [0, 0, 1, 1]}],
                }
            ],
            "facts": [
                {
                    "fact_id": oid + ":event",
                    "kind": "event",
                    "predicate": "walk",
                    "roles": {"agent": oid + ":i"},
                    "value": None,
                    "evidence_by_slot": {"predicate": [m["media_id"]], "agent": [m["media_id"]]},
                    "joint_evidence": [m["media_id"]],
                    "observed_media_ids": [m["media_id"]],
                    "unresolved_slots": [],
                }
            ],
        },
        "link_proposals": [],
        "coverage_gaps": [],
    }


def link(pid, left, right, rel="same_identity", verdict="supported"):
    return {
        "proposal_id": pid,
        "source": left + ":i",
        "target": right + ":i",
        "relation": rel,
        "verdict": verdict,
        "evidence_ids": [left + ":f", right + ":f"],
        "basis": "visual endpoints",
    }


def test_revoke_changes_projection_not_observation():
    a, b = result("a"), result("b")
    before = copy.deepcopy(a)
    mem = RRTMemory()
    mem.append(a)
    mem.append(b)
    mem.propose(link("ab", "a", "b"))
    g = mem.snapshot()
    assert g == mem.snapshot()
    assert len(g["entities"]) == 1
    mem.revoke("ab", "new contrary evidence")
    g2 = mem.snapshot()
    assert len(g2["entities"]) == 2 and a == before
    assert g["facts"][0]["roles"] == g2["facts"][0]["roles"]
    assert g["snapshot_id"] != g2["snapshot_id"]


def test_region_ownership_survives_memory_projection():
    mem = RRTMemory()
    mem.append(result("arm", "region"))
    mem.append(result("person"))
    mem.propose(link("owns", "arm", "person", "part_of"))
    g = mem.snapshot()
    f = next(f for f in g["facts"] if f["fact_id"] == "arm:event")
    assert f["owner_projections"]["agent"]["local_owner_endpoints"] == ["person:i"]
    assert f["dependencies"]["ownership"] == ["owns"]
    mem.revoke("owns", "wrong owner")
    assert not mem.snapshot()["facts"][0]["owner_projections"]


def test_cluster_conflict_quarantines_all_same_links():
    mem = RRTMemory()
    for x in "abc":
        mem.append(result(x))
    mem.propose(link("ab", "a", "b"))
    mem.propose(link("bc", "b", "c"))
    mem.propose(link("ac", "a", "c", verdict="contradicted"))
    assert len(mem.snapshot()["entities"]) == 3


def test_reader_includes_owner_identity_path():
    from rrt_echo.rrt.reader import payload_for

    mem = RRTMemory()
    arm = result("arm", "region")
    arm["observation"]["facts"][0]["predicate"] = "inscription"
    mem.append(arm)
    mem.append(result("person"))
    mem.append(result("later"))
    mem.propose(link("owns", "arm", "person", "part_of"))
    mem.propose(link("same", "person", "later"))
    payload = payload_for(mem.snapshot(), "inscription", top_k=1)
    assert {p["proposal_id"] for p in payload["binding_links"]} == {"owns"}
    assert any(d["proposal"]["proposal_id"] == "same" for d in payload["identity_decisions"])
    assert {i["instance_id"] for i in payload["instances"]} == {"arm:i", "person:i", "later:i"}


def test_replaced_object_cannot_merge_with_replacement_and_constraint_is_revisable():
    a = result("old", "object")
    b = result("new", "object")
    a["observation"]["instances"].extend(b["observation"]["instances"])
    a["observation"]["media"].extend(b["observation"]["media"])
    f = a["observation"]["facts"][0]
    f.update(
        predicate="replace",
        roles={"patient": "old:i", "theme": "new:i"},
        unresolved_slots=["agent"],
        joint_evidence=["old:f", "new:f"],
        observed_media_ids=["old:f", "new:f"],
        evidence_by_slot={
            "predicate": ["old:f", "new:f"],
            "patient": ["old:f"],
            "theme": ["new:f"],
        },
    )
    mem = RRTMemory()
    mem.append(a)
    derived = mem.journal[0]["proposal"]["proposal_id"]
    mem.propose(link("looks_same", "old", "new"))
    assert len(mem.snapshot()["entities"]) == 2
    mem.revoke(derived, "replacement event was rejected by subsequent visual review")
    assert len(mem.snapshot()["entities"]) == 1
    assert mem.observations[0]["observation"]["facts"][0]["predicate"] == "replace"


def test_cached_body_part_to_tool_link_is_not_accepted():
    mem = RRTMemory()
    mem.append(result("arm", "region"))
    mem.append(result("pen", "object"))
    proposal = link("bad", "arm", "pen", "part_of")
    proposal["focused_review"] = {"source_category": "body_part", "target_category": "object"}
    mem.propose(proposal)
    graph = mem.snapshot()
    assert not graph["facts"][0]["owner_projections"]
    assert graph["binding_links"][0]["gap_reason"] == "body_part_owner_type_mismatch"


def test_build_state_sequences_orders_by_time_and_excludes_non_state():
    from rrt_echo.rrt.memory import build_state_sequences

    graph = {
        "facts": [
            {
                "fact_id": "f2", "kind": "state", "predicate": "smooth", "value": "yes",
                "roles": {"owner": "i0"},
                "resolved_roles": {"owner": {"entity_id": "e0"}},
                "observed_times": [50.0],
            },
            {
                "fact_id": "f1", "kind": "state", "predicate": "dry", "value": "yes",
                "roles": {"owner": "i0"},
                "resolved_roles": {"owner": {"entity_id": "e0"}},
                "observed_times": [10.0],
            },
            {
                "fact_id": "f3", "kind": "event", "predicate": "mix", "value": None,
                "roles": {"agent": "i0"},
                "resolved_roles": {"agent": {"entity_id": "e0"}},
                "observed_times": [30.0],
            },
        ]
    }
    seqs = build_state_sequences(graph)
    assert len(seqs) == 1
    assert seqs[0]["status"] == "observed_sequence_not_inferred_transition"
    preds = [it["predicate"] for it in seqs[0]["sequence"]]
    assert preds == ["dry", "smooth"]  # 时间排序，event 被排除
