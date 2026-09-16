from copy import deepcopy

from conftest import make_packet
from test_core import proposal

from rrt_echo.evaluation import audit_assertions, summarize
from rrt_echo.memory import Memory
from rrt_echo.retrieval import Query, retrieve
from rrt_echo.rules import change_between, state_at


def test_reader_gets_identity_proof_and_all_media():
    memory = Memory()
    for name in "abcdefghi":
        memory.append(make_packet(name))
    for left, right in zip("abcdefgh", "bcdefghi"):
        memory.compare(proposal(left + right, left, right))
    payload = retrieve(memory.snapshot(), Query("carry"), top_k=1)
    assert len(payload["identity_decisions"]) == 1
    assert len(payload["evidence"]) == 2
    from rrt_echo.scoped_identity import identity_paths

    assert len(identity_paths(memory.snapshot(), {"i:p"})["i:p"]) == 8
    assert payload["delivery_complete"]


def test_payload_budget_is_not_silent_truncation(packet):
    memory = Memory()
    memory.append(packet)
    payload = retrieve(memory.snapshot(), Query("carry"), max_bytes=512)
    assert not payload["delivery_complete"]
    assert not payload["facts"]


def test_required_dimensions_are_separate(packet):
    memory = Memory()
    memory.append(packet)
    payload = retrieve(memory.snapshot(), Query("who", ("local_joint", "global_identity")))
    assert payload["checks"]["o1:event"]["missing_dimensions"] == ["global_identity"]


def test_output_answer_audit_rejects_unreceived_proof(packet):
    memory = Memory()
    memory.append(packet)
    payload = retrieve(memory.snapshot(), Query("carry"))
    audit = audit_assertions(
        [
            {
                "assertion": "No carry",
                "support_verdict": "supported",
                "evidence_chain": ["not-received"],
            }
        ],
        payload,
    )
    assert audit["support_verdict"] == "insufficient"
    assert audit["strict_support"] is None


def test_all_assertions_required_and_contradiction_dominates(packet):
    memory = Memory()
    memory.append(packet)
    payload = retrieve(memory.snapshot(), Query("carry"))
    rows = [
        {"assertion": "a", "support_verdict": "supported", "evidence_chain": ["o1:event"]},
        {"assertion": "b", "support_verdict": "contradicted", "evidence_chain": ["o1:event"]},
    ]
    assert audit_assertions(rows, payload)["support_verdict"] == "contradicted"


def test_failures_and_unsupported_correct_answers_stay_in_denominator():
    rows = [
        {"correct": True, "type": "R4", "support_verdict": "insufficient"},
        {"correct": False, "type": "R4", "error": "timeout"},
    ]
    result = summarize(rows)
    assert result["accuracy"] == 0.5 and result["questions"] == 2
    assert result["correct_with_support"] is None
    for r in rows:
        r["media_audit"] = {"completed": True, "answer_support": "insufficient"}
    assert summarize(rows)["correct_with_support"] == 0


def test_state_does_not_persist_between_observations():
    memory = Memory()
    memory.append(make_packet("a", 100, "holds", "bag", "state"))
    memory.append(make_packet("b", 120, "holds", "bag", "state"))
    memory.compare(proposal("ab", "a", "b"))
    graph = memory.snapshot()
    assert state_at(graph, "a:p", "holds", 100)
    assert not state_at(graph, "a:p", "holds", 110)
    assert not graph["relations"]


def test_state_change_is_not_a_cause_or_exact_transition():
    memory = Memory()
    memory.append(make_packet("a", 100, "health", "uninjured", "state"))
    memory.append(make_packet("b", 120, "health", "injured", "state"))
    memory.compare(proposal("ab", "a", "b"))
    before, after = memory.snapshot()["facts"]
    result = change_between(before, after)
    assert result["bounds"] == [100, 120]
    assert result["cause"] is None and result["exact_time"] is None
    assert result["identity_dependencies"]
    # No mutation of the observed state records.
    old = deepcopy(before)
    change_between(before, after)
    assert before == old


def test_participant_description_drives_retrieval():
    memory = Memory()
    memory.append(make_packet("a", predicate="hold"))
    memory.append(make_packet("z", predicate="hold"))
    graph = memory.snapshot()
    graph["instances"][0]["description"] = "woman carrying a bag"
    graph["instances"][1]["description"] = "infant holding a map"
    payload = retrieve(graph, Query("What is the baby holding?"), top_k=1)
    assert payload["facts"][0]["fact_id"] == "z:event"


def test_owner_projection_is_in_entity_scope():
    memory = Memory()
    memory.append(make_packet("a"))
    graph = memory.snapshot()
    graph["facts"][0]["owner_projections"] = {"agent": {"owner_entity": "owner_entity"}}
    payload = retrieve(graph, Query("", entity_ids=("owner_entity",)))
    assert len(payload["facts"]) == 1


def test_reader_snapshot_load_rejects_tampering(tmp_path):
    import json

    import pytest

    from rrt_echo.storage import read_graph

    memory = Memory()
    memory.append(make_packet())
    graph = memory.snapshot()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    assert read_graph(path)["snapshot_id"] == graph["snapshot_id"]
    graph["facts"][0]["predicate"] = "invented"
    path.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="snapshot_digest_mismatch"):
        read_graph(path)


def test_name_search_delivers_attribute_and_identity_proof():
    memory = Memory()
    memory.append(make_packet("a", 1, "name", "Alice", "attribute"))
    memory.append(make_packet("b", 2, "carry"))
    memory.compare(proposal("ab", "a", "b"))
    payload = retrieve(memory.snapshot(), Query("What does Alice carry?"), top_k=1)
    assert {f["fact_id"] for f in payload["facts"]} == {"a:event", "b:event"}
    assert payload["identity_decisions"]
    assert payload["closure_complete"]
