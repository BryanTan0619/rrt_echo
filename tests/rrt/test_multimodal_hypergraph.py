import copy

import pytest
from conftest import make_packet

from rrt_echo.evaluation import audit_assertions
from rrt_echo.memory import Memory
from rrt_echo.rrt.audio import attach_audio
from rrt_echo.rrt.hypergraph import (
    SCHEMA,
    TemporalEvidenceHypergraph,
    entity_map,
    proof_closure,
    seal,
)
from rrt_echo.rrt.multimodal import plan_bindings, story_context, submission_from_result
from rrt_echo.rrt.reader import compact_payload, payload_for
from rrt_echo.schema import digest
from rrt_echo.storage import atomic_json, read_graph


def graph():
    memory = Memory(source_video_id="v")
    memory.append(make_packet("a", time=10))
    memory.append(make_packet("b", time=10))
    audio = {
        "source_video_id": "v",
        "source": {"sha256": "a" * 64, "status": "transcribed"},
        "utterances": [
            {
                "utterance_id": "u",
                "speaker_id": None,
                "text": "I am the medic",
                "start": 9.5,
                "end": 10.5,
                "speaker_status": "unresolved",
                "evidence": {"start": 9.5, "end": 10.5},
            }
        ],
        "speakers": [],
    }
    audio["artifact_id"] = digest(audio)
    return attach_audio(memory.snapshot(), audio)


def packet(person="a:p", sid="s"):
    reviews = [
        {
            "stage": stage,
            "verdict": "supported",
            "model": "test_av_model",
            "used_modalities": ["audio", "video"],
            "story_used_as_evidence": False,
            "observed_speaking": True,
            "reason": "synchronized speaking verified in test fixture",
            "evidence_id": "av",
            "local_instance": person,
            "utterance_id": "u",
        }
        for stage in ["propose", "verify"]
    ]
    return {
        "schema": SCHEMA,
        "submission_id": sid,
        "video_id": "v",
        "source_sha256": "a" * 64,
        "evidence_bundles": {
            "av": {
                "evidence_id": "av",
                "source_sha256": "a" * 64,
                "start": 9,
                "end": 11,
                "modalities": ["audio", "video"],
                "audio_samples": 32000,
                "video_frames": 16,
                "clip_sha256": "b" * 64,
                "uri": "synthetic://av",
                "candidate_instances": ["a:p", "b:p"],
            }
        },
        "bindings": [
            {
                "proposal_id": sid + ":p",
                "relation": "speaker_of",
                "utterance_id": "u",
                "local_instance": person,
                "evidence_ids": ["av"],
                "visual_media_ids": [person.split(":")[0] + ":frame"],
                "reviews": reviews,
            }
        ],
        "claims": [
            {
                "claim_id": "c",
                "kind": "reported_claim",
                "source_utterance": "u",
                "text": "I am the medic",
                "world_fact": False,
            }
        ],
    }


def test_commit_revoke_preserves_original_and_invalidates_projection(tmp_path):
    g = graph()
    original = copy.deepcopy(g)
    memory = TemporalEvidenceHypergraph(g)
    memory.submit(packet())
    accepted = memory.snapshot()
    edge = accepted["multimodal"]["hyperedges"][0]
    assert edge["resolved_roles"]["speaker"] == entity_map(g)["a:p"]
    assert accepted["multimodal"]["bindings"][0]["status"] == "accepted"
    assert accepted["multimodal"]["hyperedges"][1]["world_fact"] is False
    memory.revoke("s:p", "wrong person in independent inspection")
    revoked = memory.snapshot()
    assert revoked["multimodal"]["hyperedges"][0]["resolved_roles"] == {}
    assert revoked["audio_memory"] == g["audio_memory"]
    assert g == original
    path = tmp_path / "graph.json"
    atomic_json(path, revoked)
    assert read_graph(path)["snapshot_id"] == revoked["snapshot_id"]


def test_conflicting_people_cannot_share_one_utterance_speaker():
    memory = TemporalEvidenceHypergraph(graph())
    memory.submit(packet())
    memory.submit(packet("b:p", "s2"))
    result = memory.snapshot()
    assert {p["status"] for p in result["multimodal"]["bindings"]} == {"conflict"}
    assert result["multimodal"]["hyperedges"][0]["unresolved_slots"] == ["speaker"]
    memory.revoke("s2:p", "contradicted")
    assert memory.snapshot()["multimodal"]["bindings"][0]["status"] == "accepted"


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda p: p.update(source_sha256="c" * 64), "source_mismatch"),
        (lambda p: p["bindings"][0].update(local_instance="missing"), "endpoint"),
        (lambda p: p["evidence_bundles"]["av"].update(end=10), "outside_evidence"),
        (lambda p: p["evidence_bundles"]["av"].update(audio_samples=0), "audio_video_required"),
        (lambda p: p["bindings"][0].update(visual_media_ids=["b:frame"]), "endpoint_missing"),
        (lambda p: p["claims"][0].update(world_fact=True), "not_world_fact"),
    ],
)
def test_submission_is_atomic(change, error):
    memory = TemporalEvidenceHypergraph(graph())
    before = memory.snapshot()
    p = packet()
    change(p)
    with pytest.raises(ValueError, match=error):
        memory.submit(p)
    assert memory.snapshot() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("used_modalities", ["video"]),
        ("story_used_as_evidence", True),
        ("observed_speaking", False),
    ],
)
def test_story_or_video_only_cannot_admit_voice_binding(field, value):
    memory = TemporalEvidenceHypergraph(graph())
    p = packet()
    p["bindings"][0]["reviews"][1][field] = value
    memory.submit(p)
    assert memory.snapshot()["multimodal"]["bindings"][0]["status"] == "unresolved"


def test_retrieval_requires_complete_speaker_proof_and_scope():
    memory = TemporalEvidenceHypergraph(graph())
    memory.submit(packet())
    g = memory.snapshot()
    payload = payload_for(
        g, "Who says medic?", require_speaker=True, speaker_entity_ids=[entity_map(g)["a:p"]]
    )
    assert payload["speaker_bindings"][0]["proposal_id"] == "s:p"
    assert payload["multimodal_hyperedges"][0]["source_utterance"] == "u"
    assert "av" in payload["audio_video_evidence"]
    assert "a:frame" in payload["evidence"]
    compact, aliases = compact_payload(payload)
    assert aliases[compact["multimodal_hyperedges"][0]["edge_id"]] == "u:says"
    wrong = proof_closure(g, ["u"], entity_ids=[entity_map(g)["b:p"]], require_speaker=True)
    assert wrong["hyperedges"] == [] and wrong["gaps"]
    assertion = [
        {"assertion": "a says medic", "support_verdict": "supported", "evidence_chain": ["u:says"]}
    ]
    assert audit_assertions(assertion, payload)["support_verdict"] == "supported"
    broken = copy.deepcopy(payload)
    broken["speaker_bindings"] = []
    assert audit_assertions(assertion, broken)["support_verdict"] == "insufficient"
    short = payload_for(g, "Who says medic?", max_bytes=512)
    assert short.get("gap") and not short.get("multimodal_hyperedges")


def test_forged_projection_rejected_even_with_new_snapshot_hash(tmp_path):
    memory = TemporalEvidenceHypergraph(graph())
    g = memory.snapshot()
    g["multimodal"]["hyperedges"][0]["resolved_roles"] = {"speaker": "fake"}
    path = tmp_path / "graph.json"
    atomic_json(path, seal(g))
    with pytest.raises(ValueError, match="projection_mismatch"):
        read_graph(path)


def test_planner_no_qa_and_blind_second_pass_disagreement():
    g = graph()
    plan = plan_bindings(g, max_tasks=1)
    assert plan["query_blind"] and len(plan["tasks"]) == 1
    assert not plan_bindings(g, max_tasks=0)["tasks"]
    assert story_context(g, 9, 11)["cast_is_exhaustive"] is False
    task = plan["tasks"][0]
    first = {
        "local_instance": "a:p",
        "verdict": "supported",
        "model": "m",
        "request_id": "first",
        "observed_speaking": True,
        "story_used_as_evidence": False,
        "reason": "test",
    }
    second = {**first, "local_instance": "b:p", "request_id": "second"}
    evidence = packet()["evidence_bundles"]["av"]
    evidence.update(start=task["start"], end=task["end"])
    p = submission_from_result(g, task, evidence, first, second)
    m = TemporalEvidenceHypergraph(g)
    m.submit(p)
    assert m.snapshot()["multimodal"]["bindings"][0]["status"] == "unresolved"


def test_utterance_cannot_prove_world_role_or_named_speaker():
    memory = TemporalEvidenceHypergraph(graph())
    g = memory.snapshot()
    payload = payload_for(g, "medic")
    a = {
        "assertion": "person is the medic",
        "support_verdict": "supported",
        "evidence_chain": ["u"],
        "claim_type": "world_state",
    }
    assert audit_assertions([a], payload)["support_verdict"] == "insufficient"
    a["claim_type"] = "speaker_identity"
    assert audit_assertions([a], payload)["support_verdict"] == "insufficient"
    a["claim_type"] = "utterance_content"
    assert audit_assertions([a], payload)["support_verdict"] == "supported"


def test_stale_journal_cannot_fabricate_a_submission(tmp_path):
    memory = TemporalEvidenceHypergraph(graph())
    memory.submit(packet())
    g = memory.snapshot()
    g["multimodal"]["journal"][0]["digest"] = "bad"
    path = tmp_path / "graph.json"
    atomic_json(path, seal(g))
    with pytest.raises(ValueError, match="journal_submission_mismatch"):
        read_graph(path)


def test_unheard_model_claim_cannot_be_written_as_speech():
    memory = TemporalEvidenceHypergraph(graph())
    p = packet()
    p["claims"][0]["text"] = "the man in black is speaking"
    with pytest.raises(ValueError, match="not_literal_utterance"):
        memory.submit(p)
    assert not memory.submissions
