import json

import pytest
from conftest import make_packet

from rrt_echo.memory import Memory
from rrt_echo.rrt.reader import evaluate, question_options


def test_boolean_and_mcq_options():
    assert question_options({"question_type": "True/False", "question": "statement"}) == {
        "True": "True",
        "False": "False",
    }
    assert question_options({"question": "Which?\n A. first\n B. second"}) == {
        "A": "first",
        "B": "second",
    }


def test_diagnostic_never_claims_acceptance(tmp_path, monkeypatch):
    memory = Memory(source_video_id="v")
    memory.append(make_packet())
    graph = memory.snapshot()
    gp = tmp_path / "g.json"
    gp.write_text(json.dumps(graph))
    ap = tmp_path / "a.json"
    ap.write_text(json.dumps({"snapshot_id": graph["snapshot_id"], "passed": False}))
    qp = tmp_path / "q.jsonl"
    qp.write_text(
        json.dumps(
            {
                "video_id": "v",
                "question_type": "True/False",
                "question": "Someone carries something.",
                "ans": "True",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="media_acceptance_required"):
        evaluate(gp, qp, ap, out=tmp_path / "strict", model="fake", base_url="unused")

    class Client:
        def __init__(self, **kw):
            self.calls = []

        def complete(self, prompt, **kw):
            self.calls.append(prompt)
            return (
                {"answer": "true"} if "Select exactly one option" in prompt else {"assertions": []}
            )

    monkeypatch.setattr("rrt_echo.rrt.reader.VisionClient", Client)
    rows = evaluate(
        gp, qp, ap, out=tmp_path / "diagnostic", model="fake", base_url="unused", diagnostic=True
    )
    assert rows[0]["correct"]
    manifest = json.loads((tmp_path / "diagnostic/manifest.json").read_text())
    assert manifest["diagnostic"] and not manifest["snapshot_media_accepted"]


def test_compact_aliases_roundtrip_evidence():
    from rrt_echo.rrt.reader import compact_payload, payload_for, restore_audit_ids

    memory = Memory()
    memory.append(make_packet())
    payload = payload_for(memory.snapshot(), "carry")
    compact, reverse = compact_payload(payload)
    short = compact["facts"][0]["fact_id"]
    assert reverse[short] == "o1:event"
    assert len(json.dumps(compact)) < len(json.dumps(payload))
    restored = restore_audit_ids({"assertions": [{"evidence_chain": [short, "unknown"]}]}, reverse)
    assert restored[0]["evidence_chain"] == ["o1:event", "unknown"]
    assert "uri" not in next(iter(compact["evidence"].values()))


def test_timestamp_and_option_scopes_exclude_other_occurrence():
    from rrt_echo.rrt.reader import payload_for

    memory = Memory()
    memory.append(make_packet("early", time=15, predicate="injury", value="blood", kind="state"))
    memory.append(make_packet("late", time=200, predicate="carry"))
    payload = payload_for(
        memory.snapshot(), "At 00:15 what is seen?\nA) blood injury\nB) carry", top_k=10
    )
    assert {f["fact_id"] for f in payload["facts"]} == {"early:event"}
    assert payload["hypotheses_are_claims"] is False
    assert set(payload["retrieval_hypotheses"]) == {"question", "option_A", "option_B"}


def test_quarantine_retains_graph_fact_but_prevents_retrieval():
    from rrt_echo.rrt.reader import payload_for
    from rrt_echo.rrt.visual_revision import quarantine_endpoints

    memory = Memory()
    memory.append(make_packet("wrong", predicate="carry"))
    original = memory.snapshot()
    graph = quarantine_endpoints(
        original,
        [
            {
                "instance_id": "wrong:p",
                "verdict": "invalid",
                "evidence_ids": ["wrong:frame"],
                "reason": "adult box requested for baby",
            }
        ],
    )
    assert len(graph["facts"]) == 1
    assert "retrieval_quarantined" not in original["facts"][0]
    assert not payload_for(graph, "carry")["facts"]
