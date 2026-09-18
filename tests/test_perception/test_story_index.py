import copy
import json

import pytest
from jsonschema import ValidationError

from perception.global_index import GlobalIndexClient, select_references
from perception.graph import compile_graph

from .fixtures import clips_at, response


def proposal():
    return {
        "candidates": [
            {
                "candidate_id": "C0",
                "kind": "person",
                "label": "person in a coat",
                "description": "dark coat, reappearing later",
                "appearances": [{"start": 0, "end": 2}, {"start": 4, "end": 6}],
                "uncertainty": "",
            }
        ],
        "episodes": [
            {"start": 0, "end": 6, "summary": "A person walks outside.", "candidate_ids": ["C0"]}
        ],
        "semantic_labels": [
            {
                "candidate_id": "C0",
                "label": "Alex",
                "relation": "name",
                "related_candidate_ids": [],
                "basis_spans": [{"start": 4, "end": 6}],
                "basis": "Possible name label; owner needs verification.",
                "epistemic_status": "inferred",
            }
        ],
        "gaps": [],
    }


def test_story_request_has_no_spatial_or_frame_output_contract(tmp_path):
    body, audit = GlobalIndexClient(model="test", base_url="unused").build_request(
        clips_at(tmp_path)
    )
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][1]["content"][0]["type"] == "video_url"
    contract = json.dumps(audit["schema"])
    assert all(x not in contract for x in ["anchors", "box", "media_id", "evidence_ids"])
    assert "SOURCE_TIMELINE" not in json.dumps(body["messages"][1]["content"][1:])
    assert audit["task"] == "story"


def test_story_intervals_are_search_hints_never_fake_identity_endpoints(tmp_path):
    clips = clips_at(tmp_path)
    index = GlobalIndexClient(model="test", base_url="unused").build(
        clips, out=tmp_path / "index", sender=lambda body: response(proposal())
    )
    assert index["index_type"] == "story_hypotheses"
    c = index["candidates"][0]
    assert "anchors" not in c
    assert c["grounding_status"] == "pending_local_observation"
    assert not c["appearances"][0]["continuous_presence_asserted"]
    assert len(index["grounding_requests"]) == 1
    label = index["semantic_labels"][0]
    assert not label["eligible_for_qa_support"]
    assert not label["eligible_for_identity_merge"]
    refs, context = select_references(
        index, clips[0], {m.media_id: m for c in clips for m in c.media}, tmp_path / "refs"
    )
    assert refs == ()
    assert context["reference_gap"]
    assert "Alex" not in json.dumps(context["candidates"])
    graph = compile_graph([], index)
    assert graph["entities"] == graph["facts"] == []


@pytest.mark.parametrize("failure", ["reversed", "outside", "unknown", "box", "too_many"])
def test_story_rejects_invalid_ranges_references_and_old_spatial_contract(tmp_path, failure):
    raw = copy.deepcopy(proposal())
    if failure == "reversed":
        raw["candidates"][0]["appearances"][0] = {"start": 2, "end": 1}
    if failure == "outside":
        raw["episodes"][0]["end"] = 9
    if failure == "unknown":
        raw["semantic_labels"][0]["related_candidate_ids"] = ["not_declared"]
    if failure == "box":
        raw["candidates"][0]["anchors"] = [{"box": [0, 0, 1, 1]}]
    if failure == "too_many":
        raw["candidates"][0]["appearances"] = [{"start": i / 10, "end": 2} for i in range(5)]
    with pytest.raises((ValueError, ValidationError)):
        GlobalIndexClient(model="test", base_url="unused").build(
            clips_at(tmp_path), out=tmp_path / "index", sender=lambda body: response(raw)
        )


def test_generation_parameters_are_explicit_and_audited(tmp_path):
    settings = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 42}
    client = GlobalIndexClient(model="test", base_url="unused", generation=settings)
    body, audit = client.build_request(clips_at(tmp_path))
    assert all(body[k] == v for k, v in settings.items())
    assert audit["generation"] == settings
    with pytest.raises(ValueError, match="unsupported_generation_option"):
        GlobalIndexClient(model="test", base_url="unused", generation={"messages": []})


def compact_proposal():
    return {
        "cast": [{"id": "A", "kind": "person", "appearance": "coat", "at": 5}],
        "events": [
            {"start": 3, "end": 6, "action": "walk", "roles": [{"role": "agent", "entity_id": "A"}]}
        ],
        "labels": [
            {
                "subject": "A",
                "label": "Alex",
                "relation": "name",
                "related": None,
                "at": 5,
                "basis": "visible name cue",
                "status": "unresolved",
            }
        ],
        "gaps": [],
    }


def test_compact_roles_remain_hypotheses_and_search_spans_use_full_video(tmp_path):
    client = GlobalIndexClient(model="test", base_url="unused", task="compact_story")
    clips = clips_at(tmp_path)
    _, audit = client.build_request(clips)
    assert audit["prompt_version"] == "global-story-v5"
    assert "appearances" not in json.dumps(audit["schema"])
    index = client.build(
        clips, out=tmp_path / "index", sender=lambda body: response(compact_proposal())
    )
    assert index["episodes"][0]["role_hypotheses"] == {"agent": "v:candidate:A"}
    assert index["candidates"][0]["appearances"][0]["end"] == 6
    assert not index["candidates"][0]["appearances"][0]["continuous_presence_asserted"]
    assert not index["semantic_labels"][0]["eligible_for_qa_support"]
    assert compile_graph([], index)["entities"] == []


@pytest.mark.parametrize("failure", ["role", "label", "duplicate", "reversed", "outside"])
def test_compact_rejects_invalid_bindings(tmp_path, failure):
    raw = compact_proposal()
    if failure == "role":
        raw["events"][0]["roles"][0]["entity_id"] = "absent"
    elif failure == "label":
        raw["labels"][0]["subject"] = "absent"
    elif failure == "duplicate":
        raw["cast"].append(dict(raw["cast"][0], appearance="different description"))
    elif failure == "reversed":
        raw["events"][0]["start"] = 6
    else:
        raw["cast"][0]["at"] = 7
    with pytest.raises((ValueError, ValidationError)):
        GlobalIndexClient(model="test", base_url="unused", task="compact_story").build(
            clips_at(tmp_path), out=tmp_path / "index", sender=lambda body: response(raw)
        )
