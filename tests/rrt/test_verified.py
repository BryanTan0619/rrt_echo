import json

import pytest
from PIL import Image

from rrt_echo.echo_perception.types import Clip, Media, digest
from rrt_echo.rrt.verified import VerifiedPerceptionClient, usable_box, validate_review


def fact(frames=(0, 1)):
    return {
        "kind": "event",
        "predicate": "hold",
        "value": None,
        "frames": list(frames),
        "roles": [
            {"role": "holder", "kind": "person", "description": "person", "frame": 0},
            {"role": "held", "kind": "object", "description": "cup", "frame": 0},
        ],
        "unresolved_slots": [],
    }


def test_review_cannot_cite_context_or_unsupplied_frames():
    with pytest.raises(ValueError, match="evidence_not_in_supplied_target"):
        validate_review({"facts": [fact((0, 2))]}, {0: None, 1: None})


def test_role_frame_must_belong_to_event():
    f = fact()
    f["roles"][1]["frame"] = 2
    with pytest.raises(ValueError, match="role_frame_outside_fact"):
        validate_review({"facts": [f]}, {0: None, 1: None, 2: None})


def test_invalid_or_invisible_box_cannot_pass():
    assert not usable_box({"visible": False, "box": [0, 0, 100, 100]})
    assert not usable_box({"visible": True, "box": [100, 0, 0, 100]})
    assert not usable_box({"visible": True, "box": [-1, 0, 100, 100]})


@pytest.mark.parametrize("joint_verdict", ["supported", "contradicted"])
@pytest.mark.parametrize("supported_frames", [[0, 1], [1], [2], []])
def test_failed_role_is_unresolved_and_bad_anchor_does_not_enter_result(
    tmp_path, joint_verdict, supported_frames
):
    media = []
    for n in range(3):
        path = tmp_path / f"{n}.jpg"
        Image.new("RGB", (80, 60)).save(path)
        media.append(Media(f"v:f{n}", str(path), n, (1, 1), digest(path), n))
    clip = Clip("clip", "v", 0, 0, 2, tuple(media), 1, 3, 3, 0, True, {})
    requests = []

    def sender(body):
        requests.append(body)
        prompt = body["messages"][0]["content"][-1]["text"]
        if body["response_format"]["json_schema"]["name"] == "temporal_review":
            assert len([c for c in body["messages"][0]["content"] if c["type"] == "image_url"]) == 3
            result = {"facts": [fact()], "rejected": [], "gaps": []}
        elif "Locate each requested" in prompt:
            result = {
                "regions": [
                    {"id": "q0_holder", "visible": True, "box": [10, 10, 500, 900]},
                    {"id": "q0_held", "visible": True, "box": [500, 500, 900, 900]},
                ]
            }
        elif "Audit this proposed fact" in prompt:
            result = {
                "verdict": joint_verdict,
                "reason": "mock joint evidence decision",
                "missing_roles": [],
                "supported_frames": supported_frames,
            }
        else:
            result = {
                "regions": [
                    {
                        "id": "q0_holder",
                        "verdict": "supported",
                        "reason": "person in box",
                        "observed_kind": "person",
                        "observed_age_group": "adult",
                        "matches_requested_subject": True,
                    },
                    {
                        "id": "q0_held",
                        "verdict": "contradicted",
                        "reason": "box on background",
                        "observed_kind": "unclear",
                    },
                ]
            }
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}

    proposals = {"observation": {"observation_id": "old", "instances": [], "facts": []}}
    result = VerifiedPerceptionClient(
        model="test", base_url="http://unused", sender=sender
    ).observe(clip, out=tmp_path / "run", proposals=proposals)
    assert len(requests) == 6
    if joint_verdict == "contradicted" or supported_frames != [0, 1]:
        assert result["observation"]["facts"] == []
        assert result["observation"]["instances"] == []
        return
    assert len(result["observation"]["instances"]) == 1
    f = result["observation"]["facts"][0]
    assert set(f["roles"]) == {"holder"} and f["unresolved_slots"] == ["held"]
    assert result["semantic_support_audited"] is False
    assert result["source_observation_id"] == "old"


def test_role_contract_preserves_missing_patient_without_inventing_instance():
    from rrt_echo.rrt.verified import enforce_role_contracts

    f = fact()
    f["roles"] = f["roles"][:1]
    enforce_role_contracts({"facts": [f]})
    assert f["unresolved_slots"] == ["held"]
    assert len(f["roles"]) == 1


def test_relation_cannot_hide_in_property_predicate():
    from rrt_echo.rrt.verified import enforce_role_contracts

    f = fact()
    f.update(kind="state", predicate="infant_holding_paper", value="yes")
    result = enforce_role_contracts({"facts": [f, fact()]})
    assert len(result["facts"]) == 1
    assert result["rejected"][0]["reason"] == "relation_or_unknown_predicate_in_property"


def test_unexpected_role_quarantines_only_that_candidate():
    from rrt_echo.rrt.verified import enforce_role_contracts

    f = fact()
    f["predicate"] = "turn"
    result = enforce_role_contracts({"facts": [f, fact()]})
    assert len(result["facts"]) == 1
    assert result["facts"][0]["predicate"] == "hold"
    assert result["rejected"][0]["reason"] == "unexpected_event_role"


def test_baby_role_cannot_accept_an_adult_box():
    from rrt_echo.rrt.verified import subject_audit_matches

    subject = {"kind": "person", "description": "baby in carrier"}
    audit = {
        "verdict": "supported",
        "matches_requested_subject": True,
        "observed_age_group": "adult",
    }
    assert not subject_audit_matches(subject, audit)
    audit["observed_age_group"] = "child"
    assert subject_audit_matches(subject, audit)
