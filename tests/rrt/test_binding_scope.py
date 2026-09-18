from types import SimpleNamespace

from perception.local_stage import coverage_report
from rrt_echo.rrt.binding_scope import text_companions, video_time_ranges
from rrt_echo.rrt.diagnostics import repair_plan
from rrt_echo.rrt.link_review import endpoint_verdict


def test_interval_includes_middle_and_excludes_cctv_clock():
    assert video_time_ranges("video (09:31 - 09:45), petrol at 21:25, car at 22:06") == (
        (566, 590),
    )
    assert video_time_ranges("early (01:14), later (06:06)") == ((69, 79), (361, 371))
    assert video_time_ranges("00:15 to 01:14") == ((10, 79),)


def text(fid, owner, obs="scene", times=None):
    return dict(
        fact_id=fid,
        kind="text",
        roles={"owner": owner},
        observed_times=times or [10],
        dependencies={"observation": obs},
    )


def test_text_closure_preserves_local_owner_and_occurrence():
    a = text("a", "arm")
    b = text("b", "arm")
    wrong = text("wrong", "map")
    later = text("later", "arm", obs="later")
    distant = text("distant", "arm", times=[50])
    assert [f["fact_id"] for f in text_companions(a, [a, b, wrong, later, distant])] == ["a", "b"]


def test_part_regions_can_share_explicit_local_owner_but_not_global_identity_alone():
    a = text("a", "r1")
    b = text("b", "r2")
    c = text("c", "r3")
    for f, local in [(a, "p1"), (b, "p1"), (c, "p2")]:
        f["owner_projections"] = {
            "owner": {"owner_entity": "global_same", "local_owner_endpoints": [local]}
        }
    assert [f["fact_id"] for f in text_companions(a, [a, b, c])] == ["a", "b"]


def test_wrong_requested_endpoint_cannot_be_merged():
    answer = dict(
        source_category="adult",
        target_category="adult",
        source_localized=True,
        target_localized=True,
        source_matches_requested=False,
        target_matches_requested=True,
        verdict="supported",
    )
    assert endpoint_verdict("same_identity", "person", "person", answer) == "unresolved"


def test_verified_observation_counts_as_covered_and_is_not_retried():
    clip = SimpleNamespace(clip_id="clip", target_start=0, target_end=8, target_ids=["m"])
    obs = dict(
        observation_id="clip:verified-v7",
        instances=[],
        facts=[dict(fact_id="f", kind="text", roles={}, joint_evidence=["m"])],
    )
    result = {"observation": obs}
    assert coverage_report([clip], [result])["completed"] == 1
    assert repair_plan([clip], [result]) == []
    assert repair_plan([clip], [])[0]["reason"] == "not_processed"


def test_open_predicate_preserves_action_instead_of_generic_interact():
    from rrt_echo.rrt.verified import enforce_role_contracts

    f = dict(
        kind="event",
        predicate="polish",
        roles=[{"role": "agent"}, {"role": "patient"}],
        unresolved_slots=[],
    )
    r = enforce_role_contracts({"facts": [f]})
    assert r["facts"][0]["predicate"] == "polish"
    assert r["facts"][0]["open_event_contract"] is True


def test_shooting_contract_keeps_missing_participants_explicit():
    from rrt_echo.rrt.verified import enforce_role_contracts

    f = dict(kind="event", predicate="shoot", roles=[{"role": "agent"}], unresolved_slots=[])
    r = enforce_role_contracts({"facts": [f]})
    assert r["facts"][0]["unresolved_slots"] == ["instrument", "patient"]
    assert len(r["facts"][0]["roles"]) == 1


def test_context_reaches_proposer_but_remains_outside_target_evidence(tmp_path):
    import json

    from PIL import Image

    from perception.types import Clip, Media, digest
    from rrt_echo.rrt.verified import VerifiedPerceptionClient

    image = tmp_path / "frame.jpg"
    Image.new("RGB", (40, 40), "gray").save(image)
    media = tuple(Media(f"m{i}", str(image), i, (1, 1), digest(image), i) for i in range(3))
    clip = Clip("clip", "v", 0, 1, 2, media, 1, 3, 3, 0, True, {})
    requests = []

    def sender(body):
        requests.append(body)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"facts": [], "rejected": [], "gaps": []})},
                }
            ]
        }

    result = VerifiedPerceptionClient(model="fake", base_url="unused", sender=sender).observe(
        clip, out=tmp_path / "run"
    )
    texts = " ".join(c.get("text", "") for c in requests[0]["messages"][0]["content"])
    assert (
        "TARGET frame 1" in texts
        and "CONTEXT ONLY frame 0" in texts
        and "CONTEXT ONLY frame 2" in texts
    )
    assert result["context_used_for_proposals"] == ["m0", "m2"]
    assert result["observation"]["facts"] == []


def test_state_sequence_view_picks_payload_entities():
    from rrt_echo.rrt.binding_scope import state_sequence_view

    graph = {
        "state_sequences": [
            {"entity_id": "e0", "status": "observed_sequence_not_inferred_transition", "sequence": []},
        ]
    }
    payload = {
        "facts": [
            {"resolved_roles": {"owner": {"entity_id": "e0"}}, "owner_projections": {}},
        ]
    }
    view = state_sequence_view(payload, graph)
    assert len(view) == 1
    assert view[0]["entity_id"] == "e0"
