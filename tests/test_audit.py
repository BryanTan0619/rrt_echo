from conftest import make_packet

from rrt_echo.audit import census
from rrt_echo.memory import Memory


def test_census_separates_singletons_and_global_links():
    memory = Memory()
    memory.append(make_packet("a"))
    result = census(memory.snapshot())
    assert result["local_instances_all"] == 1
    assert result["registered_entities_all"] == 1
    assert result["identity_accuracy"] is None
    assert result["media_audit_required"]


def test_observer_retries_bad_reference_once(monkeypatch):
    from rrt_echo.perception.vlm import VisionClient
    from rrt_echo.runtime import Deadline

    packet = make_packet("a")
    attempts = []

    def complete(prompt, *args, **kwargs):
        attempts.append(prompt)
        if len(attempts) == 1:
            return {
                "instances": [],
                "facts": [
                    {
                        "fact_id": "bad",
                        "kind": "event",
                        "predicate": "carry",
                        "roles": {"agent": "absent"},
                        "evidence_by_slot": {},
                        "joint_evidence": [],
                        "observed_media_ids": [],
                    }
                ],
            }
        return {"instances": [], "facts": []}

    client = VisionClient(model="fake")
    monkeypatch.setattr(client, "complete", complete)
    result = client.observe("a", "v", "s", packet.media, deadline=Deadline(20))
    assert not result.facts
    assert len(attempts) == 2
    assert "unknown local participant" in attempts[1]


def test_request_aliases_restore_canonical_media(monkeypatch):
    from rrt_echo.perception.vlm import VisionClient
    from rrt_echo.runtime import Deadline

    packet = make_packet("a")
    seen = []

    def complete(prompt, media, **kwargs):
        seen.extend(m.media_id for m in media)
        return {
            "instances": [
                {
                    "instance_id": "p",
                    "kind": "person",
                    "regions": [{"media_id": "f0", "box": [0.1, 0.1, 0.4, 0.9]}],
                }
            ],
            "facts": [],
        }

    client = VisionClient(model="fake")
    monkeypatch.setattr(client, "complete", complete)
    result = client.observe("a", "v", "s", packet.media, deadline=Deadline(20))
    assert seen == ["f0"]
    assert result.instances[0].regions[0].media_id == packet.media[0].media_id


def test_shot_proposals_suppress_adjacent_motion_peaks():
    from rrt_echo.perception.video import FrameIndex, assign_shots

    rows = [FrameIndex(i, i, (1, 25), i / 25, 0, 0.01) for i in range(60)]
    from dataclasses import replace

    rows[20] = replace(rows[20], motion=0.4)
    rows[21] = replace(rows[21], motion=0.3)
    rows[45] = replace(rows[45], motion=0.25)
    actual = assign_shots(rows)
    assert actual[19].shot_id == 0
    assert actual[20].shot_id == actual[21].shot_id == 1
    assert actual[45].shot_id == 2
    assert [r.pts for r in actual] == [r.pts for r in rows]


def test_staged_integer_boxes_are_normalized_and_not_clamped(monkeypatch):
    from rrt_echo.perception.vlm import VisionClient
    from rrt_echo.runtime import Deadline

    packet = make_packet("a")
    responses = iter(
        [
            {
                "instances": [
                    {
                        "instance_id": "1",
                        "kind": "person",
                        "description": "visible person",
                        "regions": [{"media_id": "f0", "box": [100, 200, 600, 900]}],
                    }
                ]
            },
            {"facts": []},
        ]
    )
    client = VisionClient(model="fake", staged=True)
    monkeypatch.setattr(client, "complete", lambda *a, **kw: next(responses))
    result = client.observe("o", "v", "s", packet.media, deadline=Deadline(20))
    assert result.instances[0].regions[0].box == (0.1, 0.2, 0.6, 0.9)


def test_scoped_identity_compares_two_targets_and_preserves_sources():
    from rrt_echo.reconcile import compare_scoped
    from rrt_echo.runtime import Deadline

    class Client:
        calls = []

        def complete(self, prompt, media, **kwargs):
            assert [m.media_id for m in media] == ["A", "B"]
            assert len(media) == 2
            return {
                "person_A": "adult",
                "person_B": "infant",
                "verdict": "different",
                "visual_basis": "two distinct people",
            }

    crops = {
        key: {
            "uri": "/unused",
            "pts": 0,
            "time_base": [1, 25],
            "sha256": "a" * 64,
            "source_media_id": frame,
        }
        for key, frame in [("left", "source1"), ("right", "source2")]
    }
    result = compare_scoped(Client(), "p", "left", "right", crops, Deadline(20))
    assert result.verdict == "different"
    assert result.media_ids == ("source1", "source2")


def test_identity_journal_restore_preserves_history_and_rejects_tampering():
    import copy

    import pytest

    from rrt_echo.identity import IdentityGraph, IdentityProposal

    memory = Memory()
    memory.append(make_packet("a"))
    memory.append(make_packet("b"))
    left, right = list(memory.instances)
    evidence = tuple(m.media_id for p in memory.packets for m in p.media)
    decision = memory.compare(
        IdentityProposal("same", left, right, "same", evidence, "visible correspondence")
    )
    memory.identity.revoke(decision.decision_id, reason="audit correction")
    rows = memory.identity.export()
    # Wire serialization turns immutable tuples into JSON arrays.
    import json

    rows = json.loads(json.dumps(rows))
    restored = IdentityGraph.restore(rows, memory.instances)
    assert restored.export() == memory.identity.export()
    broken = copy.deepcopy(rows)
    broken[0]["status"] = "deferred"
    with pytest.raises(ValueError):
        IdentityGraph.restore(broken, memory.instances)
