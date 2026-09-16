import json

from PIL import Image

from rrt_echo.echo_perception.types import Clip, Media, digest
from rrt_echo.rrt.observation import EventVisionClient


def setup(tmp_path):
    frames = []
    for n in range(3):
        p = tmp_path / f"{n}.jpg"
        Image.new("RGB", (80, 60)).save(p)
        frames.append(Media(f"v:f{n}", str(p), n, (1, 1), digest(p), n))
    clip = Clip("v:clip", "v", 0, 0, 3, tuple(frames), 1, 3, 3, 0, True, {})
    raw = {
        "instances": [
            {"id": "i0", "kind": "person", "description": "baby"},
            {"id": "i1", "kind": "object", "description": "paper"},
        ],
        "regions": [
            {"id": "r0", "instance": "i0", "frame": "f0", "box": [100, 400, 500, 900]},
            {"id": "r1", "instance": "i1", "frame": "f0", "box": [500, 500, 900, 900]},
        ],
        "events": [
            {
                "predicate": "bite",
                "frames": ["f0"],
                "roles": [{"role": "agent", "region": "r0"}, {"role": "patient", "region": "r1"}],
                "unresolved_slots": [],
            }
        ],
        "properties": [],
        "links": [],
        "gaps": [],
    }
    return clip, raw


def test_regions_and_roles_survive_to_artifact(tmp_path):
    clip, raw = setup(tmp_path)
    client = EventVisionClient(model="test", mode="images")
    result = client.observe(
        clip,
        out=tmp_path / "calls",
        sender=lambda b: {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]
        },
    )
    fact = result["observation"]["facts"][0]
    assert fact["roles"] == {"agent": "v:clip:i0", "patient": "v:clip:i1"}
    assert result["fact_region_bindings"][fact["fact_id"]]["patient"] == "v:clip:r1"
    assert len(result["spatial_observations"]) == 2


def test_fact_cannot_create_missing_region(tmp_path):
    clip, raw = setup(tmp_path)
    raw["events"][0]["roles"][1]["region"] = "r3"
    client = EventVisionClient(model="test", mode="images")
    _, audit = client.build_request(clip)
    result = client.decode_output(json.dumps(raw), audit)
    assert not result["facts"]
    assert result["coverage_gaps"]
    assert len(result["instances"][1]["regions"]) == 1


def test_no_cross_occurrence_evidence(tmp_path):
    clip, raw = setup(tmp_path)
    raw["regions"][1]["frame"] = "f1"
    client = EventVisionClient(model="test", mode="images")
    _, audit = client.build_request(clip)
    assert not client.decode_output(json.dumps(raw), audit)["facts"]


def test_occurrence_region_ids_are_scoped_and_preserve_endpoints(tmp_path):
    import copy

    from rrt_echo.rrt.observation import OccurrenceVisionClient

    clip, raw = setup(tmp_path)
    regions = raw.pop("regions")
    first = raw["events"][0]
    first["regions"] = regions
    second = copy.deepcopy(first)
    second["frames"] = ["f1"]
    for region in second["regions"]:
        region["frame"] = "f1"
    raw["events"].append(second)
    client = OccurrenceVisionClient(model="test", mode="images")
    result = client.observe(
        clip,
        out=tmp_path / "calls",
        sender=lambda body: {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]
        },
    )
    assert len(result["observation"]["facts"]) == 2
    assert result["fact_region_bindings"]["v:clip:event:0"]["patient"] == "v:clip:r1"
    assert result["fact_region_bindings"]["v:clip:event:1"]["patient"] == "v:clip:r3"
    assert result["observation"]["facts"][1]["roles"]["patient"] == "v:clip:i1"


def test_multimodal_context_is_navigation_not_target_evidence(tmp_path):
    from rrt_echo.rrt.multimodal import observation_context
    from rrt_echo.rrt.observation import OccurrenceVisionClient

    clip, raw = setup(tmp_path)
    audio = {
        "utterances": [
            {"utterance_id": "u1", "text": "Harry is medic", "start": 1.1, "end": 1.9},
            {"utterance_id": "u2", "text": "future fact", "start": 8, "end": 9},
        ]
    }
    context = observation_context([], audio, 1, 3)
    client = OccurrenceVisionClient(model="test", mode="images", multimodal_context=context)
    body, audit = client.build_request(clip)
    assert len(audit["multimodal_context"]["speech_observations"]) == 1
    assert "Harry is medic" in body["messages"][0]["content"][-1]["text"]
    assert "future fact" not in body["messages"][0]["content"][-1]["text"]
    assert set(audit["source_media_aliases"]) == {"f0", "f1", "f2"}
    assert context["speech_observations"][0]["speaker_identity"] == "unresolved"


def test_typed_ownership_maps_part_to_whole_without_role_swapping(tmp_path):
    from rrt_echo.rrt.observation import OccurrenceVisionClient

    clip, raw = setup(tmp_path)
    raw["instances"][0]["description"] = "person"
    raw["instances"][1].update(kind="region", description="forearm")
    regions = raw.pop("regions")
    raw["events"][0]["regions"] = regions
    raw["ownership"] = [
        {
            "part": "i1",
            "whole": "i0",
            "verdict": "supported",
            "evidence_ids": ["f0"],
            "basis": "visible bodily connection",
        }
    ]
    client = OccurrenceVisionClient(
        model="test", mode="images", typed_ownership=True, processor_max_pixels=401408
    )
    body, audit = client.build_request(clip)
    assert body["mm_processor_kwargs"]["max_pixels"] == 401408
    assert (
        "part_of"
        not in audit["wire_schema"]["properties"]["links"]["items"]["properties"]["relation"][
            "enum"
        ]
    )
    result = client.observe(
        clip,
        out=tmp_path / "calls",
        sender=lambda body: {
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]
        },
    )
    link = result["link_proposals"][0]
    assert (link["source"], link["target"], link["relation"]) == (
        "v:clip:i1",
        "v:clip:i0",
        "part_of",
    )
    raw["ownership"][0].update(part="i0", whole="i1")
    bad = client.decode_output(json.dumps(raw), audit)
    assert not bad["links"]
    assert bad["facts"]
