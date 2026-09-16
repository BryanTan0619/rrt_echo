import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import make_packet

from rrt_echo.cli import main, read_graph
from rrt_echo.perception.tracking import LocalTracker
from rrt_echo.perception.video import decode_selected, scan, select_windows
from rrt_echo.perception.vlm import VisionClient
from rrt_echo.pipeline import DEFAULT_CONFIG, build, load_config
from rrt_echo.runtime import Deadline, supervise
from rrt_echo.storage import atomic_json


def test_motion_tracker_is_one_to_one_and_resets_at_shots():
    tracker = LocalTracker()
    a = tracker.update([(0.1, 0.1, 0.3, 0.8), (0.6, 0.1, 0.8, 0.8)], seconds=0, shot_id=0)
    b = tracker.update([(0.12, 0.1, 0.32, 0.8), (0.62, 0.1, 0.82, 0.8)], seconds=0.5, shot_id=0)
    assert [x["track_ref"] for x in a] == [x["track_ref"] for x in b]
    assert len({x["track_ref"] for x in b}) == 2
    c = tracker.update([(0.12, 0.1, 0.32, 0.8)], seconds=1, shot_id=1)
    assert c[0]["track_ref"] not in {x["track_ref"] for x in a}


def test_process_budget_terminates_and_preserves_checkpoint(tmp_path):
    atomic_json(tmp_path / "hypergraph.json", {"snapshot_id": "last-good"})
    atomic_json(tmp_path / "progress.json", {"pending": {"window": 7}})
    started = time.monotonic()
    result = supervise([sys.executable, "-c", "import time;time.sleep(10)"], tmp_path, seconds=0.3)
    assert result == 124 and time.monotonic() - started < 3
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["snapshot_id"] == "last-good"
    assert status["gaps"][0]["pending"] == {"window": 7}


def test_replay_cli_is_self_contained_and_does_not_overwrite(tmp_path):
    examples = Path(__file__).parents[1] / "examples"
    out = tmp_path / "demo"
    assert (
        main(
            [
                "replay",
                str(examples / "observations.jsonl"),
                "--identity",
                str(examples / "identity.jsonl"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    graph = read_graph(out / "hypergraph.json")
    assert len(graph["facts"]) == 3
    assert len(graph["entities"]) == 4  # adult A, adult B, car, child
    assert not (out / "latest.json").exists()
    with pytest.raises(FileExistsError):
        main(["replay", str(examples / "observations.jsonl"), "--out", str(out)])


def test_snapshot_hash_detects_modification(tmp_path):
    path = tmp_path / "graph.json"
    atomic_json(path, {"snapshot_id": "not-a-hash", "facts": []})
    with pytest.raises(ValueError):
        read_graph(path)


def test_video_adapter_preserves_source_pts(tmp_path):
    import av

    video = tmp_path / "tiny.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), i * 15, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    deadline = Deadline(30)
    records = scan(video, deadline)
    windows, report = select_windows(records, max_frames=4)
    decoded = decode_selected(video, tmp_path / "frames", "test", windows, deadline)
    assert len(records) == 12
    assert all(len(m) <= 4 for m in decoded)
    assert report
    by_pts = {r.pts: r for r in records}
    for media in decoded:
        for m in media:
            assert m.seconds == pytest.approx(by_pts[m.pts].seconds)
            assert Path(m.uri).exists()


def test_observer_cannot_accept_model_global_identity(monkeypatch):
    client = VisionClient(model="fake")
    monkeypatch.setattr(
        client,
        "complete",
        lambda *a, **kw: {"instances": [], "facts": [], "identity_decisions": []},
    )
    with pytest.raises(ValueError):
        client.observe("o", "v", "s", (), deadline=Deadline(20))


def test_pipeline_checkpoint_and_local_failure_are_visible(tmp_path, monkeypatch):
    import rrt_echo.pipeline as pipeline

    class FakeClient:
        def __init__(self, **kwargs):
            self.calls = []

        def observe(self, oid, vid, shot, media, **kwargs):
            if oid.endswith("1"):
                raise ValueError("malformed local output")
            return make_packet(oid)

    monkeypatch.setattr(pipeline, "VisionClient", FakeClient)
    monkeypatch.setattr(pipeline, "prepare", lambda *args: ("v", [("0", ()), ("1", ())]))
    video = tmp_path / "fake-video"
    video.write_bytes(b"test stub: not decoded")
    out = tmp_path / "run"
    out.mkdir()
    assert build(video, out, DEFAULT_CONFIG | {"budget_seconds": 30, "model": "fake"}) == 2
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "partial"
    assert status["gaps"][0]["reason"] == "local_extraction_failed"
    assert len(read_graph(out / "hypergraph.json")["facts"]) == 1


def test_config_rejects_unknown_and_unbounded_run(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"budget_seconds":99999}')
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text('{"some_misspelled_setting":true}')
    with pytest.raises(ValueError):
        load_config(p)


def test_face_features_do_not_assign_ambiguous_composite_owner():
    from rrt_echo.perception.features import attach_face_candidates
    from rrt_echo.schema import Instance, Region

    instances = (
        Instance("adult", "person", (Region("f", (0.1, 0.1, 0.9, 0.9)),)),
        Instance("baby", "person", (Region("f", (0.3, 0.3, 0.6, 0.6)),)),
    )
    face = {"media_id": "f", "box": [0.4, 0.4, 0.5, 0.5], "embedding": [1.0, 0.0]}
    features, unresolved = attach_face_candidates(instances, [face])
    assert not features
    assert unresolved[0]["candidate_owners"] == ["adult", "baby"]


def test_face_backend_missing_weights_never_downloads(tmp_path):
    from rrt_echo.perception.features import InsightFaceFeatures

    with pytest.raises(FileNotFoundError):
        InsightFaceFeatures(str(tmp_path))


def test_registration_ignores_model_claimed_track_and_preserves_shot_local_key():
    from rrt_echo.perception.registration import register_packet

    p = make_packet("a")
    raw = p.to_dict()
    raw["instances"][0]["track_ref"] = "hallucinated_tracker"
    from rrt_echo.schema import ObservationPacket

    p = ObservationPacket.from_dict(raw)
    assert register_packet(p, []).instances[0].track_ref is None
    detections = [{"media_id": "a:frame", "track_ref": "body_track_1", "box": [0.1, 0.1, 0.4, 0.9]}]
    registered = register_packet(p, detections)
    assert registered.instances[0].track_ref == "v:shot:a:body_track_1"
    assert p.instances[0].track_ref == "hallucinated_tracker"


def test_qa_filters_dataset_by_graph_video_and_never_sends_gold(tmp_path, monkeypatch):
    from rrt_echo.memory import Memory
    from rrt_echo.storage import export_run

    memory = Memory(source_video_id="wanted")
    memory.append(make_packet())
    export_run(memory, tmp_path / "memory", status="complete")
    questions = tmp_path / "qa.jsonl"
    questions.write_text(
        "\n".join(
            json.dumps(
                {
                    "video_id": vid,
                    "question": "Which direction?",
                    "options": {"A": "correct action", "B": "other action"},
                    "ans": "A",
                }
            )
            for vid in ("wanted", "other")
        )
    )
    prompts = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.calls = []

        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            return {"answer": "A"} if len(prompts) == 1 else {"assertions": []}

    import rrt_echo.perception.vlm as vlm

    monkeypatch.setattr(vlm, "VisionClient", FakeClient)
    assert (
        main(
            [
                "qa",
                str(tmp_path / "memory" / "hypergraph.json"),
                str(questions),
                "--out",
                str(tmp_path / "answers"),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "answers" / "summary.json").read_text())
    assert summary["questions"] == 1 and summary["accuracy"] == 1
    assert summary["correct_with_support"] is None
    assert all('"ans"' not in p for p in prompts)


def test_identity_comparison_uses_neutral_endpoints_and_grounded_crops(monkeypatch, tmp_path):
    import hashlib
    from dataclasses import replace

    from PIL import Image

    packet = make_packet("crop")
    path = tmp_path / "source.jpg"
    Image.new("RGB", (100, 100), "white").save(path)
    source = replace(
        packet.media[0], uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    left = replace(packet.instances[0], description="future father story cue")
    right = replace(packet.instances[0], instance_id="other", description="future mother story cue")
    captured = {}

    def complete(prompt, media, **kwargs):
        captured.update(prompt=prompt, media=media, **kwargs)
        assert [m.media_id for m in media] == ["A", "B"]
        assert all(Image.open(m.uri).size == (30, 80) for m in media)
        return {
            "verdict": "unresolved",
            "person_A": "unclear",
            "person_B": "unclear",
            "visual_basis": "insufficient",
        }

    client = VisionClient(model="fake")
    monkeypatch.setattr(client, "complete", complete)
    result = client.compare("id", left, right, (source,), deadline=Deadline(20))
    assert result.verdict == "unresolved"
    assert "future father story cue" not in captured["prompt"]
    assert "future mother story cue" not in captured["prompt"]
    assert result.media_ids == (source.media_id,)
