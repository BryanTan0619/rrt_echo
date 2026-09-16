import copy

import numpy as np
import pytest

from rrt_echo.evaluation import audit_assertions
from rrt_echo.rrt.audio import attach_audio, decode_audio, retrieve_audio
from rrt_echo.rrt.reader import compact_payload
from rrt_echo.schema import digest


def artifact():
    a = {
        "source_video_id": "video",
        "source": {"sha256": "a" * 64, "status": "transcribed"},
        "utterances": [
            {
                "utterance_id": "video:speech:0",
                "speaker_id": "video:voice:0",
                "text": "I vote for the red player",
                "start": 10.25,
                "end": 11.15,
                "evidence": {"start": 10.25, "end": 11.15},
            }
        ],
        "speakers": [{"speaker_id": "video:voice:0", "visual_entity": None}],
    }
    a["artifact_id"] = digest(a)
    return a


def test_audio_scope_and_short_utterance_survives():
    graph = {"audio_memory": artifact()}
    assert retrieve_audio(graph, "vote", [(10.3, 10.5)])["utterances"][0]["start"] == 10.25
    assert retrieve_audio(graph, "vote", [(20, 21)])["utterances"] == []
    assert retrieve_audio(graph, "vote")["speakers"][0]["visual_entity"] is None


def test_attach_digest_and_identity():
    graph = {"video_id": "video", "evidence": {}}
    original = copy.deepcopy(graph)
    enriched = attach_audio(graph, artifact())
    assert graph == original
    assert enriched["snapshot_id"] == digest(
        {k: v for k, v in enriched.items() if k != "snapshot_id"}
    )
    with pytest.raises(ValueError, match="video_id"):
        attach_audio({"video_id": "different"}, artifact())
    bad = artifact()
    bad["utterances"][0]["text"] = "modified"
    with pytest.raises(ValueError, match="digest"):
        attach_audio(graph, bad)


def test_audio_aliases_and_citation_scope():
    payload = {"audio_memory": retrieve_audio({"audio_memory": artifact()}, "vote")}
    compact, aliases = compact_payload(payload)
    alias = compact["audio_memory"]["utterances"][0]["utterance_id"]
    assert aliases[alias] == "video:speech:0"
    a = [
        {
            "assertion": "A voice says vote",
            "evidence_chain": ["video:speech:0"],
            "support_verdict": "supported",
        }
    ]
    assert audit_assertions(a, payload)["support_verdict"] == "supported"
    assert audit_assertions(a, {})["support_verdict"] == "insufficient"
    assert audit_assertions(a, payload)["strict_support"] is None


def test_video_without_audio(tmp_path):
    import av

    path = tmp_path / "silent.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for _ in range(2):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    pcm, status = decode_audio(path)
    assert pcm is None
    assert status["status"] == "no_audio_stream"


def test_windows_do_not_bridge_long_silence():
    from rrt_echo.rrt.audio import speaker_windows

    windows = speaker_windows(
        [{"start": 0.1, "end": 0.9}, {"start": 1.0, "end": 2.1}, {"start": 10.0, "end": 10.5}]
    )
    assert windows
    assert all(0.1 <= a < b <= 2.1 for a, b in windows)


def test_decoder_preserves_audio_offset(tmp_path):
    from fractions import Fraction

    import av

    path = tmp_path / "offset.mkv"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("pcm_s16le", rate=16000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(
            np.ones((1, 16000), dtype=np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = 16000
        frame.time_base = Fraction(1, 16000)
        frame.pts = 32000
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    pcm, status = decode_audio(path)
    assert status["start_seconds"] == pytest.approx(2, abs=0.002)
    assert len(pcm) == pytest.approx(16000, abs=2)
