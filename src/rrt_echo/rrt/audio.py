"""Timestamped ASR and anonymous voice hypotheses, independent of visual identity.

Uses the local M3-Agent Whisper/ERes2NetV2 checkpoints and its Apache-2.0
speakerlab implementation. ASR segments are NOT guaranteed speaker turns.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
import sys
from pathlib import Path

from ..schema import digest
from ..storage import atomic_json

RULES = (
    "Speech is an ASR hypothesis about what was said, not proof that its content is true. "
    "Voice clusters are anonymous, provisional and video-local; they are not visible people, "
    "player names, or named referents in the transcript. No audiovisual identity is accepted. "
    "Mixed/short/ambiguous segments have unresolved speakers. Overlapping speech is not separated."
)


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def decode_audio(video):
    """Decode on the source timeline, preserving stream offset and timestamp gaps."""
    import av
    import numpy as np

    with av.open(str(video)) as container:
        if not container.streams.audio:
            return None, {"status": "no_audio_stream"}
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                if output.pts is None:
                    raise ValueError("audio_timestamp_missing")
                chunks.append((float(output.pts * output.time_base), output.to_ndarray().ravel()))
        for output in resampler.resample(None):
            chunks.append((float(output.pts * output.time_base), output.to_ndarray().ravel()))
        if not chunks:
            return None, {"status": "empty_audio_stream"}
        start = chunks[0][0]
        end = max(t + len(a) / 16000 for t, a in chunks)
        pcm = np.zeros(round((end - start) * 16000), dtype=np.float32)
        for t, a in chunks:
            i = round((t - start) * 16000)
            if i < 0:
                raise ValueError("non_monotonic_audio_timestamp")
            pcm[i : i + len(a)] = a[: len(pcm) - i]
        return pcm, {
            "status": "decoded",
            "stream_index": stream.index,
            "codec": stream.codec_context.name,
            "sample_rate": 16000,
            "start_seconds": start,
            "end_seconds": end,
        }


def speaker_windows(utterances, duration=1.5, step=0.75):
    """Overlapping windows on speech spans, independent of ASR sentence splits."""
    spans = []
    for u in utterances:
        if spans and u["start"] - spans[-1][1] <= 0.3:
            spans[-1][1] = max(spans[-1][1], u["end"])
        else:
            spans.append([u["start"], u["end"]])
    windows = []
    for start, end in spans:
        if end - start < duration:
            continue
        t = start
        while t + duration <= end + 1e-6:
            windows.append((t, t + duration))
            t += step
        if not windows or abs(windows[-1][1] - end) > 0.05:
            windows.append((end - duration, end))
    return windows


def spectral_voices(embeddings, max_speakers=20):
    """M3 speakerlab-style pruned affinity/eigengap, without oracle speaker count.

    Cluster memberships are hypotheses; confidence gates are applied separately.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    x = np.asarray(embeddings, dtype=float)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    if len(x) < 3:
        return np.arange(len(x)), {"estimated_clusters": len(x), "reason": "too_few_windows"}
    sim = np.maximum(x @ x.T, 0)
    np.fill_diagonal(sim, 0)
    neighbors = min(len(x) - 1, max(6, int(len(x) * 0.02)))
    keep = np.argsort(sim, axis=1)[:, -neighbors:]
    affinity = np.zeros_like(sim)
    np.put_along_axis(affinity, keep, np.take_along_axis(sim, keep, axis=1), axis=1)
    affinity = (affinity + affinity.T) / 2
    laplacian = np.diag(affinity.sum(axis=1)) - affinity
    values, vectors = np.linalg.eigh(laplacian)
    upper = min(max_speakers, len(x) - 1)
    k = int(np.argmax(np.diff(values[: upper + 1]))) + 1
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(vectors[:, :k])
    return labels, {
        "estimated_clusters": k,
        "max_speakers_search": max_speakers,
        "eigenvalues": values[: upper + 1].tolist(),
        "knn": neighbors,
    }


def diarize(utterances, pcm, offset, voice, feature, device, video_id, threshold):
    import numpy as np
    import torch

    windows = speaker_windows(utterances)
    embeddings = []
    for a, b in windows:
        wav = torch.from_numpy(
            pcm[round((a - offset) * 16000) : round((b - offset) * 16000)].copy()
        )[None]
        with torch.inference_mode():
            emb = voice(feature(wav).unsqueeze(0).to(device)).squeeze().cpu().numpy()
        embeddings.append(emb / max(float(np.linalg.norm(emb)), 1e-8))
    if not embeddings:
        return [], [], {"estimated_clusters": 0}
    labels, diagnostics = spectral_voices(embeddings)
    x = np.asarray(embeddings)
    centroids = np.array([x[labels == i].mean(axis=0) for i in range(max(labels) + 1)])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    similarities = x @ centroids.T
    evidence = []
    for n, ((a, b), label) in enumerate(zip(windows, labels)):
        score = float(similarities[n, label])
        others = np.delete(similarities[n], label)
        margin = score - float(others.max()) if len(others) else 1.0
        accepted = score >= threshold and margin >= 0.06
        evidence.append(
            {
                "start": a,
                "end": b,
                "speaker_id": f"{video_id}:voice:{label}" if accepted else None,
                "candidate_speaker_id": f"{video_id}:voice:{label}",
                "similarity": score,
                "margin": margin,
                "embedding": embeddings[n].tolist(),
            }
        )
    for u in utterances:
        # A short utterance may be covered by a longer acoustic window; disagreement
        # near a turn boundary remains unresolved rather than propagating identity.
        relevant = [
            w
            for w in evidence
            if min(u["end"], w["end"]) - max(u["start"], w["start"])
            >= min(0.35, (u["end"] - u["start"]) / 2)
        ]
        ids = {w["speaker_id"] for w in relevant}
        speaker = next(iter(ids)) if len(ids) == 1 and None not in ids else None
        u["speaker_id"] = speaker
        u["speaker_status"] = "provisional_window_consensus" if speaker else "unresolved"
        u["speaker_windows"] = [{k: v for k, v in w.items() if k != "embedding"} for w in relevant]
    speakers = [
        {
            "speaker_id": f"{video_id}:voice:{i}",
            "status": "provisional_voice_cluster",
            "visual_entity": None,
            "embedding_windows": int(sum(labels == i)),
        }
        for i in range(max(labels) + 1)
    ]
    return speakers, evidence, diagnostics


def transcribe(video, m3_root, *, device="cuda", language=None, threshold=0.6):
    video, m3_root = Path(video), Path(m3_root)
    pcm, source = decode_audio(video)
    result = {
        "schema": "rrt.audio.v1",
        "source_video_id": video.stem,
        "source": {**source, "path": str(video.resolve()), "sha256": file_hash(video)},
        "rules": RULES,
        "utterances": [],
        "speakers": [],
        "audiovisual_links": [],
        "media_verified": False,
    }
    if pcm is None:
        result["artifact_id"] = digest(result)
        return result
    import torch
    from faster_whisper import WhisperModel

    sys.path.insert(0, str(m3_root.resolve()))
    from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
    from speakerlab.process.processor import FBank

    ckpt = m3_root / "models/pretrained_eres2netv2.ckpt"
    whisper_path = m3_root / "models/faster-whisper-medium"
    voice = ERes2NetV2(feat_dim=80, embedding_size=192)
    voice.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    voice = voice.to(device).eval()
    feature = FBank(80, sample_rate=16000, mean_nor=True)
    model = WhisperModel(
        str(whisper_path), device=device, compute_type="float16" if device == "cuda" else "int8"
    )
    segments, info = model.transcribe(
        pcm,
        language=language,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    result["config"] = {
        "asr": str(whisper_path),
        "asr_model_sha256": file_hash(whisper_path / "model.bin"),
        "speaker_model_sha256": file_hash(ckpt),
        "threshold": threshold,
        "margin": 0.06,
        "min_voice_seconds": 1.5,
        "window_seconds": 1.5,
        "window_step": 0.75,
        "clustering": "offline_spectral_window_consensus",
        "language": info.language,
        "language_probability": info.language_probability,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    offset = source["start_seconds"]
    for n, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        s, e = float(segment.start), float(segment.end)
        windows, speaker = [], None
        uid = f"{video.stem}:speech:{n}"
        row = {
            "utterance_id": uid,
            "predicate": "says",
            "speaker_id": speaker,
            "text": text,
            "start": s + offset,
            "end": e + offset,
            "speaker_status": "provisional"
            if speaker
            else ("mixed_or_ambiguous" if windows else "too_short"),
            "speaker_windows": windows,
            "avg_logprob": segment.avg_logprob,
            "no_speech_prob": segment.no_speech_prob,
            "words": [
                {
                    "text": w.word,
                    "start": w.start + offset,
                    "end": w.end + offset,
                    "probability": w.probability,
                }
                for w in (segment.words or [])
            ],
            "evidence": {
                "source_sha256": result["source"]["sha256"],
                "stream_index": source["stream_index"],
                "start": s + offset,
                "end": e + offset,
            },
        }
        result["utterances"].append(row)
        print(f"{n}: {s:.2f}-{e:.2f} {speaker or 'unresolved'}: {text}", flush=True)
    result["speakers"], result["voice_evidence"], result["speaker_diagnostics"] = diarize(
        result["utterances"], pcm, offset, voice, feature, device, video.stem, threshold
    )
    result["source"]["status"] = "transcribed"
    result["artifact_id"] = digest(result)
    return result


def validate_audio(audio):
    if audio.get("artifact_id") != digest({k: v for k, v in audio.items() if k != "artifact_id"}):
        raise ValueError("audio_digest_mismatch")
    speakers = {s["speaker_id"] for s in audio["speakers"]}
    if len(speakers) != len(audio["speakers"]):
        raise ValueError("duplicate_audio_speaker")
    ids = set()
    for u in audio["utterances"]:
        if u["utterance_id"] in ids:
            raise ValueError("duplicate_utterance")
        ids.add(u["utterance_id"])
        if not all(math.isfinite(u[k]) for k in ("start", "end")) or u["end"] < u["start"]:
            raise ValueError("invalid_utterance_time")
        if u["speaker_id"] is not None and u["speaker_id"] not in speakers:
            raise ValueError("dangling_audio_speaker")
        if any(u["evidence"][k] != u[k] for k in ("start", "end")):
            raise ValueError("utterance_evidence_time_mismatch")
    if audio["source"]["status"] == "no_audio_stream" and audio["utterances"]:
        raise ValueError("speech_without_audio")


def attach_audio(graph, audio):
    expected = graph.get("source_video_id") or graph["video_id"]
    if expected != audio["source_video_id"]:
        raise ValueError("audio_video_id_mismatch")
    validate_audio(audio)
    # Check source identity against visual evidence when source hashes are available.
    hashes = {m.get("source_sha256") for m in graph.get("evidence", {}).values()} - {None}
    if hashes and audio["source"]["sha256"] not in hashes:
        raise ValueError("audio_video_hash_mismatch")
    out = copy.deepcopy(graph)
    out["audio_memory"] = copy.deepcopy(audio)
    out.pop("snapshot_id", None)
    out["snapshot_id"] = digest(out)
    return out


def retrieve_audio(graph, query, ranges=(), top_k=12):
    audio = graph.get("audio_memory")
    if not audio:
        return None
    tokens = set(re.findall(r"\w+", query.casefold())) - {
        "the",
        "a",
        "is",
        "in",
        "to",
        "of",
        "and",
        "what",
        "who",
    }
    candidates = [
        u
        for u in audio["utterances"]
        if not ranges or any(u["end"] >= a and u["start"] <= b for a, b in ranges)
    ]
    ranked = sorted(
        candidates,
        key=lambda u: (-len(tokens & set(re.findall(r"\w+", u["text"].casefold()))), u["start"]),
    )
    selected = sorted(ranked[:top_k], key=lambda u: u["start"])
    ids = {u["speaker_id"] for u in selected}
    return {
        "rules": RULES,
        "source_status": audio["source"]["status"],
        "utterances": [
            {k: v for k, v in u.items() if k not in {"words", "speaker_windows"}} for u in selected
        ],
        "speakers": [s for s in audio["speakers"] if s["speaker_id"] in ids],
        "candidate_count": len(candidates),
        "omitted_count": max(0, len(candidates) - len(selected)),
        "retrieval": "lexical_and_time_overlap",
        "audiovisual_links": [],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--m3-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--graph")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--language")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    result = transcribe(a.video, a.m3_root, device=a.device, language=a.language)
    atomic_json(out / "audio.json", result)
    if a.graph:
        from ..storage import read_graph

        atomic_json(out / "hypergraph.json", attach_audio(read_graph(Path(a.graph)), result))
    atomic_json(
        out / "summary.json",
        {
            "status": result["source"]["status"],
            "utterances": len(result["utterances"]),
            "voice_clusters": len(result["speakers"]),
            "unresolved_speaker_utterances": sum(
                u["speaker_id"] is None for u in result["utterances"]
            ),
            "media_verified": False,
        },
    )


if __name__ == "__main__":
    main()
