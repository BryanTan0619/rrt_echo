"""One offline pipeline. Bounded calls, atomic checkpoints, no QA during build."""

from __future__ import annotations

import json
from pathlib import Path

from .memory import Memory
from .perception.features import InsightFaceFeatures, attach_face_candidates, cosine
from .perception.registration import register_packet
from .perception.tracking import LocalTracker, TorchvisionDetector
from .perception.video import file_hash, prepare
from .perception.vlm import VisionClient
from .runtime import Deadline
from .schema import digest
from .storage import atomic_json, export_run

DEFAULT_CONFIG = {
    "budget_seconds": 3600,
    "window_seconds": 8,
    "base_fps": 1,
    "high_fps": 8,
    "max_frames": 16,
    "shot_threshold": 0.10,
    "max_output_tokens": 4096,
    "image_max_edge": 1024,
    "staged": True,
    "joint": False,
    "max_identity_comparisons": 400,
    "candidate_limit": 3,
    "identity_kinds": ["person", "object"],
    "face_model_root": None,
    "face_provider": "CPUExecutionProvider",
    "detector_weights": None,
    "detector_device": "cpu",
    "detector_architecture": "v1",
    "model": None,
    "base_url": None,
}


def load_config(path: Path | None):
    raw = json.loads(path.read_text()) if path else {}
    if set(raw) - set(DEFAULT_CONFIG):
        raise ValueError(f"unknown configuration fields: {set(raw) - set(DEFAULT_CONFIG)}")
    result = DEFAULT_CONFIG | raw
    if not 10 <= result["budget_seconds"] <= 3600:
        raise ValueError("construction budget must be 10..3600 seconds")
    if result["candidate_limit"] < 1 or result["max_identity_comparisons"] < 0:
        raise ValueError("invalid identity budget")
    return result


def identity_candidates(memory: Memory, *, limit=3, face_features=None, kinds=None):
    """Appearance/track cues rank candidates, never accept identity.

    Full-video availability allows distant comparisons. All instances remain in
    the graph when their comparison is not scheduled within the budget.
    """
    face_features = face_features or {}
    instances = [i for i in memory.instances.values() if kinds is None or i.kind in kinds]
    pairs = []
    for i, current in enumerate(instances):
        candidates = []
        words = set(current.description.casefold().split())
        for previous in instances[:i]:
            if previous.kind != current.kind or not previous.regions or not current.regions:
                continue
            shared_track = bool(current.track_ref and current.track_ref == previous.track_ref)
            shared_words = len(words & set(previous.description.casefold().split()))
            similarity = max(
                (
                    cosine(a, b)
                    for a in face_features.get(current.instance_id, ())
                    for b in face_features.get(previous.instance_id, ())
                ),
                default=0.0,
            )
            candidates.append(
                (
                    -(8 * similarity + 4 * shared_track + shared_words),
                    previous.instance_id,
                    previous,
                )
            )
        ranked = sorted(candidates, key=lambda x: x[:2])[:limit]
        for score, _, previous in ranked:
            pairs.append(
                (score, previous, current, tuple(x[2] for x in ranked if x[2] != previous))
            )
    return sorted(pairs, key=lambda x: (x[0], x[1].instance_id, x[2].instance_id))


def build(video: Path, out: Path, config: dict):
    deadline = Deadline(config["budget_seconds"])
    memory, gaps = Memory(source_video_id=video.stem), []
    export_run(memory, out)
    source_hash = file_hash(video)
    package = Path(__file__).parent
    atomic_json(
        out / "manifest.json",
        {
            "config": config,
            "video": str(video.resolve()),
            "video_sha256": source_hash,
            "code_sha256": {
                str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))
            },
            "query_blind": True,
            "qa_video_access": False,
            "cache_mode": "cold",
            "config_id": digest(config),
        },
    )
    client = VisionClient(
        base_url=config["base_url"],
        model=config["model"],
        max_tokens=config["max_output_tokens"],
        image_max_edge=config["image_max_edge"],
        staged=config["staged"],
        joint=config.get("joint", False),
    )
    try:
        atomic_json(out / "progress.json", {"pending": "video_scan_and_sampling"})
        video_id, windows = prepare(video, out, deadline, config)
        detector = (
            TorchvisionDetector(
                config["detector_weights"],
                config["detector_device"],
                architecture=config["detector_architecture"],
            )
            if config["detector_weights"]
            else None
        )
        face_backend = (
            InsightFaceFeatures(config["face_model_root"], provider=config["face_provider"])
            if config["face_model_root"]
            else None
        )
        face_features = {}
        feature_gaps = []
        if face_backend:
            atomic_json(out / "face_providers.json", face_backend.providers)
        tracker = LocalTracker()
        for index, (shot, media) in enumerate(windows):
            observation_id = f"obs:{index:06d}"
            atomic_json(
                out / "progress.json", {"pending": {"window": index, "total": len(windows)}}
            )
            deadline.require()
            proposals = []
            if detector:
                for m in media:
                    deadline.require()
                    proposals.extend(
                        {**row, "media_id": m.media_id}
                        for row in tracker.update(
                            detector.detect(m.uri), seconds=m.seconds, shot_id=shot
                        )
                    )
            atomic_json(
                out / "tracking" / f"{index:06d}.json",
                {"shot_id": shot, "media": [m.media_id for m in media], "detections": proposals},
            )
            try:
                packet = client.observe(
                    observation_id, video_id, shot, media, deadline=deadline, detections=proposals
                )
                packet = register_packet(packet, proposals)
                memory.append(packet)
                if client.calls and client.calls[-1].get("output_limit_reached"):
                    gaps.append(
                        {"reason": "local_output_limit_reached", "observation": observation_id}
                    )
            except (ValueError, KeyError, TimeoutError, OSError) as exc:
                gaps.append(
                    {
                        "reason": "local_extraction_failed",
                        "observation": observation_id,
                        "media_ids": [m.media_id for m in media],
                        "detail": str(exc),
                    }
                )
            else:
                if face_backend:
                    try:
                        faces = [face for m in media for face in face_backend.extract(m)]
                        assigned, unresolved = attach_face_candidates(packet.instances, faces)
                        face_features.update(assigned)
                        feature_gaps.extend(unresolved)
                        atomic_json(
                            out / "face_features.json",
                            {"features": face_features, "unresolved_owners": feature_gaps},
                        )
                    except (ValueError, KeyError, TimeoutError, OSError) as exc:
                        gaps.append(
                            {
                                "reason": "face_feature_extraction_failed",
                                "observation": observation_id,
                                "detail": str(exc),
                            }
                        )
            atomic_json(out / "calls.json", client.calls)
            export_run(memory, out, gaps=gaps)
        media_index = {m.media_id: m for p in memory.packets for m in p.media}
        candidates = identity_candidates(
            memory,
            limit=config["candidate_limit"],
            face_features=face_features,
            kinds=config["identity_kinds"],
        )
        atomic_json(out / "progress.json", {"pending": "offline_identity_comparisons"})
        for i, (_, left, right, competitors) in enumerate(candidates):
            if i >= config["max_identity_comparisons"]:
                gaps.append(
                    {"reason": "identity_comparison_budget", "remaining_pairs": len(candidates) - i}
                )
                break
            deadline.require()
            # Bounded multi-view references; record all selected endpoints.
            selected = []
            for instance in (left, right, *competitors):
                selected.extend(r.media_id for r in instance.regions[:2])
            media = tuple(media_index[m] for m in dict.fromkeys(selected))
            try:
                proposal = client.compare(
                    f"compare:{i:06d}",
                    left,
                    right,
                    media,
                    deadline=deadline,
                    competitors=competitors,
                )
                memory.compare(proposal)
            except (ValueError, KeyError, TimeoutError, OSError) as exc:
                gaps.append(
                    {
                        "reason": "identity_comparison_failed",
                        "endpoints": [left.instance_id, right.instance_id],
                        "detail": str(exc),
                    }
                )
            atomic_json(out / "calls.json", client.calls)
            export_run(memory, out, gaps=gaps)
    except Exception as exc:
        gaps.append({"reason": "pipeline_interrupted", "detail": str(exc)})
        export_run(memory, out, status="partial", gaps=gaps)
        raise
    export_run(memory, out, status="partial" if gaps else "complete", gaps=gaps)
    return 2 if gaps else 0
