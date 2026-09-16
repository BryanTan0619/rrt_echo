"""Rebuild identity from immutable observations using explicitly scoped crop pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .identity import IdentityGraph
from .memory import Memory
from .perception.features import cosine
from .perception.identity import compare_scoped, prepare_crop
from .perception.vlm import VisionClient
from .runtime import Deadline, supervise
from .schema import ObservationPacket
from .storage import atomic_json, export_run


def rank_candidates(current, prior, memory, features, limit=3, *, seen_pairs=()):
    group_of = {
        key: min(group) for group in memory.identity.components(memory.instances) for key in group
    }
    media = {m.media_id: m for p in memory.packets for m in p.media}
    shots = {i.instance_id: p.shot_id for p in memory.packets for i in p.instances}
    words = set(current.description.casefold().split())
    ranked = {}
    for ref in prior:
        other = memory.instances[ref]
        if (
            tuple(sorted((ref, current.instance_id))) in seen_pairs
            or other.kind != current.kind
            or not other.regions
            or group_of[ref] == group_of[current.instance_id]
        ):
            continue
        other_words = set(other.description.casefold().split())
        similarity = max(
            (
                cosine(a, b)
                for a in features.get(current.instance_id, [])
                for b in features.get(ref, [])
            ),
            default=0,
        )
        # Temporal proximity is a retrieval cue, never proof of identity.
        gap = min(
            abs(media[a.media_id].seconds - media[b.media_id].seconds)
            for a in current.regions
            for b in other.regions
        )
        temporal = 2 / (1 + gap) if shots[ref] == shots[current.instance_id] else 0
        score = (
            temporal
            + 8 * similarity
            + 4 * len(words & other_words) / max(1, len(words | other_words))
            + 4 * bool(current.track_ref and current.track_ref == other.track_ref)
        )
        group = group_of[ref]
        if group not in ranked or score > ranked[group][0]:
            ranked[group] = (score, ref)
    return [ref for _, ref in sorted(ranked.values(), key=lambda x: (-x[0], x[1]))[:limit]]


def run(source, out, *, model, base_url, seconds, max_calls, resume=False):
    started = time.monotonic()
    deadline = Deadline(seconds)
    original = json.loads((source / "hypergraph.json").read_text())
    memory = Memory(source_video_id=original["source_video_id"])
    for row in json.loads((source / "evidence/observations.json").read_text()):
        memory.append(ObservationPacket.from_dict(row))
    if resume:
        memory.identity = IdentityGraph.restore(original["identity_decisions"], memory.instances)
    media_index = {m.media_id: m for p in memory.packets for m in p.media}
    f = source / "face_features.json"
    if resume and not f.exists():
        parent = Path(json.loads((source / "manifest.json").read_text())["source"])
        f = parent / "face_features.json"
    features = json.loads(f.read_text())["features"] if f.exists() else {}
    people = sorted(
        (i for i in memory.instances.values() if i.kind == "person" and i.regions),
        key=lambda i: (min(media_index[r.media_id].seconds for r in i.regions), i.instance_id),
    )
    client = VisionClient(model=model, base_url=base_url, max_tokens=2048)
    candidate_log = []
    crops = {}
    prior = []
    gaps = []
    calls = 0
    processed = int(json.loads((source / "runtime.json").read_text())["processed"]) if resume else 0
    if resume and processed:
        # The last current may have exhausted the call budget between candidates.
        processed -= 1
    seen_pairs = (
        {tuple(sorted((d.proposal.left, d.proposal.right))) for d in memory.identity.decisions}
        if resume
        else set()
    )
    if not 0 <= processed <= len(people):
        raise ValueError("invalid resume frontier")
    prior = [i.instance_id for i in people[:processed]]
    if resume and (source / "crop_sources.json").exists():
        crops = json.loads((source / "crop_sources.json").read_text())
    proposal_prefix = "resume:" + original["snapshot_id"][:16] if resume else "scoped"
    (out / "crops").mkdir()
    from .perception.video import file_hash

    atomic_json(
        out / "manifest.json",
        {
            "mode": "observation_reuse_scoped_identity",
            "resume": resume,
            "previously_processed": processed,
            "source": str(source),
            "source_snapshot_id": original["snapshot_id"],
            "source_manifest": json.loads((source / "manifest.json").read_text()),
            "code_sha256": {
                str(p.relative_to(Path(__file__).parent)): file_hash(p)
                for p in sorted(Path(__file__).parent.rglob("*.py"))
            },
            "seconds": seconds,
            "max_calls": max_calls,
            "query_blind": True,
            "candidate_scope": "full_video",
            "temporal_proximity_is_identity_evidence": False,
            "identity_policy": "validated journal replay"
            if resume
            else "fresh interpretation; original journal remains in source",
        },
    )
    export_run(memory, out)
    try:
        for current in people[processed:]:
            deadline.require()
            if calls >= max_calls:
                gaps.append(
                    {"reason": "identity_call_budget", "remaining_people": len(people) - processed}
                )
                break
            if resume and any(
                d.proposal.verdict == "same"
                and current.instance_id in (d.proposal.left, d.proposal.right)
                for d in memory.identity.active()
            ):
                prior.append(current.instance_id)
                processed += 1
                continue
            # Offline build: later observations are valid candidate references.
            refs = rank_candidates(
                current,
                [i.instance_id for i in people],
                memory,
                features,
                seen_pairs=seen_pairs,
            )
            candidate_log.append(
                {
                    "current": current.instance_id,
                    "candidates": refs,
                    "scope": "full_video",
                    "ranking_only": True,
                }
            )
            atomic_json(out / "candidate_trace.json", candidate_log)
            atomic_json(
                out / "progress.json",
                {
                    "people_processed": processed,
                    "total_people": len(people),
                    "calls_completed": calls,
                    "current": current.instance_id,
                },
            )
            for ref in refs:
                deadline.require()
                if calls >= max_calls:
                    break
                for key in (ref, current.instance_id):
                    if key not in crops:
                        crops[key] = prepare_crop(memory.instances[key], media_index, out / "crops")
                try:
                    proposal = compare_scoped(
                        client,
                        f"{proposal_prefix}:{calls}",
                        ref,
                        current.instance_id,
                        crops,
                        deadline,
                    )
                    decision = memory.compare(proposal)
                    accepted = proposal.verdict == "same" and decision.status == "accepted"
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    gaps.append(
                        {
                            "reason": "identity_comparison_failed",
                            "endpoints": [ref, current.instance_id],
                            "detail": str(exc),
                        }
                    )
                    accepted = False
                seen_pairs.add(tuple(sorted((ref, current.instance_id))))
                calls += 1
                atomic_json(out / "calls.json", client.calls)
                if accepted:
                    break
            prior.append(current.instance_id)
            processed += 1
            export_run(memory, out, gaps=gaps)
    except TimeoutError:
        gaps.append({"reason": "identity_time_budget", "remaining_people": len(people) - processed})
    atomic_json(out / "crop_sources.json", crops)
    inherited = json.loads((source / "status.json").read_text())
    gaps.extend(
        {"reason": "upstream_perception_gap", "source_gap": gap}
        for gap in inherited.get("gaps", [])
        if not (resume and gap.get("reason") in {"identity_call_budget", "identity_time_budget"})
    )
    export_run(memory, out, status="partial" if gaps else "complete", gaps=gaps)
    atomic_json(
        out / "runtime.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "calls": calls,
            "people": len(people),
            "processed": processed,
            "inherited_perception_status": inherited,
            "quality_passed": None,
        },
    )
    return 2 if gaps else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = parser.parse_args()
    if not 10 <= a.seconds <= 3600 or a.max_calls < 1:
        parser.error("invalid identity budget")
    if not a.worker:
        a.out.mkdir(parents=True, exist_ok=False)
        return supervise(
            [sys.executable, "-m", "rrt_echo.reconcile", *sys.argv[1:], "--worker"],
            a.out,
            seconds=a.seconds,
        )
    return run(
        a.source,
        a.out,
        model=a.model,
        base_url=a.base_url,
        seconds=a.seconds,
        max_calls=a.max_calls,
        resume=a.resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
