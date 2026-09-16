"""Coverage-first local observation scheduling and evidence-backed reference reuse.

No QA, cast-size prior, generated boxes or pairwise identity calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from .types import Reference, digest


def evenly(items, count):
    if len(items) <= count:
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def action_clip(clips, start, end, *, context_seconds=8, context_frames=24, label="local"):
    """Preserve every available TARGET sample; cap CONTEXT only, on both sides."""
    if not clips or not start < end or context_seconds < 0 or context_frames < 0:
        raise ValueError("invalid_local_interval")
    source = clips[0]
    if any(c.video_id != source.video_id for c in clips):
        raise ValueError("mixed_videos")
    media = sorted(
        {m.media_id: m for c in clips for m in c.media}.values(), key=lambda m: m.seconds
    )
    target = [m for m in media if start <= m.seconds < end]
    if not target:
        raise ValueError("no_target_samples")
    before = [m for m in media if start - context_seconds <= m.seconds < start]
    after = [m for m in media if end <= m.seconds < end + context_seconds]
    left = min(len(before), context_frames // 2)
    right = min(len(after), context_frames - left)
    left = min(len(before), context_frames - right)
    context = evenly(before, left) + evenly(after, right)
    key = hashlib.sha256(f"{start}:{end}:{label}".encode()).hexdigest()[:12]
    return replace(
        source,
        clip_id=f"{source.video_id}:{label}:{key}",
        target_start=start,
        target_end=end,
        media=tuple(sorted(target + context, key=lambda m: m.seconds)),
        sampling={
            "planner": "continuous-local-v1",
            "target_frames": len(target),
            "available_target_frames": len(target),
            "target_frames_dropped": 0,
            "context_frames": len(context),
            "context_seconds": context_seconds,
            "preprocessing_reused": True,
            "detail_coverage_certified": False,
        },
    )


def plan_local(clips, *, max_seconds=24, max_frames=128, context_seconds=8, context_frames=24):
    """Group adjacent prepared shots, preserving cuts as context rather than events.

    A large source interval is split at actual sample times, never subsampled.
    Boundaries are scheduling boundaries, not semantic event boundaries.
    """
    if not clips or max_seconds <= 0 or max_frames < 1:
        raise ValueError("invalid_plan")
    source = clips[0]
    all_media = sorted(
        {m.media_id: m for c in clips for m in c.media}.values(), key=lambda m: m.seconds
    )
    start, finish = source.start_seconds, source.start_seconds + source.duration
    cuts = sorted({c.target_end for c in clips if start < c.target_end <= finish} | {finish})
    result = []
    while start < finish:
        cap = min(finish, start + max_seconds)
        available = [m for m in all_media if start <= m.seconds < cap]
        if len(available) > max_frames:
            cap = available[max_frames].seconds
        choices = [t for t in cuts if start < t <= cap]
        end = max(choices) if choices else cap
        # Empty intervals still have a scheduling record, never fabricated evidence.
        if any(start <= m.seconds < end for m in all_media):
            result.append(
                action_clip(
                    clips,
                    start,
                    end,
                    context_seconds=context_seconds,
                    context_frames=context_frames,
                    label="action",
                )
            )
        start = end
    return result


def coverage_order(clips, phases=4):
    """Round-robin temporal quarters: an exhausted budget does not only cover the start."""
    if phases < 1:
        raise ValueError("invalid_phases")
    if not clips:
        return []
    start = clips[0].start_seconds
    duration = clips[0].duration
    buckets = [[] for _ in range(phases)]
    for c in sorted(clips, key=lambda c: c.target_start):
        b = min(phases - 1, max(0, int((c.target_start - start) / duration * phases)))
        buckets[b].append(c)
    return [b[i] for i in range(max(map(len, buckets))) for b in buckets if i < len(b)]


def coverage_report(planned, results):
    completed = {
        r.get("input_clip_id", r["observation"]["observation_id"].removesuffix(":verified-v7")): r
        for r in results
    }
    rows = []
    for c in sorted(planned, key=lambda c: c.target_start):
        r = completed.get(c.clip_id)
        facts = r["observation"]["facts"] if r else []
        cited = {mid for f in facts for mid in f["joint_evidence"]}
        rows.append(
            {
                "clip_id": c.clip_id,
                "target": [c.target_start, c.target_end],
                "available_frames": len(c.target_ids),
                "processed": r is not None,
                "facts": len(facts),
                "fact_cited_frames": len(cited),
                "gaps": r.get("coverage_gaps", []) if r else ["not_processed"],
                "semantic_coverage_certified": False,
            }
        )
    return {
        "intervals": rows,
        "planned": len(rows),
        "completed": sum(row["processed"] for row in rows),
        "sampling_is_not_semantic_coverage": True,
    }


def reference_profile(observation, instance):
    """Owned state/attribute evidence, never identity proof or inferred persistence.

    Keep old free descriptions available to auditors, but do not turn them or OCR
    rows into structured appearance facts. Body-region ownership stays separate.
    """
    media = {m["media_id"]: m for m in observation["media"]}
    anchors = {a["media_id"] for a in instance["regions"]}
    cues = []
    for fact in observation["facts"]:
        evidence = fact.get("joint_evidence", [])
        owner_evidence = fact.get("evidence_by_slot", {}).get("owner", [])
        if (
            fact["kind"] not in {"state", "attribute"}
            or fact["roles"].get("owner") != instance["instance_id"]
            or not evidence
            or not set(evidence) <= media.keys()
            or not set(owner_evidence) <= set(evidence)
            or not set(owner_evidence) & anchors
        ):
            continue
        cues.append(
            {
                "fact_id": fact["fact_id"],
                "property": fact["predicate"],
                "value": fact["value"],
                "evidence_ids": evidence,
                "observed_times": [
                    media[x]["pts"] * media[x]["time_base"][0] / media[x]["time_base"][1]
                    for x in evidence
                ],
                "status": "model_observed; not_semantically_certified",
            }
        )
    return {
        "instance_id": instance["instance_id"],
        "description_for_audit": instance["description"],
        "cues": cues,
        "identity_proof": False,
        "persistence_inferred": False,
        "gap": None
        if cues
        else "no_owned_state_attribute_cues; do_not_promote_description_or_text",
    }


def diverse_anchors(anchors, profile, media, count):
    """Prefer views covering different recorded cues, then temporal separation.

    No synthetic body crops, face embeddings or visual distinctiveness claim.
    """
    candidates = sorted(anchors, key=lambda a: (media[a["media_id"]].seconds, str(a["box"])))
    selected, covered = [], set()
    while candidates and len(selected) < count:

        def score(a):
            cues = {c["fact_id"] for c in profile["cues"] if a["media_id"] in c["evidence_ids"]}
            separation = min(
                (
                    abs(media[a["media_id"]].seconds - media[x["media_id"]].seconds)
                    for x in selected
                ),
                default=0,
            )
            return (len(cues - covered), separation)

        chosen = max(candidates, key=score)
        selected.append(chosen)
        candidates.remove(chosen)
        covered.update(
            c["fact_id"] for c in profile["cues"] if chosen["media_id"] in c["evidence_ids"]
        )
    return selected


def local_references(results, clip, media, folder, *, max_instances=3, max_views=6):
    """Reuse localized observation endpoints. Time ranks candidates, never proves identity.

    References retain the original instance and source geometry. Neutral IDs only.
    Offline callers may provide results from either temporal direction.
    """
    candidates = []
    for result in results:
        obs = result["observation"]
        if obs["video_id"] != clip.video_id:
            raise ValueError("reference_video_mismatch")
        for instance in obs["instances"]:
            if instance["kind"] not in {"person", "object"}:
                continue
            anchors = [
                a
                for a in instance["regions"]
                if a["media_id"] in media and a["media_id"] not in clip.target_ids
            ]
            if not anchors:
                continue
            distance = min(
                abs(media[a["media_id"]].seconds - (clip.target_start + clip.target_end) / 2)
                for a in anchors
            )
            candidates.append((distance, instance, anchors, reference_profile(obs, instance)))
    candidates.sort(key=lambda x: (x[0], x[1]["instance_id"]))
    chosen = candidates[: min(max_instances, max_views)]
    refs = []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for _, instance, anchors, profile in chosen:
        # Two temporal views, not a repeated crop; no face-only requirement.
        anchors = sorted(anchors, key=lambda a: media[a["media_id"]].seconds)
        selected = diverse_anchors(
            anchors, profile, media, max(1, max_views // max(1, len(chosen)))
        )
        views = []
        sources = []
        for anchor in selected:
            m = media[anchor["media_id"]]
            if digest(m.uri) != m.sha256:
                raise ValueError("reference_source_changed")
            import json

            key = hashlib.sha256(
                json.dumps([instance["instance_id"], m.sha256, anchor["box"]]).encode()
            ).hexdigest()
            p = folder / (key + ".jpg")
            if not p.exists():
                with Image.open(m.uri) as src:
                    img = src.convert("RGB")
                    a, b, c, d = anchor["box"]
                    ImageDraw.Draw(img).rectangle(
                        (a * img.width, b * img.height, c * img.width, d * img.height),
                        outline="yellow",
                        width=3,
                    )
                    img.save(p, quality=92)
            views.append(
                replace(m, media_id="reference:" + key, uri=str(p.resolve()), sha256=digest(p))
            )
            sources.append(
                {
                    "instance_id": instance["instance_id"],
                    "source_media_id": m.media_id,
                    "source_sha256": m.sha256,
                    "box": anchor["box"],
                    "origin": "local_observation",
                    "candidate_id": None,
                    "semantic_support_audited": False,
                    "appearance_cues": [
                        c for c in profile["cues"] if m.media_id in c["evidence_ids"]
                    ],
                    "continuity_not_inferred": True,
                }
            )
        refs.append(
            Reference(
                "ref:" + hashlib.sha256(instance["instance_id"].encode()).hexdigest()[:16],
                instance["instance_id"],
                instance["kind"],
                tuple(views),
                tuple(sources),
            )
        )
    return tuple(refs), {
        "candidate_total": len(candidates),
        "selected": len(refs),
        "omitted_instance_ids": [x[1]["instance_id"] for x in candidates[len(chosen) :]],
        "selection": "temporal_candidate_rank; cue_diverse_local_views; hypothesis_only",
        "profiles": [x[3] for x in chosen],
    }
