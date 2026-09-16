"""Query-blind A/B/C/D orchestration. Coarse-index failure cannot block local evidence."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from ..echo_perception.global_index import GlobalIndexClient
from ..echo_perception.local_stage import (
    action_clip,
    coverage_order,
    coverage_report,
    local_references,
    plan_local,
)
from ..echo_perception.types import Deadline, atomic_json, clip_from_dict, digest
from ..echo_perception.video import prepare
from .anchor_review import enrich_tiny_anchors
from .association import motion_candidates, resolve
from .diagnostics import audit_observations, observation_clip_id, repair_plan
from .link_review import review_links
from .memory import RRTMemory, narrative_projection
from .observation import OccurrenceVisionClient
from .recover_links import recover_links
from .verified import VerifiedPerceptionClient


def request_inventory(out):
    """Count structured request artifacts, not words in free-form logs."""
    rows = []
    for path in sorted(Path(out).rglob("request.json")):
        directory = path.parent
        request = json.loads(path.read_text())

        def read(name):
            target = directory / name
            return json.loads(target.read_text()) if target.exists() else {}

        timing, response, result = read("timing.json"), read("response.json"), read("result.json")
        choices = response.get("choices", [])
        usage = response.get("usage") or timing.get("usage") or {}
        rows.append(
            {
                "stage": path.relative_to(out).parts[0],
                "request": str(path.relative_to(out)),
                "target": request.get("clip_id", request.get("pairs")),
                "visual_inputs": len(request.get("media", request.get("presentations", []))),
                "asr_segments": len(
                    (request.get("multimodal_context") or {}).get("speech_observations", [])
                ),
                "native_audio": False,
                "output_tokens": usage.get("completion_tokens"),
                "input_tokens": usage.get("prompt_tokens"),
                "elapsed_seconds": timing.get("elapsed_seconds", timing.get("seconds")),
                "truncated": any(c.get("finish_reason") == "length" for c in choices),
                "failed": (directory / "failure.json").exists(),
                "facts": len(result.get("observation", {}).get("facts", []))
                if isinstance(result, dict)
                else 0,
                "correspondences": len(result)
                if isinstance(result, list)
                else len(result.get("link_proposals", [])),
            }
        )
    return {
        "requests": rows,
        "request_count": len(rows),
        "request_seconds_sum": sum(r["elapsed_seconds"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "note": "Request durations can overlap; use report.elapsed_seconds for wall clock. Native-audio optional legacy calls are not counted here.",
    }


def run(
    *,
    video=None,
    clips_path=None,
    out,
    model,
    base_url,
    budget=3600,
    max_local_calls=100,
    max_local_repairs=4,
    max_identity_batches=40,
    local_seconds=8,
    request_seconds=120,
    max_tokens=3072,
    index=True,
    identity=True,
    video_transport="rrt-v1",
    mode="video",
    cases=None,
    source_video_id=None,
    verified_perception=False,
    binding_review=True,
    repair_tiny_anchors=False,
    association_workers=2,
    audio_observations=None,
    av_checkpoint=None,
    max_av_tasks=12,
    profile="simple",
    processor_max_pixels=None,
    sample_fps=4.0,
):
    if profile not in {"simple", "legacy"}:
        raise ValueError("unknown_pipeline_profile")
    if profile == "simple":
        if verified_perception or repair_tiny_anchors or av_checkpoint:
            raise ValueError("optional_multistage_modules_require_legacy_profile")
        index, binding_review, max_local_repairs = False, False, 0
    processor_max_pixels = processor_max_pixels or (401408 if profile == "simple" else 100352)
    if (video is None) == (clips_path is None):
        raise ValueError("provide_video_or_clips")
    if av_checkpoint and (not audio_observations or not video):
        raise ValueError("av_review_requires_source_video_and_asr_artifact")
    if max_av_tasks < 0:
        raise ValueError("negative_av_budget")
    if max_local_repairs < 0:
        raise ValueError("negative_repair_budget")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = Deadline(budget - 30)
    config = {
        "profile": profile,
        "processor_max_pixels": processor_max_pixels,
        "sample_fps": sample_fps,
        "offline_reserve_seconds": max(120, budget * 0.2) if profile == "simple" else 60,
        "local_output_limits": {"instances": 8, "events": 4, "properties": 4, "links": 4}
        if profile == "simple"
        else None,
        "video": str(video) if video else None,
        "clips_path": str(clips_path) if clips_path else None,
        "model": model,
        "base_url": base_url,
        "budget": budget,
        "max_tokens": max_tokens,
        "local_seconds": local_seconds,
        "request_seconds": request_seconds,
        "index": index,
        "identity": identity,
        "max_local_calls": max_local_calls,
        "max_local_repairs": max_local_repairs,
        "max_identity_batches": max_identity_batches,
        "mode": mode,
        "video_transport": video_transport,
        "verified_perception": verified_perception,
        "binding_review": binding_review,
        "repair_tiny_anchors": repair_tiny_anchors,
        "association_workers": association_workers,
        "av_checkpoint": str(av_checkpoint) if av_checkpoint else None,
        "max_av_tasks": max_av_tasks,
        "audio_observations": str(audio_observations) if audio_observations else None,
        "query_blind": True,
        "caption_access": False,
        "qa_access": False,
        "preprocessing_included": video is not None,
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[1])): digest(p)
            for p in Path(__file__).parents[1].rglob("*.py")
        },
    }
    atomic_json(out / "manifest.json", config)
    results = []
    failures = []
    coarse = None
    audio = json.loads(Path(audio_observations).read_text()) if audio_observations else None
    if audio:
        from .audio import validate_audio

        validate_audio(audio)
        if video:
            from .audio import file_hash

            if file_hash(video) != audio["source"]["sha256"]:
                raise ValueError("audio_video_hash_mismatch")
    ledger = (
        RRTMemory(
            source_video_id or (Path(video).stem if video else None), ledger_dir=out / "ledger"
        )
        if profile == "simple"
        else None
    )
    if ledger and audio is not None:
        ledger.set_audio(audio)
    association = []
    planned = []
    try:
        if video:
            clips = prepare(Path(video), out / "prepared", deadline=deadline, sample_fps=sample_fps)
        else:
            clips = [clip_from_dict(c) for c in json.loads(Path(clips_path).read_text())]
        atomic_json(out / "clips.json", [asdict(c) for c in clips])
        media = {m.media_id: m for c in clips for m in c.media}
        shots = {mid: c.shot_id for c in clips for mid in c.target_ids}
        planned = plan_local(
            clips, max_seconds=local_seconds, max_frames=40, context_seconds=4, context_frames=8
        )
        if cases:
            planned = [
                action_clip(
                    clips,
                    x["start"],
                    x["end"],
                    context_seconds=x.get("context", 4),
                    context_frames=8,
                    label=x["name"],
                )
                for x in cases
            ]
        atomic_json(out / "coverage_plan.json", coverage_report(planned, []))
        if index and deadline.remaining() > request_seconds + 120:
            atomic_json(out / "progress.json", {"stage": "coarse_index"})
            try:
                coarse = GlobalIndexClient(
                    model=model,
                    base_url=base_url,
                    max_frames=64,
                    max_candidates=8,
                    max_tokens=2048,
                    request_timeout=min(120, request_seconds),
                    mode=mode,
                    video_transport=video_transport,
                    task="compact_story",
                    schema_mode="json_schema",
                    generation={"temperature": 0, "repetition_penalty": 1.05},
                ).build(clips, out=out / "index_calls", deadline=deadline)
                atomic_json(out / "global_index.json", coarse)
            except Exception as error:
                failures.append({"stage": "coarse_index", "error": str(error)[:500]})
        ordered = coverage_order(planned)
        # Index episodes guide reference candidate selection, never constrain the local cast.
        for i, clip in enumerate(ordered[:max_local_calls]):
            if deadline.remaining() < request_seconds + config["offline_reserve_seconds"]:
                break
            atomic_json(
                out / "progress.json",
                {
                    "stage": "local",
                    "completed": i,
                    "total": len(ordered),
                    "target": [clip.target_start, clip.target_end],
                },
            )
            try:
                history = results
                if coarse:
                    episode_spans = [
                        (e["start"], e["end"])
                        for e in coarse.get("episodes", [])
                        if e["start"] < clip.target_end and clip.target_start < e["end"]
                    ]
                    if episode_spans:
                        preferred = [
                            r
                            for r in results
                            if any(
                                lo <= m["pts"] * m["time_base"][0] / m["time_base"][1] <= hi
                                for m in r["observation"]["media"]
                                for lo, hi in episode_spans
                            )
                        ]
                        history = preferred or results
                refs, _ = (
                    local_references(
                        history, clip, media, out / "references", max_instances=2, max_views=4
                    )
                    if history and not verified_perception
                    else ((), {})
                )
                from .multimodal import observation_context

                mm_context = (
                    observation_context(
                        [] if ledger else results, audio, clip.target_start, clip.target_end, coarse
                    )
                    if audio
                    else None
                )
                if verified_perception:
                    client = VerifiedPerceptionClient(
                        model=model,
                        base_url=base_url,
                        max_tokens=max_tokens,
                        request_timeout=request_seconds,
                    )
                else:
                    client = OccurrenceVisionClient(
                        model=model,
                        base_url=base_url,
                        mode=mode,
                        video_transport=video_transport,
                        max_tokens=max_tokens,
                        request_timeout=request_seconds,
                        **(
                            {
                                "max_instances": 8,
                                "max_events": 4,
                                "max_properties": 4,
                                "max_links": 4,
                                "typed_ownership": True,
                                "processor_max_pixels": processor_max_pixels,
                            }
                            if ledger
                            else {}
                        ),
                    )
                client.multimodal_context = mm_context
                r = client.observe(
                    clip, out=out / "local_calls", deadline=deadline, references=refs
                )
                r["input_clip_id"] = clip.clip_id
                if ledger:
                    ledger.append(r)
                results.append(r)
            except Exception as error:
                failures.append(
                    {
                        "stage": "local",
                        "clip_id": clip.clip_id,
                        "span": [clip.target_start, clip.target_end],
                        "error": str(error)[:500],
                    }
                )
            atomic_json(out / "observations.json", results)
            atomic_json(out / "failures.json", failures)
            atomic_json(out / "coverage.json", coverage_report(planned, results))
        # Query-blind recovery before identity comparison. Retry each selected
        # window once in an isolated cache; original observations stay on disk.
        repairs = repair_plan(planned, results, limit=min(max_local_repairs, max_local_calls))
        atomic_json(out / "binding_loss_audit.json", audit_observations(results))
        atomic_json(out / "repair_plan.json", repairs)
        atomic_json(out / "observations.before_repair.json", results)
        by_clip = {c.clip_id: c for c in planned}
        for item in repairs:
            if deadline.remaining() < request_seconds + 120:
                break
            try:
                repair_client = VerifiedPerceptionClient(
                    model=model,
                    base_url=base_url,
                    max_tokens=max_tokens,
                    request_timeout=request_seconds,
                )
                repaired = repair_client.observe(
                    by_clip[item["clip_id"]], out=out / "repair_calls", deadline=deadline
                )
                repaired["input_clip_id"] = item["clip_id"]
                old = next((r for r in results if observation_clip_id(r) == item["clip_id"]), None)
                old_issues = len(audit_observations([old])) if old else 0
                new_issues = len(audit_observations([repaired]))
                improved = (
                    old is None
                    or (not old["observation"]["facts"])
                    or (
                        new_issues < old_issues
                        and len(repaired["observation"]["facts"])
                        >= len(old["observation"]["facts"])
                    )
                )
                if repaired["observation"]["facts"] and improved:
                    results = [r for r in results if observation_clip_id(r) != item["clip_id"]] + [
                        repaired
                    ]
            except Exception as error:
                failures.append(
                    {"stage": "local_repair", "clip_id": item["clip_id"], "error": str(error)[:500]}
                )
        atomic_json(out / "observations.json", results)
        if repair_tiny_anchors and results and deadline.remaining() > 60:
            atomic_json(out / "observations.original.json", results)
            results = enrich_tiny_anchors(
                results,
                out=out / "anchor_review",
                model=model,
                base_url=base_url,
                deadline=Deadline(max(1, deadline.remaining() * 0.2)),
            )
            atomic_json(out / "observations.json", results)
        motion = motion_candidates(results, shots)
        atomic_json(out / "motion.json", motion)
        if identity and results and deadline.remaining() > 60:
            atomic_json(
                out / "progress.json", {"stage": "offline_identity", "observations": len(results)}
            )
            association = resolve(
                results,
                model=model,
                base_url=base_url,
                out=out / "association",
                deadline=Deadline(max(1, deadline.remaining() * 0.55))
                if binding_review
                else deadline,
                motion=motion,
                max_batches=max_identity_batches,
                workers=association_workers,
                skip_observed_pairs=profile == "simple",
            )
            if binding_review:
                # Salvage independent rows from a partly invalid batch, then
                # verify actual endpoints before any correspondence is committed.
                recovered = recover_links(
                    results, out / "association", out / "association_recovery"
                )
                local = [p for r in results for p in r.get("link_proposals", [])]
                candidates = list(
                    {p["proposal_id"]: p for p in [*association, *recovered, *local]}.values()
                )
                association = review_links(
                    results,
                    candidates,
                    out=out / "binding_review",
                    model=model,
                    base_url=base_url,
                    workers=association_workers,
                    deadline=deadline,
                )
        if results:
            mem = ledger or RRTMemory(source_video_id or (Path(video).stem if video else None))
            for r in [] if ledger else results:
                # Original local proposals remain in observations.json; the
                # active journal only receives the reviewed correspondence set.
                mem.append({**r, "link_proposals": []} if binding_review else r)
            for p in association:
                mem.propose(p)
            graph = mem.materialize(out) if ledger else mem.snapshot()
            if audio_observations and not ledger:
                from .audio import attach_audio
                from .hypergraph import TemporalEvidenceHypergraph

                graph = TemporalEvidenceHypergraph(attach_audio(graph, audio)).snapshot()
            from .hypergraph import TemporalEvidenceHypergraph

            graph = TemporalEvidenceHypergraph(graph).snapshot()
            atomic_json(out / "hypergraph.json", graph)
            if av_checkpoint:
                from .multimodal import run as run_multimodal

                graph = run_multimodal(
                    out / "hypergraph.json",
                    video,
                    out / "multimodal_rrt",
                    checkpoint=av_checkpoint,
                    max_tasks=max_av_tasks,
                )
                atomic_json(out / "hypergraph.json", graph)
            atomic_json(out / "storyline.json", narrative_projection(graph))
            atomic_json(out / "binding_journal.json", mem.journal)
    except Exception as error:
        failures.append({"stage": "pipeline", "error": str(error)[:1000]})
        raise
    finally:
        if ledger and ledger.observations:
            ledger.materialize(out)
        coverage = coverage_report(planned, results)
        atomic_json(out / "coverage.json", coverage)
        if (out / "association" / "report.json").exists():
            association_report = json.loads((out / "association" / "report.json").read_text())
            failures.extend(
                {"stage": "offline_identity", **f} for f in association_report["failures"]
            )
        atomic_json(
            out / "report.json",
            {
                "architecture": "echo-simple-v1" if ledger else "rrt-abcd-v1",
                "elapsed_seconds": time.monotonic() - started,
                "coverage": coverage,
                "observations": len(results),
                "identity_proposals": len(association),
                "failures": failures,
                "semantic_quality_passed": None,
                "qa_run": False,
            },
        )
        atomic_json(out / "failures.json", failures)
        atomic_json(out / "request_inventory.json", request_inventory(out))
        atomic_json(out / "DONE.json", {"execution_finished": True, "semantic_acceptance": None})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--clips")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", choices=["simple", "legacy"], default="simple")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--budget", type=int, default=3600)
    p.add_argument("--max-local-calls", type=int, default=100)
    p.add_argument("--max-local-repairs", type=int, default=4)
    p.add_argument("--max-identity-batches", type=int, default=40)
    p.add_argument("--local-seconds", type=float, default=8)
    p.add_argument("--timeout", type=float, default=80)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--processor-max-pixels", type=int)
    p.add_argument("--sample-fps", type=float, default=4.0)
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--no-identity", action="store_true")
    p.add_argument(
        "--no-binding-review",
        action="store_true",
        help="Diagnostic legacy behavior: bypass focused endpoint review",
    )
    p.add_argument("--repair-tiny-anchors", action="store_true")
    p.add_argument("--association-workers", type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument(
        "--av-checkpoint", help="Local M3-Agent/Qwen2.5-Omni checkpoint for actual AV review"
    )
    p.add_argument("--max-av-tasks", type=int, default=12)
    p.add_argument("--audio-observations", help="Timestamped ASR artifact from rrt_echo.rrt.audio")
    p.add_argument("--cases")
    p.add_argument("--source-video-id")
    p.add_argument(
        "--verified-perception",
        action="store_true",
        help="Context-assisted target extraction, single-frame grounding and box audit",
    )
    p.add_argument("--mode", choices=["images", "video"], default="video")
    p.add_argument("--video-transport", choices=["rrt-v1", "standard"], default="rrt-v1")
    a = p.parse_args()
    run(
        profile=a.profile,
        video=a.video,
        clips_path=a.clips,
        out=a.out,
        model=a.model,
        base_url=a.base_url,
        budget=a.budget,
        max_local_calls=a.max_local_calls,
        max_local_repairs=a.max_local_repairs,
        max_identity_batches=a.max_identity_batches,
        local_seconds=a.local_seconds,
        request_seconds=a.timeout,
        max_tokens=a.max_tokens,
        processor_max_pixels=a.processor_max_pixels,
        sample_fps=a.sample_fps,
        index=not a.no_index,
        identity=not a.no_identity,
        mode=a.mode,
        video_transport=a.video_transport,
        cases=json.loads(Path(a.cases).read_text()) if a.cases else None,
        source_video_id=a.source_video_id,
        verified_perception=a.verified_perception,
        binding_review=not a.no_binding_review,
        repair_tiny_anchors=a.repair_tiny_anchors,
        association_workers=a.association_workers,
        audio_observations=a.audio_observations,
        av_checkpoint=a.av_checkpoint,
        max_av_tasks=a.max_av_tasks,
    )


if __name__ == "__main__":
    main()
