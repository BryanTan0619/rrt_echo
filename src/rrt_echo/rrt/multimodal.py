"""Multimodal RRT: graph-gap scheduling, synchronized observation and binding proposals.

No QA question, caption or answer is used by the writer. The existing visual RRT
supplies local instances; this module adds independently reviewable audio bindings.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ..schema import digest
from ..storage import atomic_json, read_graph
from .audio import attach_audio, file_hash
from .hypergraph import SCHEMA, TemporalEvidenceHypergraph

CONTEXT_RULES = (
    "STORY_CONTEXT is a provisional navigation index, never new evidence. "
    "Observe TARGET first. Audio text describes speech, not a verified event or role. "
    "Do not identify a speaker by a name mentioned in speech, narrative role, or central position. "
    "Only original synchronized audio/video can support active-speaker association. "
    "Allow offscreen, overlapping speech, new people and unknown. Never force the supplied cast."
)


def seconds(media):
    return media["pts"] * media["time_base"][0] / media["time_base"][1]


def story_context(graph, start, end, *, max_facts=6):
    facts = sorted(
        graph.get("facts", []),
        key=lambda f: min(
            (abs(t - (start + end) / 2) for t in f["observed_times"]), default=float("inf")
        ),
    )
    selected = [f for f in facts if any(start - 30 <= t <= end + 30 for t in f["observed_times"])][
        :max_facts
    ]
    return {
        "rules": CONTEXT_RULES,
        "scope": [start, end],
        "observed_candidates": [
            {
                "fact_id": f["fact_id"],
                "predicate": f["predicate"],
                "roles": f["roles"],
                "value": f["value"],
                "observed_times": f["observed_times"],
                "evidence_ids": f["joint_evidence"],
                "status": "stored_observation_not_independent_verification",
            }
            for f in selected
        ],
        "open_bindings": [
            "speaker_to_visible_instance",
            "speaker_vs_mentioned_person",
            "utterance_content_vs_actual_world_state",
        ],
        "cast_is_exhaustive": False,
        "resolved_speech_candidates": [
            {
                "edge_id": e["edge_id"],
                "value": e["value"],
                "interval": e["interval"],
                "resolved_roles": e["resolved_roles"],
                "status": "model_reviewed_not_ground_truth",
            }
            for e in graph.get("multimodal", {}).get("hyperedges", [])
            if e["kind"] == "speech"
            and e["resolved_roles"]
            and e["interval"][0] <= end + 30
            and e["interval"][1] >= start - 30
        ][:4],
    }


def observation_context(results, audio, start, end, coarse=None):
    """Bounded context for the visual observer; provenance cannot become pixel proof."""
    facts = [f for r in results for f in r["observation"]["facts"]]
    # Legacy local facts do not necessarily expose observed_times; use only coarse
    # navigation and cite exact observation IDs, never fabricate their timestamps.
    prior = (
        [
            {
                "observation_id": r["observation"]["observation_id"],
                "predicates": [f["predicate"] for f in r["observation"]["facts"][:4]],
            }
            for r in results[-3:]
        ]
        if facts
        else []
    )
    speech = [
        u for u in (audio or {}).get("utterances", []) if start <= u["start"] and u["end"] <= end
    ][:12]
    episodes = [
        e for e in (coarse or {}).get("episodes", []) if e["start"] < end and start < e["end"]
    ][:3]
    return {
        "rules": CONTEXT_RULES,
        "target": [start, end],
        "prior_observations": prior,
        "episode_candidates": episodes,
        "speech_observations": [
            {
                "utterance_id": u["utterance_id"],
                "text": u["text"],
                "start": u["start"],
                "end": u["end"],
                "speaker_identity": "unresolved",
            }
            for u in speech
        ],
    }


def plan_bindings(graph, max_tasks=12, max_candidates=4):
    if max_tasks < 0 or max_candidates < 1:
        raise ValueError("invalid_multimodal_budget")
    people = [i for i in graph["instances"] if i["kind"] == "person"]
    pending = []
    existing = {
        p["utterance_id"]
        for p in graph.get("multimodal", {}).get("bindings", [])
        if p["status"] == "accepted"
    }
    gaps = []
    for u in graph.get("audio_memory", {}).get("utterances", []):
        if u["utterance_id"] in existing:
            continue
        if u["end"] - u["start"] > 9:
            gaps.append(
                {"utterance_id": u["utterance_id"], "reason": "requires_long_turn_segmentation"}
            )
            continue
        lo, hi = max(0, u["start"] - 1.0), u["end"] + 1.0
        candidates = []
        for i in people:
            regions = [
                r for r in i["regions"] if lo <= seconds(graph["evidence"][r["media_id"]]) <= hi
            ]
            if regions:
                r = min(
                    regions,
                    key=lambda r: abs(seconds(graph["evidence"][r["media_id"]]) - (lo + hi) / 2),
                )
                candidates.append(
                    {
                        "local_instance": i["instance_id"],
                        "description": i["description"],
                        "region": r,
                        "media": graph["evidence"][r["media_id"]],
                    }
                )
        if not candidates:
            gaps.append(
                {"utterance_id": u["utterance_id"], "reason": "no_local_visual_anchor_near_speech"}
            )
            continue
        task = {
            "utterance": copy.deepcopy(u),
            "start": lo,
            "end": hi,
            "candidates": candidates[:max_candidates],
            "candidate_overflow": max(0, len(candidates) - max_candidates),
            "story_context": story_context(graph, lo, hi),
        }
        task["task_id"] = digest(task)[:24]
        pending.append(task)
    # Deterministic time stratification: avoid spending the entire budget on intro speech.
    pending.sort(key=lambda t: t["start"])
    if len(pending) > max_tasks:
        selected = []
        for k in range(max_tasks):
            band = pending[k * len(pending) // max_tasks : (k + 1) * len(pending) // max_tasks]
            # Prefer longer, clearer utterances within each temporal band; no QA text.
            selected.append(
                max(
                    band,
                    key=lambda t: (
                        min(4, t["utterance"]["end"] - t["utterance"]["start"])
                        * (1 - t["utterance"].get("no_speech_prob", 0))
                    ),
                )
            )
    else:
        selected = pending
    selected_ids = {t["task_id"] for t in selected}
    gaps.extend(
        {"utterance_id": t["utterance"]["utterance_id"], "reason": "review_budget_deferred"}
        for t in pending
        if t["task_id"] not in selected_ids
    )
    return {
        "tasks": selected,
        "gaps": gaps,
        "query_blind": True,
        "selection": "time_stratified_clearer_speech_unresolved_bindings",
    }


def submission_from_result(graph, task, evidence, first, second):
    """Raw outputs cannot declare themselves accepted; hypergraph owns admission."""
    local = first.get("local_instance")
    by_id = {c["local_instance"]: c for c in task["candidates"]}
    packet = {
        "schema": SCHEMA,
        "submission_id": "mm:" + task["task_id"],
        "video_id": graph["audio_memory"]["source_video_id"],
        "source_sha256": graph["audio_memory"]["source"]["sha256"],
        "evidence_bundles": {evidence["evidence_id"]: evidence},
        "bindings": [],
        "claims": [],
        "gaps": [],
        "task_id": task["task_id"],
    }
    if local not in by_id:
        packet["gaps"].append(
            {
                "utterance_id": task["utterance"]["utterance_id"],
                "reason": first.get("verdict", "unknown_speaker"),
            }
        )
        return packet
    reviews = []
    for stage, r in [("propose", first), ("verify", second)]:
        if not r:
            continue
        agrees = r.get("local_instance") == local
        reviews.append(
            {
                "stage": stage,
                "model": r["model"],
                "verdict": r.get("verdict", "unresolved") if agrees else "contradicted",
                "local_instance": local,
                "utterance_id": task["utterance"]["utterance_id"],
                "evidence_id": evidence["evidence_id"],
                "used_modalities": ["audio", "video"],
                "observed_speaking": agrees and r.get("observed_speaking") is True,
                "story_used_as_evidence": r.get("story_used_as_evidence", True),
                "reason": r.get("reason", ""),
                "request_id": r["request_id"],
            }
        )
    packet["bindings"].append(
        {
            "proposal_id": packet["submission_id"] + ":speaker",
            "relation": "speaker_of",
            "utterance_id": task["utterance"]["utterance_id"],
            "local_instance": local,
            "visual_media_ids": [by_id[local]["region"]["media_id"]],
            "evidence_ids": [evidence["evidence_id"]],
            "reviews": reviews,
        }
    )
    # Parse claims only as reported speech; do not infer truth, names, or round ownership.
    for n, c in enumerate(first.get("claims", [])[:4]):
        if (
            isinstance(c, str)
            and c.strip()
            and c.strip().casefold() == task["utterance"]["text"].strip().casefold()
        ):
            packet["claims"].append(
                {
                    "claim_id": f"claim:{n}",
                    "kind": "reported_claim",
                    "source_utterance": task["utterance"]["utterance_id"],
                    "text": c,
                    "world_fact": False,
                }
            )
    return packet


def run(graph_path, video, out, *, checkpoint=None, max_tasks=12, dry_run=False):
    graph = read_graph(Path(graph_path))
    if not graph.get("audio_memory"):
        raise ValueError("run_asr_and_attach_audio_first")
    if file_hash(video) != graph["audio_memory"]["source"]["sha256"]:
        raise ValueError("source_video_hash_mismatch")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    plan = plan_bindings(graph, max_tasks=max_tasks)
    atomic_json(out / "plan.json", plan)
    memory = TemporalEvidenceHypergraph(graph)
    calls, failures = [], []
    if not dry_run and plan["tasks"]:
        if not checkpoint:
            raise ValueError("audio_visual_checkpoint_required")
        from .omni import OmniBindingReviewer

        reviewer = OmniBindingReviewer(checkpoint)
        for task in plan["tasks"]:
            task = copy.deepcopy(task)
            task["story_context"] = story_context(graph, task["start"], task["end"])
            task["context_snapshot"] = graph["snapshot_id"]
            try:
                result = reviewer.review(video, task, out / "reviews" / task["task_id"])
                packet = submission_from_result(graph, task, *result)
                memory.submit(packet)
                atomic_json(out / "submissions" / (task["task_id"] + ".json"), packet)
                calls.extend(reviewer.calls)
                reviewer.calls.clear()
                graph = memory.snapshot()
                atomic_json(out / "hypergraph.json", graph)
            except Exception as error:
                failures.append(
                    {"task_id": task["task_id"], "error": type(error).__name__ + ": " + str(error)}
                )
                calls.extend(reviewer.calls)
                reviewer.calls.clear()
            atomic_json(
                out / "progress.json",
                {
                    "completed": len(memory.submissions),
                    "failures": failures,
                    "model_calls": len(calls),
                },
            )
    graph = memory.snapshot()
    atomic_json(out / "hypergraph.json", graph)
    atomic_json(out / "calls.json", calls)
    atomic_json(
        out / "summary.json",
        {
            "tasks": len(plan["tasks"]),
            "failures": failures,
            "bindings": {
                s: sum(p["status"] == s for p in graph["multimodal"]["bindings"])
                for s in ["accepted", "unresolved", "conflict", "revoked"]
            },
            "speech_hyperedges": sum(
                e["kind"] == "speech" for e in graph["multimodal"]["hyperedges"]
            ),
            "model_calls": sum("output_tokens" in c for c in calls),
            "model_attempts": len(calls),
            "dry_run": dry_run,
            "media_verified": False,
        },
    )
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--max-tasks", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--audio", help="Optional ASR artifact to attach before running")
    a = p.parse_args()
    if a.audio:
        graph = attach_audio(read_graph(Path(a.graph)), json.loads(Path(a.audio).read_text()))
        temp = Path(a.out).parent / (Path(a.out).name + ".input.json")
        if temp.exists():
            raise ValueError("input_snapshot_already_exists")
        atomic_json(temp, graph)
        a.graph = str(temp)
    run(a.graph, a.video, a.out, checkpoint=a.checkpoint, max_tasks=a.max_tasks, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
