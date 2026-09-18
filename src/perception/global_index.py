"""Query-blind full-video candidate index, followed by visual reference selection.

The coarse pass proposes a cast and event index. Neither its labels nor grouping
of anchor images grants identity support. Each reference has its own endpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw

from .types import Deadline, Reference, atomic_json, digest
from .vlm import jpeg, transport_schema
from .wire import arr, enum, obj, string

INDEX_PROMPT = """Watch the chronological video sample as a whole. Build a compact search index
for a later detailed pass, not a frame-by-frame caption.
1. Candidates: recurring people and important objects. Use neutral IDs and short
visual labels. Select identifiable regions across separated appearances and
appearance changes, not several adjacent duplicate views. A body part, group or
graphic is not a new person. Keep competing identities when recurrence is unclear.
2. Episodes: major action phases with participants. Refer to start_frame and
end_frame in the supplied timeline, never invent seconds. Group related shots;
do not enumerate blurry frames or facial expressions as separate story events.
3. Semantic labels: propose names or relational roles only with cited video
evidence. State the related candidate and observable basis. Distinguish directly
observed labels from contextual inference. Appearance or carrying someone alone
does not establish kinship. A narrative role never proves visual identity.
Use [left, top, right, bottom] integer boxes in [0,1000] on the full frame.
All entries are hypotheses, not a fixed cast or committed facts. Cite supplied
frame IDs only. Record sampling gaps. Keep descriptions brief; do not fill quotas.
"""


def index_schema(frames, max_candidates=32, max_episodes=24):
    box = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    anchor = obj({"media_id": enum(frames), "box": box})
    candidate_ids = [f"C{i}" for i in range(max_candidates)]
    candidate = obj(
        {
            "candidate_id": enum(candidate_ids),
            "kind": enum(["person", "object"]),
            "label": string(80),
            "description": string(240),
            "anchors": arr(anchor, 6, 1),
            "uncertainty": {"type": "string", "maxLength": 160},
        }
    )
    episode = obj(
        {
            "start_frame": enum(frames),
            "end_frame": enum(frames),
            "summary": string(240),
            "candidate_ids": arr(enum(candidate_ids), max_candidates),
            "evidence_ids": arr(enum(frames), 8, 1),
        }
    )
    return obj(
        {
            "candidates": arr(candidate, max_candidates),
            "episodes": arr(episode, max_episodes),
            "semantic_labels": arr(
                obj(
                    {
                        "candidate_id": enum(candidate_ids),
                        "label": string(80),
                        "relation": string(80),
                        "related_candidate_ids": arr(enum(candidate_ids), 4),
                        "evidence_ids": arr(enum(frames), 8, 1),
                        "basis": string(240),
                        "epistemic_status": enum(["observed", "inferred", "unresolved"]),
                    }
                ),
                24,
            ),
            "gaps": arr(string(160), 32),
        }
    )


def even_sample(items, count):
    if count < 1:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def storyboard(clips, maximum):
    if not clips or maximum < 2:
        raise ValueError("need_clips_and_at_least_two_overview_frames")
    if len({c.video_id for c in clips}) != 1:
        raise ValueError("mixed_video_index")
    media = {}
    for clip in clips:
        for m in clip.media:
            if m.media_id in media and media[m.media_id] != m:
                raise ValueError("inconsistent_media_identity")
            media[m.media_id] = m
    ordered = sorted(media.values(), key=lambda m: (m.seconds, m.media_id))
    if not ordered:
        raise ValueError("empty_index_media")
    # Include temporal endpoints, then spread shot representatives over the
    # entire video. Never consume the budget on the beginning of the story.
    representatives = []
    for clip in sorted(clips, key=lambda c: c.target_start):
        target = sorted(
            (m for m in clip.media if m.media_id in clip.target_ids), key=lambda m: m.seconds
        )
        if target:
            representatives.append(target[len(target) // 2])
    chosen = {m.media_id: m for m in (ordered[0], ordered[-1])}
    for m in even_sample(representatives, max(0, maximum - len(chosen))):
        chosen[m.media_id] = m
    for m in even_sample(ordered, maximum):
        if len(chosen) >= maximum:
            break
        chosen[m.media_id] = m
    selected = sorted(chosen.values(), key=lambda m: (m.seconds, m.media_id))
    selected_ids = set(chosen)
    report = {
        "available_frames": len(ordered),
        "selected_frames": len(selected),
        "selected_media_ids": [m.media_id for m in selected],
        "unrepresented_clip_ids": [c.clip_id for c in clips if not c.target_ids & selected_ids],
        "max_sample_gap_seconds": max(
            (b.seconds - a.seconds for a, b in zip(selected, selected[1:])), default=0
        ),
        "detail_coverage_certified": False,
    }
    return selected, report


class GlobalIndexClient:
    def __init__(
        self,
        *,
        model,
        base_url,
        max_frames=128,
        max_tokens=4096,
        image_max_edge=768,
        request_timeout=180,
        max_candidates=32,
        mode="video",
        video_transport="rrt-v1",
        schema_mode="json_object",
        task="story",
        generation=None,
    ):
        self.model = model
        self.base_url = base_url
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.image_max_edge = image_max_edge
        self.request_timeout = request_timeout
        self.max_candidates = max_candidates
        if mode not in {"images", "video"} or video_transport not in {"standard", "rrt-v1"}:
            raise ValueError("invalid_index_transport")
        self.mode = mode
        self.video_transport = video_transport
        if schema_mode not in {"json_schema", "json_object"}:
            raise ValueError("invalid_index_schema_mode")
        self.schema_mode = schema_mode
        if task not in {"story", "compact_story", "grounded"}:
            raise ValueError("invalid_index_task")
        self.task = task
        self.generation = {"temperature": 0} if generation is None else dict(generation)
        allowed = {"temperature", "top_p", "top_k", "repetition_penalty", "seed"}
        if not self.generation.keys() <= allowed:
            raise ValueError("unsupported_generation_option")

    def build_request(self, clips):
        selected, coverage = storyboard(clips, self.max_frames)
        aliases = {f"f{i}": m for i, m in enumerate(selected)}
        content, presentations, frames, timeline = [], [], [], []
        for alias, m in aliases.items():
            encoded, geometry = jpeg(m, self.image_max_edge)
            frames.append(encoded)
            timeline.append(
                {"media_id": alias, "seconds": m.seconds, "source_frame_index": m.index}
            )
            if self.mode == "images":
                content.extend(
                    [
                        {"type": "text", "text": f"OVERVIEW {alias} at {m.seconds:.3f}s"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded},
                        },
                    ]
                )
            presentations.append({"alias": alias, **asdict(m), **geometry})
        source = clips[0]
        metadata = {
            "fps": source.fps,
            "frames_indices": [m.index for m in selected],
            "total_num_frames": source.total_frames,
            "duration": source.duration,
            "do_sample_frames": False,
        }
        if self.mode == "video":
            from .video_transport import pack, validate_metadata

            if any(not c.constant_frame_rate for c in clips):
                raise ValueError("VFR_source_requires_explicit_timestamp_transport")
            if any(
                (c.fps, c.total_frames, c.duration, c.start_seconds)
                != (source.fps, source.total_frames, source.duration, source.start_seconds)
                for c in clips
            ):
                raise ValueError("inconsistent_video_metadata")
            if any(
                abs(m.seconds - (source.start_seconds + m.index / source.fps)) > 1 / source.fps
                for m in selected
            ):
                raise ValueError("source_timestamp_index_mismatch")
            validate_metadata(metadata, len(frames))
            url = (
                pack(frames, metadata)
                if self.video_transport == "rrt-v1"
                else ("data:video/jpeg;base64," + ",".join(frames))
            )
            content.append({"type": "video_url", "video_url": {"url": url}})
        content.append(
            {
                "type": "text",
                "text": "SOURCE_TIMELINE\\n"
                + json.dumps(
                    {
                        "frames": timeline,
                        "video_timestamp_offset_seconds": source.start_seconds,
                        "sampled_video": True,
                        "detail_coverage_certified": False,
                    },
                    separators=(",", ":"),
                ),
            }
        )
        wire = index_schema(aliases, self.max_candidates)
        prompt = INDEX_PROMPT
        if self.task in {"story", "compact_story"}:
            from .story_index import STORY_PROMPT, story_schema

            prompt = STORY_PROMPT
            wire = story_schema(
                source.start_seconds, source.start_seconds + source.duration, self.max_candidates
            )
            if self.task == "compact_story":
                from .story_index import COMPACT_STORY_PROMPT, compact_story_schema

                prompt = COMPACT_STORY_PROMPT
                wire = compact_story_schema(
                    source.start_seconds,
                    source.start_seconds + source.duration,
                    self.max_candidates,
                )
            # Native video carries source timestamps. Do not repeat a long table
            # of frame IDs in the story task, which never emits frame references.
            content[-1] = {
                "type": "text",
                "text": json.dumps(
                    {
                        "source_video_seconds": [
                            source.start_seconds,
                            source.start_seconds + source.duration,
                        ],
                        "video_timestamp_offset_seconds": source.start_seconds,
                        "sampled_frames": len(selected),
                        "sampling_gaps": coverage["max_sample_gap_seconds"],
                        "purpose": "coarse_search_plan_only",
                    },
                    separators=(",", ":"),
                ),
            }
        body = {
            "model": self.model,
            **self.generation,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "global_candidate_index", "schema": transport_schema(wire)},
            },
        }
        if self.schema_mode == "json_object" or self.task in {"story", "compact_story"}:
            # Show the contract to the model; do not rely on token masks to teach
            # provenance. Full validation still runs before materialization.
            content.append(
                {
                    "type": "text",
                    "text": "OUTPUT_SCHEMA\n" + json.dumps(wire, separators=(",", ":")),
                }
            )
            if self.schema_mode == "json_object":
                body["response_format"] = {"type": "json_object"}
        if self.mode == "video":
            body["mm_processor_kwargs"] = {"do_sample_frames": False}
            if self.video_transport == "standard":
                body["media_io_kwargs"] = {"video": metadata}
        audit = {
            "prompt_version": {
                "story": "global-story-v3",
                "compact_story": "global-story-v5",
                "grounded": "global-index-video-v2",
            }[self.task],
            "task": self.task,
            "prompt": prompt,
            "video_id": clips[0].video_id,
            "presentations": presentations,
            "coverage": coverage,
            "schema": wire,
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "query_blind": True,
            "mode": self.mode,
            "schema_mode": self.schema_mode,
            "generation": self.generation,
            "video_transport": self.video_transport if self.mode == "video" else None,
            "video_metadata": metadata if self.mode == "video" else None,
            "timeline": timeline,
            "actual_server_sampling": "unverified; match processor audit before acceptance",
        }
        return body, audit

    def build(self, clips, *, out, deadline=None, sender=None):
        deadline = deadline or Deadline()
        deadline.require()
        body, audit = self.build_request(clips)
        key = hashlib.sha256(
            json.dumps({"body": body, "audit": audit}, sort_keys=True).encode()
        ).hexdigest()
        run = Path(out) / key
        run.mkdir(parents=True, exist_ok=True)
        self.last_cache_hit = (run / "result.json").exists()
        if self.last_cache_hit:
            return json.loads((run / "result.json").read_text())
        if (run / "failure.json").exists():
            raise RuntimeError("cached_index_failure; inspect response or change inputs")
        atomic_json(run / "request.json", audit)
        started = time.monotonic()
        response = {}
        try:
            deadline.require()
            if sender is not None:
                response = sender(body)
            else:
                headers = {"Content-Type": "application/json"}
                if os.environ.get("RRT_VLM_API_KEY"):
                    headers["Authorization"] = "Bearer " + os.environ["RRT_VLM_API_KEY"]
                req = urllib.request.Request(
                    self.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers=headers,
                )
                with urllib.request.urlopen(
                    req, timeout=min(self.request_timeout, deadline.remaining())
                ) as h:
                    response = json.load(h)
            atomic_json(run / "response.json", response)
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("truncated_global_index")
            raw = json.loads(choice["message"]["content"])
            Draft202012Validator(audit["schema"]).validate(raw)
            if self.task in {"story", "compact_story"}:
                from .story_index import materialize_story

                if self.task == "compact_story":
                    from .story_index import expand_compact_story

                    source = clips[0]
                    raw = expand_compact_story(
                        raw, source.start_seconds, source.start_seconds + source.duration
                    )
                result = materialize_story(raw, audit, clips, key, self.max_candidates)
                atomic_json(run / "result.json", result)
                return result
            ids = [c["candidate_id"] for c in raw["candidates"]]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate_candidate_id")
            media = {p["alias"]: p for p in audit["presentations"]}
            candidates = []
            prefix = clips[0].video_id + ":candidate:"
            for c in raw["candidates"]:
                c = copy.deepcopy(c)
                c["candidate_id"] = prefix + c["candidate_id"]
                c["status"] = "hypothesis"
                for n, anchor in enumerate(c["anchors"]):
                    a, b, x, y = anchor["box"]
                    if not a < x or not b < y:
                        raise ValueError("invalid_candidate_box")
                    m = media[anchor["media_id"]]
                    anchor.update(
                        media_id=m["media_id"],
                        box=[v / 1000 for v in anchor["box"]],
                        endpoint_id=c["candidate_id"] + f":anchor:{n}",
                    )
                candidates.append(c)
            episodes = []

            def seconds(mid):
                m = media[mid]
                return m["pts"] * m["time_base"][0] / m["time_base"][1]

            for i, e in enumerate(raw["episodes"]):
                e = copy.deepcopy(e)
                e["start"], e["end"] = seconds(e["start_frame"]), seconds(e["end_frame"])
                if e["start"] > e["end"]:
                    raise ValueError("reversed_episode_frames")
                if not set(e["candidate_ids"]) <= set(ids):
                    raise ValueError("undeclared_episode_candidate")
                if any(not e["start"] <= seconds(mid) <= e["end"] for mid in e["evidence_ids"]):
                    raise ValueError("episode_evidence_outside_span")
                episodes.append(
                    {
                        **e,
                        "episode_id": f"episode:{i}",
                        "status": "hypothesis",
                        "candidate_ids": [prefix + x for x in e["candidate_ids"]],
                        "evidence_ids": [media[x]["media_id"] for x in e["evidence_ids"]],
                        "start_frame": media[e["start_frame"]]["media_id"],
                        "end_frame": media[e["end_frame"]]["media_id"],
                        "boundary_precision": "sampled_frame_extent; not_exact_event_boundary",
                    }
                )
            semantics = []
            for i, label in enumerate(raw["semantic_labels"]):
                if not {label["candidate_id"], *label["related_candidate_ids"]} <= set(ids):
                    raise ValueError("undeclared_semantic_candidate")
                semantics.append(
                    {
                        **label,
                        "semantic_id": prefix + f"semantic:{i}",
                        "candidate_id": prefix + label["candidate_id"],
                        "related_candidate_ids": [
                            prefix + x for x in label["related_candidate_ids"]
                        ],
                        "evidence_ids": [media[x]["media_id"] for x in label["evidence_ids"]],
                        "status": "hypothesis",
                        "semantic_support_audited": False,
                        "eligible_for_identity_merge": False,
                        "eligible_for_qa_support": False,
                    }
                )
            gaps = list(raw["gaps"])
            if len(candidates) >= self.max_candidates:
                gaps.append("candidate_capacity_reached; cast_not_complete")
            if len(episodes) >= 24:
                gaps.append("episode_capacity_reached")
            result = {
                "version": key,
                "video_id": clips[0].video_id,
                "candidates": candidates,
                "episodes": episodes,
                "semantic_labels": semantics,
                "evidence": [
                    {k: p[k] for k in ("media_id", "uri", "pts", "time_base", "sha256", "index")}
                    for p in audit["presentations"]
                ],
                "gaps": gaps,
                "coverage": audit["coverage"],
                "status": "hypothesis",
                "query_blind": True,
                "semantic_support_audited": False,
            }
            atomic_json(run / "result.json", result)
            return result
        except Exception as error:
            atomic_json(run / "failure.json", {"kind": type(error).__name__, "error": str(error)})
            raise
        finally:
            atomic_json(
                run / "timing.json",
                {"elapsed_seconds": time.monotonic() - started, "usage": response.get("usage")},
            )


def select_references(index, clip, media, folder, *, max_candidates=3, max_views=6):
    """Retrieve hypotheses by episode/time, not commit by text similarity.

    Keep a stable anchor plus a nearby alternate when budget permits. Every
    anchor is a separate reference endpoint: coarse grouping is never a merge.
    """
    if index["video_id"] != clip.video_id or max_candidates < 1 or max_views < 1:
        raise ValueError("invalid_reference_selection")
    active = [
        e
        for e in index["episodes"]
        if e["start"] < clip.target_end and clip.target_start < e["end"]
    ]
    mentioned = {cid for e in active for cid in e["candidate_ids"]}
    if index.get("index_type") == "story_hypotheses":
        # Search windows are not marked people. Never fabricate a full-frame box
        # or canonical reference endpoint to satisfy the old reference interface.
        relevant = [
            c
            for c in index["candidates"]
            if c["candidate_id"] in mentioned
            or any(
                s["start"] < clip.target_end and clip.target_start < s["end"]
                for s in c["appearances"]
            )
        ]
        chosen = relevant[:max_candidates]
        return (), {
            "index_version": index["version"],
            "candidates": [
                {
                    k: c[k]
                    for k in (
                        "candidate_id",
                        "kind",
                        "label",
                        "description",
                        "uncertainty",
                        "status",
                    )
                }
                for c in chosen
            ],
            "episode_hints": active,
            "omitted_candidate_ids": [c["candidate_id"] for c in relevant[max_candidates:]],
            "identity_proven_by_index": False,
            "reference_gap": "coarse_intervals_need_local_grounding; no_identity_endpoint",
        }

    midpoint = (clip.target_start + clip.target_end) / 2

    def rank(c):
        distance = min(abs(media[a["media_id"]].seconds - midpoint) for a in c["anchors"])
        return (c["candidate_id"] not in mentioned, distance, c["candidate_id"])

    ranked = sorted(index["candidates"], key=rank)
    chosen = ranked[: min(max_candidates, max_views)]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    refs, context, omissions = [], [], []
    for c in chosen:
        # First anchor is stable across requests; alternatives retain their own
        # identity until the detailed observations actually connect them.
        ordered = sorted(
            c["anchors"][1:], key=lambda a: abs(media[a["media_id"]].seconds - midpoint)
        )
        count = max(1, max_views // max(1, len(chosen)))
        anchors = [c["anchors"][0], *ordered][:count]
        ref_ids = []
        for anchor in anchors:
            m = media[anchor["media_id"]]
            if digest(m.uri) != m.sha256:
                raise ValueError("reference_source_changed")
            endpoint = anchor["endpoint_id"]
            key = hashlib.sha256(
                json.dumps(
                    {"endpoint": endpoint, "source": m.sha256, "box": anchor["box"]}, sort_keys=True
                ).encode()
            ).hexdigest()
            path = folder / (key + ".jpg")
            if not path.exists():
                with Image.open(m.uri) as source:
                    image = source.convert("RGB")
                    draw = ImageDraw.Draw(image)
                    a, b, x, y = anchor["box"]
                    draw.rectangle(
                        (a * image.width, b * image.height, x * image.width, y * image.height),
                        outline="yellow",
                        width=3,
                    )
                    image.save(path, quality=92)
            view = replace(
                m, media_id="reference:" + key, uri=str(path.resolve()), sha256=digest(path)
            )
            ref_id = "ref:" + key[:16]
            refs.append(
                Reference(
                    ref_id,
                    endpoint,
                    c["kind"],
                    (view,),
                    (
                        {
                            "source_media_id": m.media_id,
                            "source_sha256": m.sha256,
                            "box": anchor["box"],
                            "instance_id": endpoint,
                            "candidate_id": c["candidate_id"],
                            "selection_method": "episode_time; marked_full_frame; hypothesis_only",
                        },
                    ),
                )
            )
            ref_ids.append(ref_id)
        context.append(
            {
                "candidate_id": c["candidate_id"],
                "kind": c["kind"],
                "label": c["label"],
                "description": c["description"],
                "uncertainty": c["uncertainty"],
                "reference_ids": ref_ids,
                "status": "hypothesis",
            }
        )
    chosen_ids = {c["candidate_id"] for c in chosen}
    omissions.extend(c["candidate_id"] for c in ranked if c["candidate_id"] not in chosen_ids)
    # Candidate entries may not name unsupplied references. Episode hints can
    # mention omitted IDs, but that omission is explicit in the retrieval audit.
    return tuple(refs), {
        "index_version": index["version"],
        "candidates": context,
        "episode_hints": active,
        "omitted_candidate_ids": omissions,
        "identity_proven_by_index": False,
    }


def contextual_clip(index, clip, media, *, maximum_context_frames=16):
    """Expand visible context to the coarse action interval without changing TARGET.

    The original shot-aware windows still schedule work. This allows an action's
    establishing/ending shots to be seen in the same request as a close-up.
    """
    if maximum_context_frames < 2:
        raise ValueError("context_budget_requires_two_frames")
    episodes = [
        e
        for e in index["episodes"]
        if e["start"] < clip.target_end and clip.target_start < e["end"]
    ]
    original_context = [m for m in clip.media if m.media_id not in clip.target_ids]
    start = min(
        [clip.target_start, *[m.seconds for m in original_context], *[e["start"] for e in episodes]]
    )
    end = max(
        [clip.target_end, *[m.seconds for m in original_context], *[e["end"] for e in episodes]]
    )
    candidates = sorted(
        (
            m
            for m in media.values()
            if start <= m.seconds <= end and m.media_id not in clip.target_ids
        ),
        key=lambda m: (m.seconds, m.media_id),
    )
    # Reserve both sides, so a long preceding scene cannot hide the ending shot.
    before = [m for m in candidates if m.seconds < clip.target_start]
    after = [m for m in candidates if m.seconds >= clip.target_end]
    before_budget = min(len(before), maximum_context_frames // 2)
    after_budget = min(len(after), maximum_context_frames - before_budget)
    before_budget = min(len(before), maximum_context_frames - after_budget)
    context = [*even_sample(before, before_budget), *even_sample(after, after_budget)]
    target = [m for m in clip.media if m.media_id in clip.target_ids]
    combined = sorted([*target, *context], key=lambda m: (m.seconds, m.media_id))
    sampling = {
        **clip.sampling,
        "context_policy": "episode_and_adjacent; target_unchanged",
        "context_episode_ids": [e["episode_id"] for e in episodes],
        "available_context_frames": len(candidates),
        "selected_context_frames": len(context),
        "context_budget_limited": len(context) < len(candidates),
        "context_max_sample_gap_seconds": max(
            (b.seconds - a.seconds for a, b in zip(context, context[1:])), default=0
        ),
    }
    return replace(clip, media=tuple(combined), sampling=sampling)
