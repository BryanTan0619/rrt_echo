"""Compact offline story hypotheses. Spatial grounding belongs to the local pass."""

import copy

from .wire import arr, enum, obj, string

STORY_PROMPT = """Watch the sampled video as a whole and make a concise search plan.
Return recurring people/important objects, major action phases, and possible
names or relational roles. Use neutral candidate IDs; briefly describe each
candidate once and note appearance changes. A later local pass will locate them.
Give a few approximate appearance intervals in source-video seconds. These are
places to inspect, not continuous presence claims. Never output boxes, frame
lists or per-frame descriptions. Do not repeat an entry to fill a quota.
Summarize actions with candidate participants, not vague facial expressions.
Names/roles need a concrete observed basis or an explicitly marked inference;
gender, clothing or carrying someone alone does not establish kinship.
Do not force every observation into the initial cast. Keep uncertainty and gaps.
All output is provisional: neither the narrative nor a role confirms identity.
Return one JSON object with candidates, episodes, semantic_labels, gaps.
"""


def story_schema(start, end, max_candidates=32):
    seconds = {"type": "number", "minimum": start, "maximum": end}
    span = obj({"start": seconds, "end": seconds})
    cid = string(32)
    return obj(
        {
            "candidates": arr(
                obj(
                    {
                        "candidate_id": cid,
                        "kind": enum(["person", "object"]),
                        "label": string(80),
                        "description": string(240),
                        "appearances": arr(span, 4, 1),
                        "uncertainty": {"type": "string", "maxLength": 160},
                    }
                ),
                max_candidates,
            ),
            "episodes": arr(
                obj(
                    {
                        "start": seconds,
                        "end": seconds,
                        "summary": string(240),
                        "candidate_ids": arr(cid, max_candidates),
                    }
                ),
                16,
            ),
            "semantic_labels": arr(
                obj(
                    {
                        "candidate_id": cid,
                        "label": string(80),
                        "relation": string(80),
                        "related_candidate_ids": arr(cid, 4),
                        "basis_spans": arr(span, 3, 1),
                        "basis": string(240),
                        "epistemic_status": enum(["observed", "inferred", "unresolved"]),
                    }
                ),
                16,
            ),
            "gaps": arr(string(160), 16),
        }
    )


def materialize_story(raw, audit, clips, version, max_candidates):
    # JSON Schema validation is performed by the shared client before this call.
    result = copy.deepcopy(raw)
    ids = [c["candidate_id"] for c in raw["candidates"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_candidate_id")
    prefix = clips[0].video_id + ":candidate:"
    media = [
        {k: p[k] for k in ("media_id", "uri", "pts", "time_base", "sha256", "index")}
        for p in audit["presentations"]
    ]

    def search_span(span):
        if span["start"] >= span["end"]:
            raise ValueError("invalid_story_interval")
        return {
            **span,
            "status": "search_hint",
            "source_media_ids": [
                m["media_id"]
                for m in media
                if span["start"] <= m["pts"] * m["time_base"][0] / m["time_base"][1] <= span["end"]
            ],
            "continuous_presence_asserted": False,
        }

    def checked_ids(values):
        if not set(values) <= set(ids):
            raise ValueError("undeclared_story_candidate")
        return [prefix + x for x in values]

    requests = []
    for c in result["candidates"]:
        c["candidate_id"] = prefix + c["candidate_id"]
        c["appearances"] = [search_span(s) for s in c["appearances"]]
        c.update(status="hypothesis", grounding_status="pending_local_observation")
        requests.append(
            {
                "candidate_id": c["candidate_id"],
                "search_spans": c["appearances"],
                "task": "locate_candidate_in_context_and_jointly_extract_local_facts",
                "status": "pending",
                "creates_identity_endpoint": False,
            }
        )
    for i, e in enumerate(result["episodes"]):
        extent = search_span({"start": e["start"], "end": e["end"]})
        e.update(
            episode_id=f"episode:{i}",
            status="hypothesis",
            candidate_ids=checked_ids(e["candidate_ids"]),
            source_media_ids=extent["source_media_ids"],
            boundary_precision="approximate_search_interval; not_event_evidence",
        )
    for e in result["episodes"]:
        if "role_hypotheses" in e:
            e["role_hypotheses"] = {
                role: checked_ids([cid])[0] for role, cid in e["role_hypotheses"].items()
            }
    for i, label in enumerate(result["semantic_labels"]):
        label["candidate_id"] = checked_ids([label["candidate_id"]])[0]
        label["related_candidate_ids"] = checked_ids(label["related_candidate_ids"])
        label["basis_spans"] = [search_span(s) for s in label["basis_spans"]]
        label.update(
            semantic_id=prefix + f"semantic:{i}",
            status="hypothesis",
            semantic_support_audited=False,
            eligible_for_identity_merge=False,
            eligible_for_qa_support=False,
        )
    for field, cap in [("candidates", max_candidates), ("episodes", 16), ("semantic_labels", 16)]:
        if len(result[field]) == cap:
            result["gaps"].append(field + "_capacity_reached; coverage_not_certified")
    result.update(
        version=version,
        video_id=clips[0].video_id,
        index_type="story_hypotheses",
        evidence=media,
        coverage=audit["coverage"],
        grounding_requests=requests,
        status="hypothesis",
        query_blind=True,
        semantic_support_audited=False,
        spatial_grounding_complete=False,
    )
    return result


COMPACT_STORY_PROMPT = """Watch the entire video sample and outline its main story.
Return a short JSON object with cast, events, labels, gaps, using the supplied contract.
Cast: one row per plausible recurring person or plot-relevant object. Give one
identifiable time (at), not every appearance. Distinguish different people;
appearance or health changes alone need not mean a new person.
Events: major actions in order. Each role binding has role (such as agent or
patient) and entity_id (an exact cast ID, never an appearance description). Describe actual
interaction direction and visible state changes. Keep action phrases brief.
Labels: only personal names or interpersonal relationships, with one time and a
concrete basis. Omit clothing, appearance and object attributes here. Empty is valid.
Mark inferred relationships; do not infer kinship from carrying alone.
All times are approximate source-video seconds, all entries are search hypotheses.
Do not invent missing actions or connect people just because a story sounds plausible.
Use few representative entries rather than filling limits. No boxes, frame lists,
appearance-interval lists, repeated descriptions, or explanations outside JSON.
"""


def compact_story_schema(start, end, max_candidates=32):
    seconds = {"type": "number", "minimum": start, "maximum": end}
    return obj(
        {
            "cast": arr(
                obj(
                    {
                        "id": string(32),
                        "kind": enum(["person", "object"]),
                        "appearance": string(120),
                        "at": seconds,
                    }
                ),
                max_candidates,
            ),
            "events": arr(
                obj(
                    {
                        "start": seconds,
                        "end": seconds,
                        "action": string(160),
                        "roles": arr(obj({"role": string(32), "entity_id": string(32)}), 6),
                    }
                ),
                16,
            ),
            "labels": arr(
                obj(
                    {
                        "subject": string(32),
                        "label": string(64),
                        "relation": string(64),
                        "related": {"type": ["string", "null"], "maxLength": 32},
                        "at": seconds,
                        "basis": string(160),
                        "status": enum(["observed", "inferred", "unresolved"]),
                    }
                ),
                12,
            ),
            "gaps": arr(string(160), 12),
        }
    )


def expand_compact_story(raw, start, end):
    # Convert once, in code. No second caption-to-fact model call. A representative
    # timestamp creates a search window only, never an observed presence interval.
    ids = [c["id"] for c in raw["cast"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_candidate_id")

    def window(t):
        return {"start": max(start, t - 2), "end": min(end, t + 2)}

    candidates = [
        {
            "candidate_id": c["id"],
            "kind": c["kind"],
            "label": c["appearance"],
            "description": c["appearance"],
            "appearances": [window(c["at"])],
            "representative_time": c["at"],
            "uncertainty": "coarse_candidate_not_identity_verified",
        }
        for c in raw["cast"]
    ]
    episodes = []
    for e in raw["events"]:
        bindings = {b["role"]: b["entity_id"] for b in e["roles"]}
        if len(bindings) != len(e["roles"]):
            raise ValueError("duplicate_event_role")
        if not set(bindings.values()) <= set(ids):
            raise ValueError("undeclared_story_candidate")
        episodes.append(
            {
                "start": e["start"],
                "end": e["end"],
                "summary": e["action"],
                "candidate_ids": list(dict.fromkeys(bindings.values())),
                "role_hypotheses": bindings,
            }
        )
    labels = [
        {
            "candidate_id": x["subject"],
            "label": x["label"],
            "relation": x["relation"],
            "related_candidate_ids": [x["related"]] if x["related"] is not None else [],
            "basis_spans": [window(x["at"])],
            "basis": x["basis"],
            "epistemic_status": x["status"],
        }
        for x in raw["labels"]
    ]
    return {
        "candidates": candidates,
        "episodes": episodes,
        "semantic_labels": labels,
        "gaps": [
            *raw["gaps"],
            *(
                ["semantic_labels_capacity_reached; coverage_not_certified"]
                if len(raw["labels"]) == 12
                else []
            ),
        ],
    }
