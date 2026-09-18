"""Target-only temporal review, independent single-frame grounding, then box audit.

Model review is a proposal filter, not an independent quality certificate. Every
call and rejected role is retained. No caption, QA answer or hand audit is input.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from perception.local_stage import evenly
from perception.types import Deadline, atomic_json
from perception.vlm import jpeg
from perception.wire import arr, enum, joint_schema, materialize, obj, string, validate

# Bounded interaction vocabulary: unknown interactions remain explicit generic events.
EVENT_ROLES = {
    "hold": ("holder", "held"),
    "carry": ("holder", "held"),
    "write": ("writer", "instrument", "surface"),
    "bite": ("agent", "patient"),
    "chew": ("agent", "patient"),
    "touch": ("agent", "patient"),
    "attack": ("agent", "patient"),
    "give": ("agent", "theme", "recipient"),
    "take": ("agent", "theme"),
    "exit": ("agent", "source"),
    "enter": ("agent", "destination"),
    "lie_on": ("patient", "surface"),
    "lean_over": ("agent", "patient"),
    "look_at": ("agent", "patient"),
    "turn": ("agent",),
    "walk": ("agent",),
    "fall": ("agent",),
    "kneel": ("agent",),
    "look": ("agent",),
    "shoot": ("agent", "patient", "instrument"),
    "inspect": ("agent", "patient"),
    "remove": ("agent", "theme", "source"),
    "replace": ("agent", "patient", "theme"),
    "attach": ("agent", "theme", "destination"),
    "insert": ("agent", "theme", "destination"),
    "open": ("agent", "patient"),
    "close": ("agent", "patient"),
    "repair": ("agent", "patient", "instrument"),
    "interact": ("agent", "patient"),
}
PROPERTY_PREDICATES = (
    "text",
    "injury",
    "blood_stain",
    "damage",
    "posture",
    "expression",
    "gaze_direction",
    "orientation",
    "clothing",
    "hair_color",
    "eye_color",
    "appearance",
    "color",
    "material",
    "shape",
    "quantity",
    "condition",
)


def _enforce_role_contracts(reviewed):
    """Never invent an endpoint to satisfy a predicate's arity."""
    for fact in reviewed["facts"]:
        if fact["kind"] == "event":
            required = EVENT_ROLES.get(fact["predicate"])
            if required is None:
                if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", fact["predicate"]):
                    raise ValueError("invalid_event_predicate")
                required = tuple(r["role"] for r in fact["roles"])
                fact["open_event_contract"] = True
            names = {r["role"] for r in fact["roles"]}
            if names - set(required):
                raise ValueError("unexpected_event_role")
            fact["unresolved_slots"] = sorted(
                set(fact["unresolved_slots"]) | (set(required) - names)
            )
        elif fact["predicate"] not in PROPERTY_PREDICATES:
            raise ValueError("relation_or_unknown_predicate_in_property")
    return reviewed


def enforce_role_contracts(reviewed):
    """Quarantine invalid candidates without discarding unrelated valid facts."""
    valid = []
    for n, fact in enumerate(reviewed["facts"]):
        try:
            _enforce_role_contracts({"facts": [fact]})
        except ValueError as exc:
            reviewed.setdefault("rejected", []).append(
                {"id": f"contract:{n}", "reason": str(exc), "candidate": fact}
            )
        else:
            valid.append(fact)
    reviewed["facts"] = valid
    return reviewed


def review_schema(indices):
    frame = {"type": "integer", "enum": list(indices)}
    kinds = enum(["person", "object", "region"])
    role = obj(
        {
            "role": enum(
                [
                    "agent",
                    "patient",
                    "holder",
                    "held",
                    "writer",
                    "instrument",
                    "surface",
                    "source",
                    "destination",
                    "recipient",
                    "theme",
                ]
            ),
            "kind": kinds,
            "description": string(160),
        }
    )
    owner = obj({"role": enum(["owner"]), "kind": kinds, "description": string(160)})
    event = obj(
        {
            "kind": enum(["event"]),
            "predicate": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$", "maxLength": 64},
            "value": {"type": "null"},
            "frames": arr(frame, 4, 1),
            "roles": arr(role, 6, 1),
            "unresolved_slots": arr(string(32), 6),
        }
    )
    prop = obj(
        {
            "kind": enum(["text", "state", "attribute"]),
            "predicate": enum(PROPERTY_PREDICATES),
            "value": string(256),
            "frames": arr(frame, 4, 1),
            "roles": arr(owner, 1, 1),
            "unresolved_slots": arr(string(32), 6),
        }
    )
    return obj(
        {
            "facts": arr({"oneOf": [event, prop]}, 8),
            "rejected": arr(obj({"id": string(32), "reason": string(240)}), 24),
            "gaps": arr(string(240), 12),
        }
    )


def grounding_schema(subjects):
    box = {
        "type": ["array", "null"],
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    return obj(
        {
            "regions": arr(
                obj(
                    {
                        "id": enum(s["id"] for s in subjects),
                        "visible": {"type": "boolean"},
                        "box": box,
                    }
                ),
                len(subjects),
            )
        }
    )


def box_audit_schema(subjects):
    return obj(
        {
            "regions": arr(
                obj(
                    {
                        "id": enum(s["id"] for s in subjects),
                        "verdict": enum(["supported", "contradicted", "insufficient"]),
                        "reason": string(240),
                        "observed_kind": enum(["person", "object", "region", "unclear"]),
                        "observed_age_group": enum(["adult", "child", "not_person", "unclear"]),
                        "matches_requested_subject": {"type": "boolean"},
                    }
                ),
                len(subjects),
            )
        }
    )


def fact_audit_schema(indices):
    return obj(
        {
            "verdict": enum(["supported", "contradicted", "insufficient"]),
            "reason": string(400),
            "missing_roles": arr(string(32), 8),
            "supported_frames": arr({"type": "integer", "enum": list(indices)}, 4),
        }
    )


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def validate_review(raw, media):
    """Membership checks never substitute for temporal visual evidence."""
    if not isinstance(raw, dict) or not isinstance(raw.get("facts"), list):
        raise ValueError("missing_review_facts")
    if len(raw["facts"]) > 12:
        raise ValueError("review_fact_budget_exceeded")
    for fact in raw["facts"]:
        if fact["kind"] not in {"event", "text", "state", "attribute"}:
            raise ValueError("invalid_fact_kind")
        if not isinstance(fact["predicate"], str) or not fact["predicate"].strip():
            raise ValueError("missing_predicate")
        frames = fact["frames"]
        if (
            not frames
            or len(frames) > 4
            or any(type(f) is not int or f not in media for f in frames)
        ):
            raise ValueError("evidence_not_in_supplied_target")
        if not fact["roles"] or len(fact["roles"]) > 6:
            raise ValueError("invalid_roles")
        names = set()
        for role in fact["roles"]:
            if role["role"] in names or role["kind"] not in {"person", "object", "region"}:
                raise ValueError("invalid_or_duplicate_role")
            names.add(role["role"])
            if "frame" in role and (role["frame"] not in frames or role["frame"] not in media):
                raise ValueError("role_frame_outside_fact")
            if not isinstance(role["description"], str) or not role["description"].strip():
                raise ValueError("missing_visual_subject")
        if fact["kind"] != "event" and ("owner" not in names or not isinstance(fact["value"], str)):
            raise ValueError("property_requires_owner_and_value")
        if not isinstance(fact.get("unresolved_slots"), list) or names & set(
            fact["unresolved_slots"]
        ):
            raise ValueError("invalid_unresolved_slots")
    return raw


def usable_box(row):
    box = row.get("box")
    return (
        row.get("visible") is True
        and isinstance(box, list)
        and len(box) == 4
        and all(type(v) is int and 0 <= v <= 1000 for v in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


class VerifiedPerceptionClient:
    def __init__(
        self,
        *,
        model,
        base_url,
        max_tokens=3500,
        request_timeout=120,
        target_frames=24,
        sender=None,
        multimodal_context=None,
    ):
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.target_frames = target_frames
        self.sender = sender
        self.multimodal_context = multimodal_context

    def call(self, content, *, out, deadline, task, tokens=None, schema=None):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "repetition_penalty": 1.05,
            "max_tokens": tokens or self.max_tokens,
            "response_format": {"type": "json_object"},
            "mm_processor_kwargs": {"max_pixels": 401408 if task == "temporal_review" else 1048576},
        }
        if schema is not None:
            from perception.vlm import transport_schema

            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": task, "schema": transport_schema(schema)},
            }
        key = hashlib.sha256(
            json.dumps({"body": body, "base_url": self.base_url}, sort_keys=True).encode()
        ).hexdigest()
        path = Path(out) / task / key
        path.mkdir(parents=True, exist_ok=True)
        if (path / "result.json").exists():
            return json.loads((path / "result.json").read_text())
        if (path / "failure.json").exists():
            raise ValueError("cached_verification_failure")
        safe = []
        for c in content:
            if c["type"] == "image_url":
                safe.append(
                    {
                        "type": "image_sha256",
                        "sha256": hashlib.sha256(c["image_url"]["url"].encode()).hexdigest(),
                    }
                )
            else:
                safe.append(c)
        atomic_json(
            path / "request.json",
            {
                "task": task,
                "model": self.model,
                "messages": safe,
                "max_tokens": body["max_tokens"],
                "mm_processor_kwargs": body["mm_processor_kwargs"],
            },
        )
        started = time.monotonic()
        response = {}
        try:
            deadline.require()
            if self.sender:
                response = self.sender(body)
            else:
                req = urllib.request.Request(
                    self.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(
                    req, timeout=min(self.request_timeout, deadline.remaining())
                ) as h:
                    response = json.load(h)
            atomic_json(path / "response.json", response)
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("truncated_verification_output")
            result = parse_json(choice["message"]["content"])
            atomic_json(path / "result.json", result)
            return result
        except Exception as e:
            atomic_json(path / "failure.json", {"error": str(e)})
            raise
        finally:
            atomic_json(
                path / "timing.json",
                {"seconds": time.monotonic() - started, "usage": response.get("usage")},
            )

    def observe(self, clip, *, out, deadline=None, proposals=None, references=(), detections=()):
        deadline = deadline or Deadline()
        out = Path(out) / clip.clip_id.replace(":", "_")
        out.mkdir(parents=True, exist_ok=False)
        # Baseline predictions are audit-only; fresh extraction never consumes them.
        if proposals is None:
            proposals = {
                "observation": {
                    "observation_id": clip.clip_id + ":unseeded",
                    "instances": [],
                    "facts": [],
                }
            }
        atomic_json(out / "input_proposals.json", proposals)
        target = evenly(
            [m for m in clip.media if m.media_id in clip.target_ids], self.target_frames
        )
        media = {m.index: m for m in target}
        content = []
        for m in target:
            encoded, _ = jpeg(m, 1024)
            content.extend(
                [
                    {"type": "text", "text": f"TARGET frame {m.index}, time {m.seconds:.2f}s"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + encoded},
                    },
                ]
            )
        context = evenly([m for m in clip.media if m.media_id not in clip.target_ids], 8)
        for m in context:
            encoded, _ = jpeg(m, 1024)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"CONTEXT ONLY frame {m.index}, time {m.seconds:.2f}s; not eligible as target evidence",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + encoded},
                    },
                ]
            )
        obs = proposals["observation"]
        prompt = """Examine ordered TARGET images with the separately labelled CONTEXT images. CONTEXT may clarify the action process, but cannot supply a missing target participant or be cited as target evidence.
Return only facts directly supported by these images, localizing
actual evidence. A later/earlier action must not be assigned to a visible but unrelated
frame. An inscription is text, never proof that somebody is currently writing.
Do not infer causes or intentions from injury or objects. An event requires visible
motion/change/contact across the selected frames; cite before and during/after when
available. Do not fill missing roles with nearby people or use an infant as a tool.
An isolated hand/arm/inscribed skin is a region, not a new person. Its owner may remain
unknown; do not substitute a face from another time for the action's region.
Use literal OCR only when readable. A property owner should be the actual surface/subject.
Merge repetitions of one occurrence. Include at most 8 useful facts, fewer is fine. Prioritize interactions and readable text over routine appearance.
For each role give a neutral description of a subject visible in at least one of the event evidence frames. A later spatial stage will find its frame and box. Do not provide role-level frame numbers or boxes. Use semantic roles, not subject names: agent/patient, holder/held, writer/instrument/surface, owner. A transitive event needs both actor and acted-on subject, or explicitly name the missing role in unresolved_slots. A wound or damaged car is a state, not an observed impact event. Predicates must be short lowercase verbs/attributes with underscores; never sentences containing participants.
Output JSON of this exact shape (example values are placeholders):
{"facts":[{"kind":"event|state|attribute|text","predicate":"observed predicate",
"value":null,"frames":[123,124],"roles":[{"role":"role name","kind":"person|object|region",
"description":"visible subject"}],"unresolved_slots":[]}],
"rejected":[{"id":"candidate id","reason":"why unsupported"}],"gaps":["missing evidence"]}
Use value=null for events, actual string value for properties, and role=owner for properties.
All frame indices MUST be from supplied TARGET images. Empty facts is allowed.
People including infants are kind=person. Objects are inanimate. Isolated body parts and skin surfaces are kind=region. kind=text is ONLY actual written characters, never bloodstains or injury descriptions. Use short predicates and directed roles such as agent/patient, holder/held, writer/instrument/surface. No candidate facts are provided; rejected can be empty.
"""
        prompt += (
            "\nKNOWN EVENT CONTRACTS (use these exact roles when a predicate applies; otherwise preserve the observed predicate): "
            + json.dumps(EVENT_ROLES)
        )
        prompt += "\nProperties describe intrinsic appearance, posture or literal text ONLY. Holding, chewing, carrying and contact MUST be events with separate participants, NEVER a yes/no property or a value containing the interaction. For text use predicate=text, owner=the visible inscribed region. Do not substitute the presumed whole person. If an interaction is not in the vocabulary preserve a short snake_case predicate with explicit roles from the role vocabulary. Missing participants belong in unresolved_slots, never invented boxes."
        prompt += "\nFor replace: patient is the removed old item, theme is the newly installed item. Do not merge them because they look alike. Inspecting damage is not causing damage. A result such as a person falling does not by itself prove who fired a shot. Keep separate occurrences, participants and instruments distinct. Text overlays/subtitles are observations of text; attributing their claim to a depicted person requires explicit reference evidence, not physical part_of."
        if self.multimodal_context:
            prompt += "\nMULTIMODAL NAVIGATION CONTEXT (not pixel evidence):\n" + json.dumps(
                self.multimodal_context
            )
        content.append({"type": "text", "text": prompt})
        reviewed = self.call(
            content, out=out, deadline=deadline, task="temporal_review", schema=review_schema(media)
        )
        validate_review(reviewed, media)
        enforce_role_contracts(reviewed)
        atomic_json(out / "temporal_review.json", reviewed)
        requested = {}
        for n, f in enumerate(reviewed["facts"]):
            for role in f["roles"]:
                key = f"q{n}_{role['role']}"
                for frame in dict.fromkeys(f["frames"]):
                    requested.setdefault(frame, []).append(
                        {"id": key, "description": role["description"], "kind": role["kind"]}
                    )
        grounded = {}
        rejected = []
        for frame, subjects in requested.items():
            subjects = [subject for subject in subjects if subject["id"] not in grounded]
            if not subjects:
                continue
            atomic_json(
                out / "progress.json",
                {
                    "stage": "ground_and_audit",
                    "frame": frame,
                    "completed_roles": len(grounded),
                    "total_frames": len(requested),
                },
            )
            m = media[frame]
            encoded, _ = jpeg(m, 1280)
            image = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}
            try:
                boxes = self.call(
                    [
                        image,
                        {
                            "type": "text",
                            "text": """Locate each requested subject in this SINGLE image independently.
Output JSON {"regions":[{"id":"request id","visible":true,"box":[x0,y0,x1,y1]}]}.
Coordinates are integer xyxy normalized 0..1000 across the FULL image, including black bars.
Use a tight box around the visible subject. Do not invent an off-screen subject. If absent,
set visible=false and box=null. Do not substitute a hand for its tool, another arm for
the surface, or a carrier's body for a passenger. No default boxes. Requests:
"""
                            + json.dumps(subjects),
                        },
                    ],
                    out=out,
                    deadline=deadline,
                    task="single_frame_grounding",
                    tokens=1600,
                    schema=grounding_schema(subjects),
                )
                rows = boxes.get("regions", [])
                if len({r["id"] for r in rows}) != len(rows):
                    raise ValueError("duplicate_grounding_id")
                by_id = {r["id"]: r for r in rows}
                valid = {
                    s["id"]: by_id[s["id"]]
                    for s in subjects
                    if s["id"] in by_id and usable_box(by_id[s["id"]])
                }
                # Box audit is a new call using the real full frame with marks, never the grounder's rationale.
                audits = {}
                if valid:
                    im = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                    draw = ImageDraw.Draw(im)
                    for j, (key, row) in enumerate(valid.items()):
                        a, b, c, d = row["box"]
                        xy = [
                            a * im.width / 1000,
                            b * im.height / 1000,
                            c * im.width / 1000,
                            d * im.height / 1000,
                        ]
                        draw.rectangle(
                            xy, outline=["red", "yellow", "cyan", "lime"][j % 4], width=3
                        )
                        draw.text(
                            (xy[0] + 2, xy[1] + 2),
                            key,
                            fill="white",
                            stroke_width=1,
                            stroke_fill="black",
                        )
                    im.save(out / f"grounded_{frame}.jpg", quality=95)
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=95)
                    answer = self.call(
                        [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,"
                                    + base64.b64encode(buf.getvalue()).decode()
                                },
                            },
                            {
                                "type": "text",
                                "text": """Audit the labeled boxes against the image. For EACH request, does its labeled box
actually localize that subject? Reject boxes on background, another person, a hand instead
of a pen, the wrong body part, or a tiny unrelated sliver. Partial occlusion is okay only
if the requested subject remains identifiable. A sliver of clothing alone is insufficient for a person anchor.
A region box must isolate that body part or surface, not cover the whole person/frame.
First describe the visible age group in EACH box: observed_age_group=adult|child|not_person|unclear. A box on an adult cannot satisfy a baby/child request, even if a baby is elsewhere in the frame. Set matches_requested_subject only if the actual boxed subject matches this request.
Also return observed_kind=person|object|region|unclear based on pixels: a detached arm close-up is region even if the request says person. Do not assume a label makes a box correct.
Return JSON {"regions":[{"id":"request id","verdict":"supported|contradicted|insufficient","reason":"visual reason"}]}.
Requests: """
                                + json.dumps([s for s in subjects if s["id"] in valid]),
                            },
                        ],
                        out=out,
                        deadline=deadline,
                        task="box_audit",
                        tokens=1800,
                        schema=box_audit_schema(
                            [subject for subject in subjects if subject["id"] in valid]
                        ),
                    )
                    audits = {r["id"]: r for r in answer.get("regions", [])}
                for subject in subjects:
                    key = subject["id"]
                    if (
                        key in valid
                        and subject_audit_matches(subject, audits.get(key, {}))
                        and audits[key].get("observed_kind") in {"person", "object", "region"}
                        and not (
                            subject["kind"] == "region" and audits[key]["observed_kind"] != "region"
                        )
                    ):
                        grounded[key] = {
                            "frame": frame,
                            "box": valid[key]["box"],
                            "description": subject["description"],
                            "kind": audits[key]["observed_kind"],
                        }
                    else:
                        rejected.append(
                            {
                                "role_id": key,
                                "frame": frame,
                                "reason": "grounding_or_visual_audit_failed",
                                "audit": audits.get(key),
                            }
                        )
            except Exception as e:
                rejected.extend(
                    {"role_id": s["id"], "frame": frame, "reason": str(e)} for s in subjects
                )
            atomic_json(out / "grounded_roles.json", grounded)
            atomic_json(out / "rejected_roles.json", rejected)
        rejected = [row for row in rejected if row["role_id"] not in grounded]
        fact_audits = []
        raw = {"instances": [], "facts": [], "links": [], "coverage_gaps": []}
        for n, f in enumerate(reviewed["facts"]):
            roles = {}
            evidence = {"predicate": [str(i) for i in dict.fromkeys(f["frames"])]}
            missing = list(f["unresolved_slots"])
            for role in f["roles"]:
                key = f"q{n}_{role['role']}"
                g = grounded.get(key)
                if not g:
                    missing.append(role["role"])
                    continue
                iid = f"i{len(raw['instances'])}"
                roles[role["role"]] = iid
                evidence[role["role"]] = [str(g["frame"])]
                raw["instances"].append(
                    {
                        "instance_id": iid,
                        "kind": g["kind"],
                        "description": g["description"][:160],
                        "regions": [{"media_id": str(g["frame"]), "box": g["box"]}],
                    }
                )
            if not roles or (f["kind"] != "event" and "owner" not in roles):
                continue
            # Final occurrence audit checks the action/value AND its directed bindings.
            audit_images = []
            role_regions = {role: grounded[f"q{n}_{role}"] for role in roles}
            for idx in dict.fromkeys(f["frames"]):
                encoded, _ = jpeg(media[idx], 1024)
                audit_images.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Evidence frame {idx}, {media[idx].seconds:.2f}s",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded},
                        },
                    ]
                )
            claim = {
                "kind": f["kind"],
                "predicate": f["predicate"],
                "value": f["value"],
                "roles": role_regions,
                "unresolved_slots": missing,
            }
            audit_images.append(
                {
                    "type": "text",
                    "text": "Audit this proposed fact ONLY against these evidence images. Does the predicate/action "
                    "actually occur here, with the stated role direction? A visible result, inscription or damaged "
                    "object does not establish an action that produced it. Verify literal text values from pixels. "
                    "Check the supplied role regions, not just their descriptions. A transitive action with an "
                    "unseen actor or recipient must list that role as missing. Do not accept an action from another "
                    'part of the video. Return JSON {"verdict":"supported|contradicted|insufficient",'
                    '"reason":"visual reason","missing_roles":[],"supported_frames":[]}. '
                    "Return supported_frames containing ONLY supplied frames that support this same subject and "
                    "claim. For properties and OCR each retained frame must itself show the exact value; exclude "
                    "unfinished writing and frames of other people. For events retain the before/during/after "
                    "frames needed to demonstrate this occurrence, excluding unrelated shots. Empty means no support. Claim: "
                    + json.dumps(claim),
                }
            )
            try:
                judgment = self.call(
                    audit_images,
                    out=out,
                    deadline=deadline,
                    task="joint_fact_audit",
                    tokens=800,
                    schema=fact_audit_schema(f["frames"]),
                )
            except Exception as e:
                judgment = {"verdict": "insufficient", "reason": str(e), "missing_roles": []}
            fact_audits.append({"fact_index": n, "judgment": judgment})
            atomic_json(out / "fact_audits.json", fact_audits)
            if judgment.get("verdict") != "supported" or set(
                judgment.get("missing_roles", [])
            ) & set(roles):
                continue
            supported = judgment.get("supported_frames", [])
            if not supported or any(
                type(idx) is not int or idx not in f["frames"] for idx in supported
            ):
                continue
            evidence["predicate"] = [str(idx) for idx in dict.fromkeys(supported)]
            # A removed evidence frame cannot remain an accepted role anchor.
            for role in list(roles):
                if not set(evidence[role]) <= set(evidence["predicate"]):
                    del roles[role]
                    del evidence[role]
                    missing.append(role)
            if not roles or (f["kind"] != "event" and "owner" not in roles):
                continue
            missing.extend(judgment.get("missing_roles", []))
            if f["kind"] != "event":
                evidence["value"] = evidence["predicate"]
            raw["facts"].append(
                {
                    "fact_id": f"verified:{n}",
                    "kind": f["kind"],
                    "predicate": f["predicate"][:64],
                    "roles": roles,
                    "value": None if f["kind"] == "event" else f["value"][:256],
                    "evidence_by_slot": evidence,
                    "joint_evidence": evidence["predicate"],
                    "observed_media_ids": evidence["predicate"],
                    "unresolved_slots": list(dict.fromkeys(missing)),
                }
            )
        used = {i for f in raw["facts"] for i in f["roles"].values()}
        raw["instances"] = [i for i in raw["instances"] if i["instance_id"] in used]
        if (
            rejected
            or reviewed.get("rejected")
            or reviewed.get("gaps")
            or any(a["judgment"].get("verdict") != "supported" for a in fact_audits)
        ):
            raw["coverage_gaps"] = [
                {"reason": "verification_rejected_or_unresolved_candidates", "media_ids": []}
            ]
        aliases = {str(k): m.media_id for k, m in media.items()}
        validate(raw, joint_schema(aliases, aliases, max_instances=80, max_facts=12))
        result = materialize(raw, replace(clip, clip_id=clip.clip_id + ":verified-v7"), aliases, {})
        result.update(
            architecture="rrt-verified-perception-v7",
            verification="model_review_pending_independent_acceptance",
            source_observation_id=obs["observation_id"],
            rejected_records=rejected,
            temporal_review=reviewed,
            fact_audits=fact_audits,
            semantic_support_audited=False,
            reference_evidence=[],
            context_used_for_proposals=[m.media_id for m in context],
        )
        atomic_json(out / "result.json", result)
        atomic_json(
            out / "DONE.json",
            {
                "facts": len(raw["facts"]),
                "grounded_roles": len(grounded),
                "rejected_roles": len(rejected),
                "semantic_acceptance": None,
            },
        )
        return result


def subject_audit_matches(subject, audit):
    """Grounding must identify the requested participant, not just any person."""
    if audit.get("verdict") != "supported" or audit.get("matches_requested_subject") is not True:
        return False
    description = subject["description"].casefold()
    if subject["kind"] == "person":
        if re.search(r"\b(baby|infant|child|toddler)\b", description):
            return audit.get("observed_age_group") == "child"
        if re.search(r"\b(man|woman|adult)\b", description):
            return audit.get("observed_age_group") == "adult"
    return True
