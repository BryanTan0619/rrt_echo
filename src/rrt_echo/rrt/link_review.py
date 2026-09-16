"""Focused visual verification before a correspondence can enter the journal."""

from __future__ import annotations

import base64
import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from ..echo_perception.types import Deadline, atomic_json
from ..echo_perception.wire import enum, obj, string
from .association import inventory, seconds
from .verified import VerifiedPerceptionClient


def review_links(results, proposals, *, out, model, base_url, workers=2, deadline=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = deadline or Deadline(7200)
    instances, media, _ = inventory(results)
    client = VerifiedPerceptionClient(
        model=model, base_url=base_url, max_tokens=700, request_timeout=300
    )
    # Negative/unresolved identity proposals remain explicit; positive identity and
    # all ownership candidates receive a focused check, without the first verdict.
    work = [
        p
        for p in proposals
        if p["verdict"] != "unresolved"
        or p["relation"] == "part_of"
        or p.get("prior_model_verdict") in {"same", "belongs", "different", "not_belongs"}
    ]
    unchanged = [copy.deepcopy(p) for p in proposals if p not in work]
    schema = obj(
        {
            "source_category": enum(
                ["adult", "child", "body_part", "object", "background", "unclear"]
            ),
            "target_category": enum(
                ["adult", "child", "body_part", "object", "background", "unclear"]
            ),
            "verdict": enum(["supported", "contradicted", "unresolved"]),
            "source_matches_requested": {"type": "boolean"},
            "target_matches_requested": {"type": "boolean"},
            "basis": string(1200),
            "source_localized": {"type": "boolean"},
            "target_localized": {"type": "boolean"},
        }
    )

    def one(p):
        a, b = instances[p["source"]], instances[p["target"]]
        chosen = {}
        for label, inst in [("SOURCE", a), ("TARGET", b)]:
            # Prefer a substantial visible anchor, not simply the earliest one.
            ranked = sorted(
                inst["regions"],
                key=lambda r: (
                    -(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]),
                    seconds(media[r["media_id"]]),
                ),
            )
            chosen[label] = ranked[:2] if p["relation"] == "same_identity" else ranked[:1]
        content = []
        evidence = []
        for label, anchors in chosen.items():
            for anchor in anchors:
                m = media[anchor["media_id"]]
                evidence.append(m["media_id"])
                with Image.open(m["uri"]) as raw:
                    im = raw.convert("RGB")
                    w, h = im.size
                    x0, y0, x1, y1 = anchor["box"]
                    box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                    crop = im.crop(box)
                    draw = ImageDraw.Draw(im)
                    draw.rectangle(box, outline="yellow", width=5)
                    views = [(crop, "ONLY this cropped endpoint is being compared")]
                    if p["relation"] == "part_of":
                        views.insert(
                            0, (im, "yellow box marks endpoint; context shows physical attachment")
                        )
                    for image, view in views:
                        image.thumbnail((960, 960))
                        buf = io.BytesIO()
                        image.save(buf, format="JPEG", quality=90)
                        content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": f"{label} {view}; {m['media_id']}; {seconds(m):.2f}s",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,"
                                        + base64.b64encode(buf.getvalue()).decode()
                                    },
                                },
                            ]
                        )
        prompt = """Judge ONE physical relation between the marked SOURCE and TARGET endpoints.
Do not use a shared scene, shared role, general similar clothes, or narrative plausibility as proof of identity.
For same_identity, the claim is ONE AND THE SAME individual/object. A man and a woman in the same car are DIFFERENT, not "the same individuals". Judge faces, body continuity and distinguishing visual details.
For part_of, the claim is that SOURCE is a physical body/part region belonging to TARGET. This is not identity: an arm is not a whole person. Grass under a baby is not part of the baby. Co-occurrence alone is insufficient.
For part_of, different anatomy is EXPECTED: a stomach may belong to a baby whose face is in TARGET. Do NOT reject because stomach and face look different. Determine bodily ownership from visible connection or unmistakable body continuity.
Check localization separately. A clothing sliver, background, wrong body part or a composite box is insufficient as an identifiable person endpoint. A body part is not a whole-person identity anchor unless connected to a clearly identifiable body in these views.
Return supported only with direct visual evidence for both endpoints and their relation. Otherwise contradicted (clear counterevidence) or unresolved (not enough evidence). Do not speculate. First classify the visible endpoint in each crop (adult, child, body_part, object, background, unclear), then choose the verdict and give ONE short sentence of visual evidence (under 180 characters). Keep the explanation consistent with the verdict. Classify the marked endpoint, not everyone in the surrounding scene.
"""
        prompt += "\nCheck source_matches_requested and target_matches_requested against the requested subject descriptions. Descriptions specify what the endpoint was meant to localize; they are NEVER evidence that two endpoints share identity. A baby request with an adult crop is a localization failure. Replacement parts, twins, identical uniforms, or matching products are not established as the same physical entity by appearance alone. Text overlays about a person are not physical part_of relations.\n"
        claim = {
            "relation": p["relation"],
            "source_kind": a["kind"],
            "target_kind": b["kind"],
            "requested_source": a.get("description", ""),
            "requested_target": b.get("description", ""),
        }

        if p["relation"] == "part_of":
            # Region category matters (grass vs skin); a description is a request,
            # never ownership evidence. No target description/global story is given.
            claim["requested_source_region"] = a.get("description", "")
        content.append({"type": "text", "text": prompt + json.dumps(claim)})
        try:
            answer = client.call(
                content,
                out=out / "calls",
                deadline=deadline,
                task="focused_link_review",
                schema=schema,
            )
            verdict = answer["verdict"]
            verdict = endpoint_verdict(p["relation"], a["kind"], b["kind"], answer)
            result = {
                **copy.deepcopy(p),
                "verdict": verdict,
                "evidence_ids": list(dict.fromkeys(evidence)),
                "basis": answer["basis"],
                "focused_review": answer,
                "previous_verdict": p["verdict"],
                "status": "proposed",
                "semantic_support_audited": False,
            }
            return result
        except Exception as exc:
            return {
                **copy.deepcopy(p),
                "verdict": "unresolved",
                "basis": "Focused review failed: " + str(exc)[:150],
                "review_error": str(exc),
                "previous_verdict": p["verdict"],
                "status": "proposed",
            }

    reviewed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, work):
            reviewed.append(row)
            atomic_json(out / "reviewed_links.json", reviewed + unchanged)
            atomic_json(out / "progress.json", {"reviewed": len(reviewed), "total": len(work)})
    return reviewed + unchanged


def endpoint_verdict(relation, source_kind, target_kind, answer):
    """Endpoint evidence must be usable before either a positive or negative link."""
    a, b = answer["source_category"], answer["target_category"]
    if (
        answer.get("source_matches_requested") is False
        or answer.get("target_matches_requested") is False
    ):
        return "unresolved"
    if not answer["source_localized"] or not answer["target_localized"]:
        return "unresolved"
    if "unclear" in (a, b):
        return "unresolved"
    if target_kind == "person" and b not in {"adult", "child"}:
        return "unresolved"
    if relation == "same_identity":
        if source_kind == "person" and a not in {"adult", "child"}:
            return "unresolved"
        if target_kind == "person" and b not in {"adult", "child"}:
            return "unresolved"
        if {a, b} == {"adult", "child"}:
            return "contradicted"
    if relation == "part_of" and a == "body_part" and b in {"object", "background"}:
        return "unresolved"
    if relation == "part_of" and a == "background":
        return "contradicted"
    return answer["verdict"]
