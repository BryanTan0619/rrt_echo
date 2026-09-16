"""Add visually rechecked anchors for tiny person sightings; preserve original records."""

import base64
import copy
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ..echo_perception.types import Deadline, atomic_json
from ..echo_perception.wire import enum, obj, string
from .verified import VerifiedPerceptionClient, usable_box


def enrich_tiny_anchors(results, *, out, model, base_url, deadline=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    deadline = deadline or Deadline(1800)
    updated = copy.deepcopy(results)
    changes = []
    client = VerifiedPerceptionClient(
        model=model, base_url=base_url, request_timeout=300, max_tokens=700
    )
    for result in updated:
        o = result["observation"]
        media = {m["media_id"]: m for m in o["media"]}
        groups = {}
        for inst in o["instances"]:
            if inst["kind"] != "person" or not inst["regions"]:
                continue
            if (
                max(
                    (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])
                    for r in inst["regions"]
                )
                >= 0.045
            ):
                continue
            key = json.dumps([inst["description"], inst["regions"]], sort_keys=True)
            groups.setdefault(key, []).append(inst)
        for members in groups.values():
            ids = {i["instance_id"] for i in members}
            mids = {
                m
                for f in o["facts"]
                if ids.intersection(f["roles"].values())
                for m in f["joint_evidence"]
            }
            ordered = sorted(
                mids,
                key=lambda m: media[m]["pts"] * media[m]["time_base"][0] / media[m]["time_base"][1],
            )
            # Search later witnesses first when the initial anchor was tiny/occluded.
            # This is candidate scheduling only; the independent box audit still decides.
            ordered.reverse()
            if len(ordered) < 2:
                continue
            if len(ordered) > 8:
                ordered = [ordered[round(n * (len(ordered) - 1) / 7)] for n in range(8)]
            content = []
            for idx, mid in enumerate(ordered):
                with Image.open(media[mid]["uri"]) as raw:
                    im = raw.convert("RGB")
                    im.thumbnail((1024, 1024))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=90)
                content.extend(
                    [
                        {"type": "text", "text": f"Frame {idx}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(buf.getvalue()).decode()
                            },
                        },
                    ]
                )
            description = members[0]["description"]
            content.append(
                {
                    "type": "text",
                    "text": "Locate this local subject in its OWN occurrence evidence. Choose the clearest visible body/face, not a clothing sliver or a nearby carrier. Do not infer global identity. If no recognizable subject is visible return visible=false. Return frame index and tight integer normalized 0..1000 xyxy box across full image. Subject: "
                    + description,
                }
            )
            schema = obj(
                {
                    "visible": {"type": "boolean"},
                    "frame": {"type": "integer", "enum": list(range(len(ordered)))},
                    "box": {
                        "type": ["array", "null"],
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                }
            )
            try:
                answer = client.call(
                    content,
                    out=out / "calls",
                    deadline=deadline,
                    task="clear_anchor_proposal",
                    schema=schema,
                )
                if not usable_box(answer):
                    continue
                mid = ordered[answer["frame"]]
                box = answer["box"]
                with Image.open(media[mid]["uri"]) as raw:
                    im = raw.convert("RGB")
                    w, h = im.size
                    d = ImageDraw.Draw(im)
                    d.rectangle(
                        [
                            box[0] * w / 1000,
                            box[1] * h / 1000,
                            box[2] * w / 1000,
                            box[3] * h / 1000,
                        ],
                        outline="yellow",
                        width=5,
                    )
                    im.thumbnail((1024, 1024))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=90)
                audit = client.call(
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
                            "text": "Does the marked box tightly identify this requested subject, with enough visible body/face to support identity comparison? Reject a different person, body part only, clothing sliver or background. Subject: "
                            + description,
                        },
                    ],
                    out=out / "calls",
                    deadline=deadline,
                    task="clear_anchor_audit",
                    schema=obj(
                        {
                            "verdict": enum(["supported", "contradicted", "unresolved"]),
                            "basis": string(160),
                        }
                    ),
                )
                if audit["verdict"] != "supported":
                    continue
                anchor = {"media_id": mid, "box": [v / 1000 for v in box]}
                for inst in members:
                    if anchor not in inst["regions"]:
                        inst["regions"].append(copy.deepcopy(anchor))
                record = {
                    "instances": sorted(ids),
                    "added_anchor": anchor,
                    "basis": audit["basis"],
                    "status": "model_checked_anchor_pending_visual_review",
                }
                changes.append(record)
                im.save(out / f"anchor_{len(changes):03d}.jpg")
            except Exception as exc:
                changes.append({"instances": sorted(ids), "error": str(exc)})
            atomic_json(out / "anchor_changes.json", changes)
            atomic_json(out / "observations.json", updated)
    atomic_json(out / "anchor_changes.json", changes)
    atomic_json(out / "observations.json", updated)
    return updated
