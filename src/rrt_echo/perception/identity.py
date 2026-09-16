"""Single-target visual identity comparison with auditable crop provenance."""

from __future__ import annotations

import hashlib
import json

from ..identity import IdentityProposal
from ..schema import Media

SCOPED_IDENTITY = (
    "Exactly TWO target images: image A and image B. First describe the primary human "
    "visible in A ONLY, then the primary human visible in B ONLY. Are A's primary human "
    "and B's primary human the SAME single person, or TWO DIFFERENT people? Do NOT "
    "compare continuity inside A or inside B. Compare A AGAINST B. Ignore shared scene "
    "context and partially visible background people. If either target is ambiguous "
    "or an unowned body part with no identifiable individual, use unresolved. Do not "
    "infer identity from clothing, story roles or shared objects alone. The conclusion "
    "must agree with the independently observed target descriptions."
)
SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "person_A": {"type": "string"},
        "person_B": {"type": "string"},
        "verdict": {"type": "string", "enum": ["different", "same", "unresolved"]},
        "visual_basis": {"type": "string"},
    },
    "required": ["person_A", "person_B", "verdict", "visual_basis"],
    "additionalProperties": False,
}


def prepare_crop(instance, media_index, folder):
    from PIL import Image

    region = instance.regions[0]
    source = media_index[region.media_id]
    name = hashlib.sha256(instance.instance_id.encode()).hexdigest()[:20] + ".jpg"
    path = folder / name
    with Image.open(source.uri) as image:
        w, h = image.size
        x1, y1, x2, y2 = region.box
        box = (
            int(x1 * w),
            int(y1 * h),
            max(int(x1 * w) + 1, int(x2 * w)),
            max(int(y1 * h) + 1, int(y2 * h)),
        )
        crop = image.crop(box).convert("RGB")
        crop.save(path, quality=95)
    return {
        "uri": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_media_id": source.media_id,
        "box": region.box,
        "pixel_box": box,
        "pts": source.pts,
        "time_base": source.time_base,
    }


def compare_scoped(client, proposal_id, left, right, crops, deadline):
    media = []
    mapping = {}
    for alias, ref in [("A", left), ("B", right)]:
        crop = crops[ref]
        media.append(
            Media(alias, crop["uri"], crop["pts"], tuple(crop["time_base"]), crop["sha256"])
        )
        mapping[alias] = {"instance_id": ref, **crop}
    try:
        result = client.complete(
            SCOPED_IDENTITY, tuple(media), deadline=deadline, output_schema=SCOPE_SCHEMA
        )
    finally:
        if client.calls:
            client.calls[-1].update(stage="scoped_identity", target_regions=mapping)
    if result["verdict"] not in {"same", "different", "unresolved"}:
        raise ValueError("invalid identity verdict")
    basis = json.dumps(
        {
            "person_A": result["person_A"],
            "person_B": result["person_B"],
            "visual_basis": result["visual_basis"],
        },
        ensure_ascii=False,
    )
    return IdentityProposal(
        proposal_id,
        left,
        right,
        result["verdict"],
        tuple(dict.fromkeys(crops[ref]["source_media_id"] for ref in [left, right])),
        basis,
    )
