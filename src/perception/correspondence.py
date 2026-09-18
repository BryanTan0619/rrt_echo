"""Compare frozen local endpoints. Context cannot create or overwrite observations."""

import base64
import hashlib
import io
import json
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from PIL import Image, ImageDraw

from .types import Deadline, atomic_json
from .vlm import jpeg, transport_schema
from .wire import arr, enum, obj, string

PROMPT = """Compare only the listed endpoint pairs using the supplied ordered frames and regions.
Determine visual continuity or region ownership, not narrative plausibility.
A name, similar clothing, shared object or neighboring shot alone is insufficient.
Use same/different/unresolved for same_identity and belongs/not_belongs/unresolved for part_of.
Different means the marked regions depict distinct individuals; simultaneous distinct people are positive evidence for different. Do not answer unresolved merely because you cannot establish same.
Unresolved is appropriate when the visual evidence cannot distinguish these alternatives.
Cite a visible anchor for BOTH endpoints. Missing connecting views: unresolved.
Return only pair_id, verdict, evidence_ids and a short visual basis for each pair.
Do not return gap_reason; unresolved verdict already records a missing connection.
Evidence IDs form a set: if both endpoints occur in the same frame cite that frame ONCE.
part_of asks physical region ownership, NOT whether a body part and whole person are identical.
Same_identity: same visible person/object appearing twice may be same even within one frame when boxes overlap; distinct co-visible individuals are different.
Return correspondences only. Do not create people, events, properties or captions.
"""


def comparison_schema(pairs, media_ids):
    variants = []
    for p in pairs:
        resolved = (
            ["same", "different"]
            if p["relation"] == "same_identity"
            else ["belongs", "not_belongs"]
        )
        for verdicts, gaps in [
            (resolved, ["none"]),
            (
                ["unresolved"],
                ["not_visible", "missing_connection", "competing_candidates", "conflicting_views"],
            ),
        ]:
            variants.append(
                obj(
                    {
                        "pair_id": enum([p["pair_id"]]),
                        "verdict": enum(verdicts),
                        "evidence_ids": arr(enum(media_ids), 12),
                        "basis": string(240),
                        "gap_reason": enum(gaps),
                    }
                )
            )
    return obj({"correspondences": arr({"oneOf": variants}, len(pairs))})


def validate_correspondences(raw, pairs, instances, media_ids):
    Draft202012Validator(comparison_schema(pairs, media_ids)).validate(raw)
    by_id = {p["pair_id"]: p for p in pairs}
    seen = set()
    results = []
    for row in raw["correspondences"]:
        pid = row["pair_id"]
        if pid in seen:
            raise ValueError("duplicate_pair_result")
        seen.add(pid)
        p = by_id[pid]
        if row["verdict"] != "unresolved" and row["gap_reason"] != "none":
            raise ValueError("resolved_correspondence_has_unresolved_gap")
        if row["verdict"] != "unresolved":
            for endpoint in (p["left"], p["right"]):
                anchors = {r["media_id"] for r in instances[endpoint]["regions"]}
                if not anchors.intersection(row["evidence_ids"]):
                    raise ValueError("correspondence_missing_endpoint_evidence")
        results.append({**p, **row, "status": "proposed", "semantic_support_audited": False})
    for pid in by_id.keys() - seen:
        results.append(
            {
                **by_id[pid],
                "verdict": "unresolved",
                "evidence_ids": [],
                "basis": "No comparison returned",
                "gap_reason": "omitted_pair",
                "status": "proposed",
                "semantic_support_audited": False,
            }
        )
    return results


def validate_partial_correspondences(raw, pairs, instances, media_ids):
    """Retain independently valid rows, never coerce a wrong relation verdict."""
    rows, rejected = [], []
    for pair in pairs:
        candidates = [
            r for r in raw.get("correspondences", []) if r.get("pair_id") == pair["pair_id"]
        ]
        try:
            if len(candidates) > 1:
                raise ValueError("duplicate_pair_result")
            valid = validate_correspondences(
                {"correspondences": candidates}, [pair], instances, media_ids
            )
            rows.extend(valid)
        except (ValueError, ValidationError) as error:
            # JSON-schema and endpoint errors concern only this specified pair.
            rejected.append(
                {"pair_id": pair["pair_id"], "error": str(error)[:500], "rows": candidates}
            )
            rows.extend(
                validate_correspondences({"correspondences": []}, [pair], instances, media_ids)
            )
    return rows, rejected


class CorrespondenceClient:
    def __init__(self, *, model, base_url, max_tokens=1200, partial_rows=False):
        self.partial_rows = partial_rows
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens

    def compare(self, *, pairs, instances, media, out, deadline=None, sender=None):
        if not pairs:
            return []
        if len(pairs) > 8:
            raise ValueError("comparison_batch_exceeds_eight")
        if len({p["pair_id"] for p in pairs}) != len(pairs):
            raise ValueError("duplicate_pair_id")
        selected = {}
        for p in pairs:
            a, b = instances[p["left"]], instances[p["right"]]
            if a["instance_id"] == b["instance_id"]:
                raise ValueError("identical_endpoints")
            if p["relation"] == "same_identity":
                if a["kind"] == "region" or a["kind"] != b["kind"]:
                    raise ValueError("identity_kind_mismatch")
            elif p["relation"] == "part_of":
                if a["kind"] != "region" or b["kind"] not in {"person", "object"}:
                    raise ValueError("invalid_owner_endpoints")
            else:
                raise ValueError("unknown_relation")
            selected[a["instance_id"]] = a
            selected[b["instance_id"]] = b
        media_by_id = {m.media_id: m for m in media}
        if not all(r["media_id"] in media_by_id for i in selected.values() for r in i["regions"]):
            raise ValueError("endpoint_anchor_not_in_comparison")
        aliases = {m.media_id: f"f{n}" for n, m in enumerate(media)}
        originals = {v: k for k, v in aliases.items()}
        person_alias = {key: f"P{n}" for n, key in enumerate(selected)}
        content = []
        for m in sorted(media, key=lambda x: (x.seconds, x.media_id)):
            encoded, _ = jpeg(m, 768)
            im = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            draw = ImageDraw.Draw(im)
            for endpoint, inst in selected.items():
                for region in inst["regions"]:
                    if region["media_id"] != m.media_id:
                        continue
                    x0, y0, x1, y1 = region["box"]
                    box = (x0 * im.width, y0 * im.height, x1 * im.width, y1 * im.height)
                    draw.rectangle(box, outline="yellow", width=3)
                    draw.text(
                        (box[0] + 2, box[1] + 2),
                        person_alias[endpoint],
                        fill="red",
                        stroke_width=1,
                        stroke_fill="white",
                    )
            buf = io.BytesIO()
            im.save(buf, format="JPEG")
            encoded = base64.b64encode(buf.getvalue()).decode()
            content.extend(
                [
                    {"type": "text", "text": f"{aliases[m.media_id]} at {m.seconds:.3f}s"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + encoded},
                    },
                ]
            )
        # Neutral endpoint IDs and regions only: descriptions must not seed identity.
        endpoints = [
            {k: i[k] for k in ("instance_id", "kind", "regions")} for i in selected.values()
        ]
        endpoints = [
            {
                **i,
                "instance_id": person_alias[i["instance_id"]],
                "regions": [{**r, "media_id": aliases[r["media_id"]]} for r in i["regions"]],
            }
            for i in endpoints
        ]
        content.append(
            {
                "type": "text",
                "text": PROMPT
                + json.dumps(
                    {
                        "endpoints": endpoints,
                        "pairs": [
                            {
                                **p,
                                "left": person_alias[p["left"]],
                                "right": person_alias[p["right"]],
                            }
                            for p in pairs
                        ],
                    }
                ),
            }
        )
        # A compact transport schema avoids repeatedly expanding oneOf per pair.
        verdicts = set()
        for pair in pairs:
            verdicts.update(
                ["same", "different", "unresolved"]
                if pair["relation"] == "same_identity"
                else ["belongs", "not_belongs", "unresolved"]
            )
        schema = obj(
            {
                "correspondences": arr(
                    obj(
                        {
                            "pair_id": enum(p["pair_id"] for p in pairs),
                            "verdict": enum(sorted(verdicts)),
                            "evidence_ids": arr(enum(originals), 12),
                            "basis": string(240),
                        }
                    ),
                    len(pairs),
                )
            }
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "structured_outputs": {"disable_any_whitespace": True},
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "correspondence", "schema": transport_schema(schema)},
            },
        }
        key = hashlib.sha256(
            json.dumps(
                {
                    "body": body,
                    "endpoint_versions": selected,
                    "base_url": self.base_url,
                    "partial_rows": self.partial_rows,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        run = Path(out) / key
        run.mkdir(parents=True, exist_ok=True)
        if (run / "result.json").exists():
            return json.loads((run / "result.json").read_text())
        if (run / "failure.json").exists():
            raise RuntimeError("cached_correspondence_failure")
        atomic_json(
            run / "request.json",
            {
                "pairs": pairs,
                "endpoints": endpoints,
                "media": [asdict(m) for m in media],
                "prompt": content[-1]["text"],
            },
        )
        deadline = deadline or Deadline()
        deadline.require()
        start = time.monotonic()
        response = {}
        try:
            if sender:
                response = sender(body)
            else:
                req = urllib.request.Request(
                    self.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=min(300, deadline.remaining())) as f:
                    response = json.load(f)
            atomic_json(run / "response.json", response)
            choice = response["choices"][0]
            if choice["finish_reason"] == "length":
                raise ValueError("truncated_correspondence")
            raw = json.loads(choice["message"]["content"])
            # Repeating the same evidence ID adds no information; retain raw response above.
            for row in raw.get("correspondences", []):
                old_gap = row.pop("gap_reason", None)
                if old_gap not in {None, "none"} and row.get("verdict") != "unresolved":
                    row["verdict"] = "unresolved"
                if isinstance(row.get("evidence_ids"), list):
                    row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"]))
            Draft202012Validator(schema).validate(raw)
            for row in raw["correspondences"]:
                row["gap_reason"] = (
                    "missing_connection" if row["verdict"] == "unresolved" else "none"
                )
                row["evidence_ids"] = [originals[x] for x in row["evidence_ids"]]
            if self.partial_rows:
                results, rejected = validate_partial_correspondences(
                    raw, pairs, instances, list(media_by_id)
                )
                atomic_json(run / "rejected_rows.json", rejected)
            else:
                results = validate_correspondences(raw, pairs, instances, list(media_by_id))
            atomic_json(run / "result.json", results)
            return results
        except Exception as e:
            atomic_json(run / "failure.json", {"error": str(e)})
            raise
        finally:
            atomic_json(
                run / "timing.json",
                {"seconds": time.monotonic() - start, "usage": response.get("usage")},
            )
