"""Single joint call per clip; native vLLM video frame-sequence or explicit image ablation.
No automatic retry, prompt rewriting, recursive splitting, or identity commitment.
"""

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .prompts import JOINT_LOCAL, VERSION
from .types import Deadline, atomic_json, digest
from .wire import joint_schema, materialize, validate


def jpeg(media, max_edge, label=None):
    if digest(media.uri) != media.sha256:
        raise ValueError("media_bytes_changed")
    with Image.open(media.uri) as source:
        image = source.convert("RGB")
        original = image.size
        image.thumbnail((max_edge, max_edge))
        content_size = image.size
        if label:
            canvas = Image.new("RGB", (image.width, image.height + 32), (20, 20, 20))
            canvas.paste(image, (0, 32))
            ImageDraw.Draw(canvas).text(
                (4, 5), label, fill="white", font=ImageFont.load_default(size=18)
            )
            image = canvas
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
    data = buf.getvalue()
    return base64.b64encode(data).decode(), {
        "source_size": original,
        "sent_size": image.size,
        "content_box": [0, 32 if label else 0, content_size[0], image.height],
        "visible_label": label,
        "sent_sha256": hashlib.sha256(data).hexdigest(),
    }


def transport_schema(value):
    """Grammar compatibility only; full schema is still validated after generation."""
    if isinstance(value, dict):
        return {k: transport_schema(v) for k, v in value.items() if k != "uniqueItems"}
    if isinstance(value, list):
        return [transport_schema(v) for v in value]
    return value


class VisionClient:
    def __init__(
        self,
        *,
        model,
        base_url=None,
        mode="video",
        max_tokens=3072,
        image_max_edge=768,
        max_instances=16,
        max_facts=24,
        max_reference_views=6,
        request_timeout=120,
        schema_mode="json_schema",
        video_transport="standard",
        frame_labels=False,
    ):
        if mode not in {"video", "images"}:
            raise ValueError("mode must be video or images")
        if schema_mode not in {"json_schema", "prompt"}:
            raise ValueError("unknown_schema_mode")
        self.model = model
        self.base_url = base_url or os.environ.get("RRT_VLM_BASE_URL", "http://127.0.0.1:8000/v1")
        if video_transport not in {"standard", "rrt-v1"}:
            raise ValueError("unknown_video_transport")
        self.video_transport = video_transport
        self.frame_labels = frame_labels
        self.mode = mode
        self.max_tokens = max_tokens
        self.image_max_edge = image_max_edge
        self.max_instances = max_instances
        self.max_facts = max_facts
        self.max_reference_views = max_reference_views
        self.request_timeout = request_timeout
        self.schema_mode = schema_mode

    def build_request(self, clip, *, references=(), detections=()):
        if not clip.target_ids:
            raise ValueError("empty_target")
        if self.mode == "video" and not clip.constant_frame_rate:
            raise ValueError("VFR_source: use explicit images mode; do not invent CFR timestamps")
        if sum(len(r.media) for r in references) > self.max_reference_views:
            raise ValueError("reference_budget_exceeded")
        if len({r.ref_id for r in references}) != len(references):
            raise ValueError("duplicate_reference_id")
        aliases = {m.media_id: f"f{k}" for k, m in enumerate(clip.media)}
        alias_to_source = {v: k for k, v in aliases.items()}
        ref_index = {}
        presentations = []
        frames = []
        table = []
        content = []
        for m in clip.media:
            alias = aliases[m.media_id]
            scope = "TARGET" if m.media_id in clip.target_ids else "CONTEXT"
            encoded, info = jpeg(
                m,
                self.image_max_edge,
                f"{scope} {alias} | {m.seconds:.2f}s" if self.frame_labels else None,
            )
            frames.append(encoded)
            table.append(
                {
                    "media_id": alias,
                    "scope": scope,
                    "time_seconds": m.seconds,
                    "source_frame_index": m.index,
                }
            )
            presentations.append(
                {
                    **asdict(m),
                    "media_id": alias,
                    "source_media_id": m.media_id,
                    "scope": scope,
                    **info,
                }
            )
            if self.mode == "images":
                content.extend(
                    [
                        {"type": "text", "text": json.dumps(table[-1])},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded},
                        },
                    ]
                )
        if self.mode == "video":
            url = "data:video/jpeg;base64," + ",".join(frames)
            if self.video_transport == "rrt-v1":
                from .video_transport import pack

                url = pack(
                    frames,
                    {
                        "fps": clip.fps,
                        "frames_indices": [m.index for m in clip.media],
                        "total_num_frames": clip.total_frames,
                        "duration": clip.duration,
                        "do_sample_frames": False,
                    },
                )
            content.append({"type": "video_url", "video_url": {"url": url}})
        for j, ref in enumerate(references):
            if (
                not ref.ref_id.startswith("ref:")
                or not ref.media
                or len(ref.media) != len(ref.sources)
            ):
                raise ValueError("invalid_reference_bundle")
            if ref.kind not in {"person", "object"}:
                raise ValueError("reference_must_be_entity")
            mids = []
            for k, (m, source) in enumerate(zip(ref.media, ref.sources)):
                alias = f"r{j}_{k}"
                mids.append(alias)
                alias_to_source[alias] = m.media_id
                encoded, info = jpeg(m, self.image_max_edge)
                label = {
                    "media_id": alias,
                    "scope": "REFERENCE",
                    "ref_id": ref.ref_id,
                    "kind": ref.kind,
                    "time_seconds": m.seconds,
                }
                label["appearance_cues"] = [
                    {
                        "property": c["property"],
                        "value": c["value"],
                        "frame": alias,
                        "status": "reference observation hypothesis; compare pixels",
                        "continuity_asserted": False,
                    }
                    for c in source.get("appearance_cues", [])
                ]
                content.extend(
                    [
                        {"type": "text", "text": json.dumps(label)},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded},
                        },
                    ]
                )
                presentations.append({**asdict(m), **label, "source_region": source, **info})
            ref_index[ref.ref_id] = {
                "instance_id": ref.instance_id,
                "kind": ref.kind,
                "media_ids": mids,
            }
        target = [aliases[m.media_id] for m in clip.media if m.media_id in clip.target_ids]
        schema = joint_schema(
            target, alias_to_source, ref_index, self.max_instances, self.max_facts
        )
        proposals = []
        for d in detections:
            if d["media_id"] not in aliases:
                raise ValueError("detection_outside_clip")
            a, b, c, e = d["box"]
            if not 0 <= a < c <= 1 or not 0 <= b < e <= 1:
                raise ValueError("invalid_detection_box")
            geom = next(x for x in presentations if x.get("source_media_id") == d["media_id"])
            w, h = geom["sent_size"]
            x0, y0, x1, y1 = geom["content_box"]
            box = [
                (a * (x1 - x0) + x0) / w,
                (b * (y1 - y0) + y0) / h,
                (c * (x1 - x0) + x0) / w,
                (e * (y1 - y0) + y0) / h,
            ]
            proposals.append(
                {
                    "media_id": aliases[d["media_id"]],
                    "box": [round(x * 1000) for x in box],
                    "track_ref": d.get("track_ref"),
                    "status": "proposal",
                }
            )
        prompt = (
            JOINT_LOCAL
            + "\nINPUT_MANIFEST\n"
            + json.dumps(
                {
                    "frames": table,
                    "target_interval": [clip.target_start, clip.target_end],
                    "video_timestamp_offset_seconds": clip.start_seconds,
                    "references": {
                        k: {"kind": v["kind"], "media_ids": v["media_ids"]}
                        for k, v in ref_index.items()
                    },
                    "region_proposals": proposals,
                },
                separators=(",", ":"),
            )
        )
        if self.schema_mode == "prompt":
            prompt += "\nOUTPUT_SCHEMA\n" + json.dumps(schema, separators=(",", ":"))
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self.schema_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "joint_observation", "schema": transport_schema(schema)},
            }
        if self.mode == "video":
            body["media_io_kwargs"] = {
                "video": {
                    "fps": clip.fps,
                    "frames_indices": [m.index for m in clip.media],
                    "total_num_frames": clip.total_frames,
                    "duration": clip.duration,
                    "do_sample_frames": False,
                }
            }
            body["mm_processor_kwargs"] = {"do_sample_frames": False}
        if self.mode == "video" and self.video_transport == "rrt-v1":
            body.pop("media_io_kwargs", None)
        # Audit never includes credentials or bulky base64 payloads.
        audit = {
            "prompt_version": VERSION,
            "prompt": prompt,
            "schema": schema,
            "mode": self.mode,
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "presentations": presentations,
            "reference_endpoints": ref_index,
            "source_media_aliases": alias_to_source,
            "sampling": clip.sampling,
            "video_metadata": {
                "fps": clip.fps,
                "frames_indices": [m.index for m in clip.media],
                "total_num_frames": clip.total_frames,
                "duration": clip.duration,
                "do_sample_frames": False,
            },
            "video_transport": self.video_transport,
            "transport_schema_omissions": ["uniqueItems"]
            if self.schema_mode == "json_schema"
            else [],
            "actual_server_sampling": "unverified; inspect deployed processor before quality acceptance",
        }
        return body, audit

    def decode_output(self, text, audit):
        return json.loads(text)

    def observe(self, clip, *, out, deadline=None, references=(), detections=(), sender=None):
        deadline = deadline or Deadline()
        deadline.require()
        body, audit = self.build_request(clip, references=references, detections=detections)
        key = hashlib.sha256(
            json.dumps(
                {
                    "result_schema_version": "reference-media-v2",
                    "body": body,
                    "clip_id": clip.clip_id,
                    "video_id": clip.video_id,
                    "endpoints": audit["reference_endpoints"],
                    "sources": audit["source_media_aliases"],
                    "base_url": self.base_url,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        run = Path(out) / key
        run.mkdir(parents=True, exist_ok=True)
        result_path = run / "result.json"
        self.last_cache_hit = result_path.exists()
        if self.last_cache_hit:
            return json.loads(result_path.read_text())
        # Do not silently rerun failed/unresolved work with identical evidence.
        if (run / "failure.json").exists():
            raise RuntimeError(
                "cached_failure: inspect audit; change evidence/config or explicitly use a new output directory"
            )
        atomic_json(run / "request.json", audit)
        start = time.monotonic()
        response = None
        deadline.require()
        try:
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
                raise ValueError(
                    "truncated_output: retained raw response; no recursive re-extraction"
                )
            raw = self.decode_output(choice["message"]["content"], audit)
            atomic_json(run / "normalized.json", raw)
            validate(raw, audit["schema"], audit["reference_endpoints"])
            result = materialize(
                raw, clip, audit["source_media_aliases"], audit["reference_endpoints"]
            )
            result["reference_evidence"] = [
                {
                    **x,
                    "presentation_id": x["media_id"],
                    "media_id": audit["source_media_aliases"][x["media_id"]],
                }
                for x in audit["presentations"]
                if x["scope"] == "REFERENCE"
            ]
            if len(raw["instances"]) >= self.max_instances or len(raw["facts"]) >= self.max_facts:
                result["coverage_gaps"].append(
                    {
                        "reason": "transport_limit_reached; completeness_not_certified",
                        "media_ids": [],
                    }
                )
            if clip.sampling.get("target_budget_limited"):
                result["coverage_gaps"].append(
                    {"reason": "target_sampling_budget_limited", "media_ids": []}
                )
            result["request_key"] = key
            result["format_normalizations"] = audit.get("format_normalizations", [])
            result["rejected_records"] = audit.get("rejected_records", [])
            result["structural_status"] = (
                "partial" if result["rejected_records"] or result["coverage_gaps"] else "valid"
            )
            atomic_json(
                run / "normalization_audit.json",
                {
                    "format_normalizations": result["format_normalizations"],
                    "rejected_records": result["rejected_records"],
                },
            )
            atomic_json(result_path, result)
            return json.loads(result_path.read_text())
        except Exception as e:
            failure = {"error": str(e), "kind": type(e).__name__, "clip_id": clip.clip_id}
            if isinstance(e, urllib.error.HTTPError):
                failure["http_status"] = e.code
                failure["server_detail"] = e.read(16384).decode("utf-8", errors="replace")
            atomic_json(run / "failure.json", failure)
            raise
        finally:
            atomic_json(
                run / "timing.json",
                {
                    "elapsed_seconds": time.monotonic() - start,
                    "usage": (response or {}).get("usage"),
                    "server_sampling_verified": False,
                },
            )
