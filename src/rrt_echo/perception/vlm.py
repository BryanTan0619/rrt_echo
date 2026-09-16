"""Minimal OpenAI-compatible transport; credentials come only from environment."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path

from ..runtime import Deadline
from ..schema import ObservationPacket
from .prompts import JOINT_LOCAL, LOCAL


class VisionClient:
    def __init__(
        self,
        *,
        base_url=None,
        model=None,
        max_tokens=4096,
        image_max_edge=1024,
        staged=False,
        joint=False,
        request_timeout=120,
    ):
        self.base_url = base_url or os.environ.get("RRT_VLM_BASE_URL", "http://127.0.0.1:8000/v1")
        self.model = model or os.environ.get("RRT_VLM_MODEL")
        if not self.model:
            raise ValueError("set RRT_VLM_MODEL or config.model")
        self.request_timeout = request_timeout
        self.staged = staged
        self.joint = joint
        self.image_max_edge = image_max_edge
        self.max_tokens = max_tokens
        self.calls: list[dict] = []

    def complete(self, prompt, media=(), *, deadline: Deadline, crops=(), output_schema=None):
        deadline.require()
        content = []
        presentations = []
        for m in media:
            path = Path(m.uri)
            import io

            from PIL import Image

            with Image.open(path) as source:
                source = source.convert("RGB")
                original_size = source.size
                source.thumbnail((self.image_max_edge, self.image_max_edge))
                buffer = io.BytesIO()
                source.save(buffer, format="JPEG", quality=95)
                frame_bytes = buffer.getvalue()
                presentations.append(
                    {
                        "media_id": m.media_id,
                        "source_size": original_size,
                        "sent_size": source.size,
                        "source_sha256": m.sha256,
                    }
                )
            content.append(
                {"type": "text", "text": f"MEDIA_ID={m.media_id}; time={m.seconds:.6f}s"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(frame_bytes).decode()
                    },
                }
            )
        if crops:
            import io

            from PIL import Image

            by_id = {m.media_id: m for m in media}
            for label, region in crops:
                source = by_id[region.media_id]
                with Image.open(source.uri) as image:
                    w, h = image.size
                    x1, y1, x2, y2 = region.box
                    image = image.crop(
                        (
                            int(x1 * w),
                            int(y1 * h),
                            max(int(x1 * w) + 1, int(x2 * w)),
                            max(int(y1 * h) + 1, int(y2 * h)),
                        )
                    ).convert("RGB")
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=90)
                content.append(
                    {
                        "type": "text",
                        "text": f"CROP_OF={label}; SOURCE_MEDIA_ID={source.media_id}; box={region.box}",
                    }
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(buffer.getvalue()).decode()
                        },
                    }
                )
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if output_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "local_observation", "schema": output_schema},
            }
            body["repetition_penalty"] = 1.05
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("RRT_VLM_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        start = time.monotonic()
        response = None
        try:
            with urllib.request.urlopen(
                request, timeout=min(self.request_timeout, deadline.remaining())
            ) as handle:
                response = json.load(handle)
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("truncated_model_output")
            text = choice["message"]["content"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("expected JSON object")
            return result
        finally:
            self.calls.append(
                {
                    "elapsed_seconds": time.monotonic() - start,
                    "usage": (response or {}).get("usage"),
                    "response": response,
                    "prompt": prompt,
                    "output_schema": output_schema,
                    "media_ids": [m.media_id for m in media],
                    "presentations": presentations,
                    "crop_regions": [
                        {"instance_id": label, **asdict(region)} for label, region in crops
                    ],
                }
            )

    def observe(self, observation_id, video_id, shot_id, media, *, deadline, detections=()):
        aliases = {m.media_id: f"f{k}" for k, m in enumerate(media)}
        sources = {value: key for key, value in aliases.items()}
        model_media = tuple(replace(m, media_id=aliases[m.media_id]) for m in media)
        detections = [{**d, "media_id": aliases[d["media_id"]]} for d in detections]
        prompt = LOCAL + "\nREGION_PROPOSALS:\n" + json.dumps(detections)
        repair_note = ""
        for attempt in range(2):
            try:
                if self.joint:
                    from ..schema import Region
                    from .wire import joint_schema

                    try:
                        raw = self.complete(
                            JOINT_LOCAL
                            + repair_note
                            + "\nREGION_PROPOSALS:\n"
                            + json.dumps(detections),
                            model_media,
                            deadline=deadline,
                            output_schema=joint_schema(sources),
                        )
                    finally:
                        if self.calls:
                            self.calls[-1].update(stage="local_joint", attempt=attempt)
                    for item in raw["instances"]:
                        for region in item["regions"]:
                            region["box"] = [float(x) / 1000 for x in region["box"]]
                            Region(region["media_id"], tuple(region["box"]))
                    if self.calls:
                        self.calls[-1]["output_limit_reached"] = (
                            len(raw["instances"]) >= 12 or len(raw["facts"]) >= 20
                        )
                elif self.staged:
                    from .wire import fact_schema, instance_schema

                    first = self.complete(
                        "Ground all visible people (including infants), objects and owner regions across these frames. Use neutral local IDs; one ID per visibly continuous individual, not per detection box. Distinct people must remain separate. Kind is person/object/region. Detector boxes are fallible proposals. Do not invent people from isolated body parts. Cite up to four actual frame regions per instance. Box format is [x_min,y_min,x_max,y_max] in INTEGER 0..1000 coordinates, with x_min < x_max and y_min < y_max. Example box [120,80,610,930]. Do not register generic background scenery. No events, story roles or global identity decisions in this stage. Return compact JSON.\n"
                        + json.dumps(detections),
                        model_media,
                        deadline=deadline,
                        output_schema=instance_schema(sources),
                    )
                    if self.calls:
                        self.calls[-1].update(
                            stage="local_instances",
                            source_media_aliases=sources,
                            observation_id=observation_id,
                        )
                    from ..schema import Region

                    for item in first["instances"]:
                        for region in item["regions"]:
                            region["box"] = [float(x) / 1000 for x in region["box"]]
                            Region(region["media_id"], tuple(region["box"]))
                    ids = [i["instance_id"] for i in first["instances"]]
                    if ids:
                        second = self.complete(
                            "Inspect these current frames and the LOCAL_INSTANCES below. Extract distinct joint events, states, attributes and text. All role values must use supplied instance IDs. Use directed agent/patient roles for events, owner plus value for state/attribute/text. Record each fact once, do not repeat paraphrases. evidence_by_slot MUST include a predicate key and a key for EVERY role, each containing supplied frame IDs. State/attribute/text also require a value key. Do not invent generic has_texture/has_shape facts; include only visible task-independent observations. Do not infer persistence, global names or unobserved transitions. Keep uncertain slots unresolved. Output compact JSON with at most 20 facts.\nLOCAL_INSTANCES:\n"
                            + json.dumps(first),
                            model_media,
                            deadline=deadline,
                            output_schema=fact_schema(sources, ids),
                        )
                    else:
                        second = {"facts": []}
                    raw = {**first, **second}
                    if self.calls:
                        self.calls[-1]["stage"] = "local_facts" if ids else "local_instances"
                        self.calls[-1]["output_limit_reached"] = (
                            len(ids) >= 12 or len(raw["facts"]) >= 20
                        )
                else:
                    raw = self.complete(prompt, model_media, deadline=deadline)
                # IDs are presentation aliases only. The stored packet retains canonical sources.
                raw = json.loads(json.dumps(raw))
                for instance in raw.get("instances", []):
                    for region in instance.get("regions", []):
                        region["media_id"] = sources.get(region["media_id"], region["media_id"])
                for fact in raw.get("facts", []):
                    fact["evidence_by_slot"] = {
                        key: [sources.get(mid, mid) for mid in ids]
                        for key, ids in fact.get("evidence_by_slot", {}).items()
                    }
                    for field in ("joint_evidence", "observed_media_ids", "boundary_evidence"):
                        if field in fact:
                            fact[field] = [sources.get(mid, mid) for mid in fact[field]]
                return self._observation(raw, observation_id, video_id, shot_id, media)
            except (ValueError, KeyError, TypeError) as exc:
                if attempt:
                    raise
                repair_note = (
                    "\nPrevious output failed validation: "
                    + str(exc)
                    + "\nCorrect the referenced field using the same visible evidence.\n"
                )
                prompt = (
                    LOCAL
                    + "\nPrior response violated the wire contract: "
                    + str(exc)
                    + "\nUse only the provided frame IDs. Return compact JSON; do not enumerate imagined frames. Reinspect the SAME frames.\nREGION_PROPOSALS:\n"
                    + json.dumps(detections)
                )
            finally:
                if self.calls:
                    self.calls[-1]["source_media_aliases"] = sources
                    self.calls[-1]["observation_id"] = observation_id

    def _observation(self, raw, observation_id, video_id, shot_id, media):
        if set(raw) - {"instances", "facts"}:
            raise ValueError("local model exceeded observation interface")
        if any(
            i.get("kind") not in {"person", "object", "region"} for i in raw.get("instances", [])
        ):
            raise ValueError("instance kind must be person, object or region")
        instances = raw.get("instances", [])
        mapping = {i["instance_id"]: f"{observation_id}:{i['instance_id']}" for i in instances}
        for i in instances:
            i["instance_id"] = mapping[i["instance_id"]]
        for f in raw.get("facts", []):
            f["fact_id"] = f"{observation_id}:{f['fact_id']}"
            f["roles"] = {role: mapping.get(ref, ref) for role, ref in f.get("roles", {}).items()}
            # Boundary certification is a separate operation, not VLM authority.
            if f.get("time_bounds") is not None:
                raise ValueError("local model cannot certify complete event boundaries")
        return ObservationPacket.from_dict(
            {
                **raw,
                "observation_id": observation_id,
                "video_id": video_id,
                "shot_id": str(shot_id),
                "media": [asdict(m) for m in media],
                "extractor": self.model,
                "family_id": observation_id,
            }
        )

    def compare(self, proposal_id, left, right, media, *, deadline, competitors=()):
        # Compare targets, not entire scenes or within-group continuity.
        from tempfile import TemporaryDirectory

        from .identity import compare_scoped, prepare_crop

        index = {m.media_id: m for m in media}
        with TemporaryDirectory(prefix="rrt_identity_") as directory:
            crops = {i.instance_id: prepare_crop(i, index, Path(directory)) for i in (left, right)}
            result = compare_scoped(
                self, proposal_id, left.instance_id, right.instance_id, crops, deadline
            )
        if self.calls:
            self.calls[-1]["candidate_competitors_not_shown"] = [i.instance_id for i in competitors]
        return result
