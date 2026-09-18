"""A: occurrence-local roles refer to independently declared spatial observations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from perception.types import atomic_json
from perception.vlm import VisionClient, transport_schema
from perception.wire import arr, enum, obj, string, validate

PROMPT = """Observe the TARGET video interval as a sequence of distinct occurrences.
CONTEXT helps motion and ownership; REFERENCE helps identity only. Neither may
supply an action absent from TARGET. Ignore cinematic expectations and narrative labels.
First declare the visible people, objects, and isolated body regions. Give each
one a stable local instance ID throughout this interval. A close-up does not make
a new person. A hand/arm with uncertain owner is kind=region, NOT a new person.
Then declare spatial observations: region ID, instance ID, actual frame and tight
box. One region depicts one subject, not a carrier plus passenger or person plus
held object. Do not copy a person's box to the object being manipulated.
Each EVENT describes ONE occurrence and cites its own actual TARGET evidence.
Use a short neutral predicate (e.g. hold, bite, write), with directed role labels
(agent/patient, holder/held, writer/instrument/surface). Roles reference region IDs,
not just instance IDs. Identify the actual contact object; mere co-visibility is
not interaction. Register necessary tools and acted-on objects if visible; otherwise
leave their role in unresolved_slots. Do not substitute a nearby person for them.
For writing, writer, instrument and surface are distinct semantic roles. A body
surface is a region instance with a separate part_of link to its owner if visible.
Properties describe owned appearance/state; text contains literal readable characters
on its own surface. For each text property set carrier: inscribed for characters
physically on an object surface (dial, label, sign), screen for characters shown by
an electronic screen or instrument display (readings, step numbers), overlay for
captions/subtitles/narration added over the frame rather than present in the scene.
Use unknown when uncertain; non-text properties use carrier=unknown. Do not infer
death, infection, cause, kinship, reading/comprehension,
or unchanged state during gaps from appearance alone. Names are not identity proof.
For links: continues=short visual continuity, part_of=region ownership (never holding),
same_identity=cross-segment person/object identity. Compare supplied references directly,
cite visible frames for BOTH endpoints, and use unresolved for uncertain matches.
Instance descriptions are visible appearance, not stories. Carry discriminative
attributes for every instance: object dimensions are color/shape/material/size/state
(e.g. shape=umbrella, color=brown, state=raw/collapsing); person dimensions are
attire/color (e.g. attire=leather jacket, color=green tracksuit). A generic noun
alone (food/cookie) or a bare person is not enough. Give at most a few useful
anchors per instance, spread across its appearances; do not duplicate an instance when
anchor capacity is exhausted. Empty lists are valid. Plan the region list together with events: each role's region frame MUST occur in that event's frames. Create a separate region observation at another frame when needed, reusing the same instance ID. Cite 1-4 decisive frames, not a full frame enumeration. Avoid generic look/sit/stand repetitions; prioritize distinctive actions and observable changes. Stop after the actual events; maxima are not quotas. A writer is a PERSON; the arm being written on is the SURFACE, not the writer. Locate the writer at that action frame (or leave writer unresolved). A previously written inscription is a text observation, never evidence of an ongoing writing action. Link isolated body regions to their person with part_of, using the region INSTANCE ID as source and person INSTANCE ID as target and frames for both. Output compact JSON only.
"""


def event_schema(target, all_frames, refs, max_instances=20, max_events=8):
    ids = [f"i{i}" for i in range(max_instances)]
    rids = [f"r{i}" for i in range(128)]
    frames = arr(enum(target), 12, 1)
    inst = obj(
        {
            "id": enum(ids),
            "kind": enum(["person", "object", "region"]),
            "description": string(160),
        }
    )
    # Discriminative attributes: closed dimensions, open values.
    # Optional at the schema level; the prompt requires discriminative detail so
    # a generic noun (food/cookie) or bare person can still be distinguished.
    #   color/shape/material/size -> object appearance
    #   attire -> what a person wears (jacket, tracksuit, ...)
    #   state  -> object/person condition (raw, dry, smooth, collapsing, ...)
    inst["properties"]["attributes"] = arr(
        obj(
            {
                "dimension": enum(
                    ["color", "shape", "material", "size", "attire", "state"]
                ),
                "value": string(40),
            }
        ),
        6,
        0,
    )
    region = obj(
        {
            "id": enum(rids),
            "instance": enum(ids),
            "frame": enum(target),
            "box": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
        }
    )
    role = obj({"role": string(32), "region": enum(rids)})
    event = obj(
        {
            "predicate": string(48),
            "frames": frames,
            "roles": arr(role, 8, 1),
            "unresolved_slots": arr(string(32), 8),
        }
    )
    prop = obj(
        {
            "kind": enum(["state", "attribute", "text"]),
            "predicate": string(48),
            "value": string(180),
            "owner_region": enum(rids),
            "carrier": enum(["inscribed", "screen", "overlay", "unknown"]),
            "frames": frames,
        }
    )
    link = obj(
        {
            "source": enum(ids),
            "target": enum([*ids, *refs]),
            "relation": enum(["continues", "part_of", "same_identity"]),
            "verdict": enum(["supported", "contradicted", "unresolved"]),
            "evidence_ids": arr(enum(all_frames), 8, 1),
            "basis": string(180),
        }
    )
    return obj(
        {
            "instances": arr(inst, max_instances),
            "regions": arr(region, 128),
            "events": arr(event, max_events),
            "properties": arr(prop, 8),
            "links": arr(link, 16),
            "gaps": arr(string(160), 12),
        }
    )


def expand_events(raw, audit):
    Draft202012Validator(transport_schema(audit["wire_schema"])).validate(raw)
    raw = copy.deepcopy(raw)
    for record in [*raw["events"], *raw["properties"]]:
        record["frames"] = list(dict.fromkeys(record["frames"]))
    instances = {i["id"]: i for i in raw["instances"]}
    regions = {r["id"]: r for r in raw["regions"]}
    if len(instances) != len(raw["instances"]) or len(regions) != len(raw["regions"]):
        raise ValueError("duplicate_instance_or_region_id")
    geometry = {p["media_id"]: p for p in audit["presentations"]}
    anchors = {i: [] for i in instances}
    for r in regions.values():
        if r["instance"] not in instances:
            raise ValueError("undeclared_region_instance")
        p = geometry[r["frame"]]
        w, h = p["sent_size"]
        x0, y0, x1, y1 = p["content_box"]
        a, b, c, d = r["box"]
        a = max(x0, a * w / 1000)
        b = max(y0, b * h / 1000)
        c = min(x1, c * w / 1000)
        d = min(y1, d * h / 1000)
        if a >= c or b >= d:
            raise ValueError("invalid_region_box")
        box = [
            round(1000 * (a - x0) / (x1 - x0)),
            round(1000 * (b - y0) / (y1 - y0)),
            round(1000 * (c - x0) / (x1 - x0)),
            round(1000 * (d - y0) / (y1 - y0)),
        ]
        anchor = {"media_id": r["frame"], "box": box}
        if anchor not in anchors[r["instance"]]:
            anchors[r["instance"]].append(anchor)
    local = [
        {
            "instance_id": key,
            "kind": v["kind"],
            "description": v["description"],
            "attributes": v.get("attributes", []),
            "regions": anchors[key],
        }
        for key, v in instances.items()
        if anchors[key]
    ]
    base = {"instances": local, "facts": [], "links": [], "coverage_gaps": []}
    validate(base, audit["schema"], audit["reference_endpoints"])
    facts = []
    bindings = {}
    rejected = []
    seen_events = set()
    for n, event in enumerate(raw["events"]):
        key = json.dumps(event, sort_keys=True)
        if key in seen_events:
            rejected.append({"record": event, "reason": "exact_duplicate_event"})
            continue
        seen_events.add(key)
        try:
            role_names = [r["role"] for r in event["roles"]]
            if len(set(role_names)) != len(role_names):
                raise ValueError("duplicate_role")
            selected = {r["role"]: regions[r["region"]] for r in event["roles"]}
            if any(r["frame"] not in event["frames"] for r in selected.values()):
                raise ValueError("role_outside_occurrence_evidence")
            fact = {
                "fact_id": f"event:{n}",
                "kind": "event",
                "predicate": event["predicate"],
                "roles": {k: r["instance"] for k, r in selected.items()},
                "value": None,
                "evidence_by_slot": {
                    "predicate": event["frames"],
                    **{k: [r["frame"]] for k, r in selected.items()},
                },
                "joint_evidence": event["frames"],
                "observed_media_ids": event["frames"],
                "unresolved_slots": event["unresolved_slots"],
            }
            validate({**base, "facts": [fact]}, audit["schema"], audit["reference_endpoints"])
        except (KeyError, ValueError, ValidationError) as error:
            rejected.append({"record": event, "reason": str(error)})
            continue
        facts.append(fact)
        bindings[fact["fact_id"]] = {r["role"]: r["region"] for r in event["roles"]}
    for n, p in enumerate(raw["properties"]):
        try:
            owner = regions[p["owner_region"]]
            if owner["frame"] not in p["frames"]:
                raise ValueError("owner_outside_property_evidence")
            fact = {
                "fact_id": f"property:{n}",
                "kind": p["kind"],
                "predicate": p["predicate"],
                "roles": {"owner": owner["instance"]},
                "value": p["value"],
                "evidence_by_slot": {
                    "predicate": p["frames"],
                    "value": p["frames"],
                    "owner": [owner["frame"]],
                },
                "joint_evidence": p["frames"],
                "observed_media_ids": p["frames"],
                "unresolved_slots": [],
            }
            if p["kind"] == "text":
                fact["carrier"] = p.get("carrier", "unknown")
            validate({**base, "facts": [fact]}, audit["schema"], audit["reference_endpoints"])
        except (KeyError, ValueError, ValidationError) as error:
            rejected.append({"record": p, "reason": str(error)})
            continue
        facts.append(fact)
        bindings[fact["fact_id"]] = {"owner": p["owner_region"]}
    links = []
    for link in raw["links"]:
        try:
            validate({**base, "links": [link]}, audit["schema"], audit["reference_endpoints"])
        except (ValueError, ValidationError) as error:
            rejected.append({"record": link, "reason": str(error)})
        else:
            links.append(link)
    audit["rejected_records"] = rejected
    gaps = [{"reason": g, "media_ids": []} for g in raw["gaps"]]
    if rejected:
        gaps.append({"reason": "invalid_occurrence_or_link_quarantined", "media_ids": []})
    audit["region_bindings"] = bindings
    return {**base, "facts": facts, "links": links, "coverage_gaps": gaps}


class EventVisionClient(VisionClient):
    def __init__(self, processor_max_pixels=100352, **kwargs):
        self.processor_max_pixels = processor_max_pixels
        kwargs.setdefault("image_max_edge", 1024)
        kwargs.setdefault("frame_labels", True)
        kwargs.setdefault("max_instances", 20)
        kwargs.setdefault("max_facts", 28)
        super().__init__(**kwargs)

    def build_request(self, clip, *, references=(), detections=()):
        body, audit = super().build_request(clip, references=references, detections=detections)
        target = [p["media_id"] for p in audit["presentations"] if p["scope"] == "TARGET"]
        wire = event_schema(
            target,
            list(audit["source_media_aliases"]),
            list(audit["reference_endpoints"]),
            self.max_instances,
        )
        manifest = audit["prompt"].split("INPUT_MANIFEST\n")[1].split("\nOUTPUT_SCHEMA")[0]
        prompt = (
            PROMPT
            + "\nCoordinates: integer xyxy [0,1000], relative to the full labeled image.\nINPUT_MANIFEST\n"
            + manifest
        )
        body["messages"][0]["content"][-1] = {"type": "text", "text": prompt}
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "event_regions", "schema": transport_schema(wire)},
        }
        body.setdefault("mm_processor_kwargs", {})["max_pixels"] = self.processor_max_pixels
        audit["processor_max_pixels"] = self.processor_max_pixels
        body["temperature"] = 0
        body["repetition_penalty"] = 1.05
        body["max_tokens"] = self.max_tokens
        audit.update(prompt=prompt, prompt_version="rrt-event-regions-v3", wire_schema=wire)
        audit["schema"]["properties"]["instances"]["items"]["properties"]["regions"]["maxItems"] = (
            128
        )
        return body, audit

    def normalize_wire(self, raw):
        return raw

    def decode_output(self, text, audit):
        return expand_events(json.loads(text), audit)

    def observe(self, clip, *, out, **kwargs):
        result = super().observe(clip, out=out, **kwargs)
        path = Path(out) / result["request_key"]
        raw = json.loads(
            json.loads((path / "response.json").read_text())["choices"][0]["message"]["content"]
        )
        raw = self.normalize_wire(raw)
        audit = json.loads((path / "request.json").read_text())
        prefix = clip.clip_id + ":"
        # Explicit spatial records survive normalization; no fact may create a new anchor.
        insts = {i["instance_id"]: i for i in result["observation"]["instances"]}
        spatial = []
        for r in raw["regions"]:
            key = prefix + r["instance"]
            mid = audit["source_media_aliases"][r["frame"]]
            if key not in insts:
                continue
            spatial.append(
                {
                    "region_id": prefix + r["id"],
                    "local_instance": key,
                    "media_id": mid,
                    "declared_box": r["box"],
                    "coordinate_space": "sent_image_1000",
                    "semantic_support_audited": False,
                }
            )
        result["spatial_observations"] = spatial
        kept = {f["fact_id"] for f in result["observation"]["facts"]}
        result["fact_region_bindings"] = {
            prefix + f"event:{n}": {r["role"]: prefix + r["region"] for r in f["roles"]}
            for n, f in enumerate(raw["events"])
            if prefix + f"event:{n}" in kept
        }
        result["fact_region_bindings"].update(
            {
                prefix + f"property:{n}": {"owner": prefix + p["owner_region"]}
                for n, p in enumerate(raw["properties"])
                if prefix + f"property:{n}" in kept
            }
        )
        result["architecture"] = "rrt-abcd-v1"
        atomic_json(path / "result.json", result)
        return result


class OccurrenceVisionClient(EventVisionClient):
    """Spatial declarations are grouped with the event whose roles consume them.

    The role never declares geometry. Per-occurrence tables remove the need to
    anticipate all later event anchor times in one global spatial list.
    """

    def __init__(
        self,
        candidate_observations=None,
        multimodal_context=None,
        max_events=8,
        max_properties=8,
        max_links=6,
        typed_ownership=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.typed_ownership = typed_ownership
        self.candidate_observations = candidate_observations
        self.multimodal_context = multimodal_context
        if not (1 <= max_events <= 8 and 0 <= max_properties <= 8 and 0 <= max_links <= 6):
            raise ValueError("invalid_occurrence_output_budget")
        self.output_limits = (max_events, max_properties, max_links)

    def build_request(self, clip, *, references=(), detections=()):
        body, audit = super().build_request(clip, references=references, detections=detections)
        flat = copy.deepcopy(audit["wire_schema"])
        wire = copy.deepcopy(flat)
        region = wire["properties"]["regions"]["items"]
        event = wire["properties"]["events"]["items"]
        fields = event["properties"]
        fields = {
            "predicate": fields["predicate"],
            "frames": arr(fields["frames"]["items"], 4, 1),
            "regions": arr(region, 8, 1),
            "roles": fields["roles"],
            "unresolved_slots": fields["unresolved_slots"],
        }
        wire["properties"]["events"]["items"] = obj(fields)
        prop = wire["properties"]["properties"]["items"]["properties"]
        wire["properties"]["properties"]["items"] = obj(
            {
                "kind": prop["kind"],
                "predicate": prop["predicate"],
                "value": prop["value"],
                "region": region,
                "carrier": prop["carrier"],
                "frames": arr(prop["frames"]["items"], 4, 1),
            }
        )
        wire["properties"].pop("regions")
        wire["required"].remove("regions")
        for key, limit in zip(("events", "properties", "links"), self.output_limits):
            wire["properties"][key]["maxItems"] = limit
        if self.typed_ownership:
            link_fields = wire["properties"]["links"]["items"]["properties"]
            link_fields["relation"] = enum(["continues", "same_identity"])
            wire["properties"]["ownership"] = arr(
                obj(
                    {
                        "part": link_fields["source"],
                        "whole": link_fields["target"],
                        "verdict": link_fields["verdict"],
                        "evidence_ids": link_fields["evidence_ids"],
                        "basis": link_fields["basis"],
                    }
                ),
                self.output_limits[2],
            )
            wire["required"].append("ownership")
        extra = """\nOUTPUT ORGANIZATION: instances, events, properties, links, gaps.
Each EVENT contains its OWN small regions table, immediately before roles. Region
IDs are scoped to that event and may be reused in another event. Each region frame
must be one of that event's 1-4 evidence frames. Include a person-region at the action
time for the actor, and separate object/body-surface regions for other roles.
Each PROPERTY contains a single region record for its owner at a property evidence frame.
The same instance ID must persist across these spatial observations. Do not infer an
ongoing action from an object, inscription or injury that is merely visible.
Links are between INSTANCE IDs, not region IDs. Do not emit self-links. Empty links
is correct if there are no distinct endpoints to connect. For part_of, source is the
region-kind instance and target is the person/object instance; holding is never part_of.
"""
        extra += """
JSON LAYOUT (limits are maxima, never quotas):
{"instances":[{"id":"i0","kind":"person|object|region","description":"visible appearance",
   "attributes":[{"dimension":"color|shape|material|size","value":"open text, e.g. umbrella"}]}],
 "events":[{"predicate":"specific observed action","frames":["f0"],
   "regions":[{"id":"r0","instance":"i0","frame":"f0","box":[0,0,1000,1000]}],
   "roles":[{"role":"directed semantic role","region":"r0"}],"unresolved_slots":[]}],
 "properties":[{"kind":"state|attribute|text","predicate":"property","value":"observed value",
   "region":{"id":"r0","instance":"i0","frame":"f0","box":[0,0,1000,1000]},"frames":["f0"]}],
 "links":[{"source":"i0","target":"i1","relation":"continues|part_of|same_identity",
   "verdict":"supported|contradicted|unresolved","evidence_ids":["f0"],"basis":"visual reason"}],
 "gaps":[]}
Replace every placeholder with what is actually visible. The example box is not a
valid default. One continuous action is ONE event, not an event for each sampled
frame. Choose representative frames across its process. Distinct participants
need independently located tight regions. Do not copy identical boxes to different
people. If a participant is off-camera, leave that role unresolved.
"""
        extra += (
            "\nOutput budgets: "
            + json.dumps(dict(zip(("events", "properties", "links"), self.output_limits)))
            + ". These are upper bounds, not quotas.\n"
        )
        if self.typed_ownership:
            extra = extra.replace("continues|part_of|same_identity", "continues|same_identity")
            extra += """
OWNERSHIP SCHEMA OVERRIDE: emit physical part-whole relations in the separate
ownership array, NOT links. Each row is {"part":"local region instance ID",
"whole":"person/object instance ID", "verdict":"supported|contradicted|unresolved",
"evidence_ids":["target/reference frame IDs"],"basis":"visible physical connection"}.
The hand or forearm is the PART; the person it belongs to is the WHOLE. Never reverse
these endpoints. Holding a tool, carrying another person, or touching an object
is an event, not ownership. Include visual anchors for both endpoints in the cited
frames. If a connected limb and torso establish physical belonging, record it;
a face need not be visible to establish local ownership. Global identity remains separate.
Read text only from its actual surface. Preserve units and distinguish similar digits;
if unreadable report a gap. Do not complete text from context. Record one continuous
occurrence across its frames, not several copies of the same action.
"""
        if self.candidate_observations:
            extra += (
                "\nCANDIDATE OBSERVATIONS FROM A SEPARATE VIEWING OF THIS INPUT:\n"
                + self.candidate_observations
                + "\nThese are unverified proposals. Recheck the video; discard errors. "
                "They cannot supply pixel evidence or certify identities.\n"
            )
        if self.multimodal_context:
            extra += "\nMULTIMODAL NAVIGATION CONTEXT (not pixel evidence):\n" + json.dumps(
                self.multimodal_context
            )
        audit["multimodal_context"] = self.multimodal_context
        audit["candidate_observations"] = self.candidate_observations
        prompt = (
            audit["prompt"].split("\nINPUT_MANIFEST\n")[0]
            + extra
            + "\nINPUT_MANIFEST\n"
            + audit["prompt"].split("\nINPUT_MANIFEST\n")[1]
        )
        body["messages"][0]["content"][-1] = {"type": "text", "text": prompt}
        body["response_format"]["json_schema"]["schema"] = transport_schema(wire)
        audit.update(
            prompt=prompt,
            prompt_version="rrt-occurrence-typed-ownership-v6"
            if self.typed_ownership
            else "rrt-occurrence-regions-v5",
            wire_schema=wire,
            flat_schema=flat,
        )
        return body, audit

    def normalize_wire(self, raw):
        raw = copy.deepcopy(raw)
        for row in raw.pop("ownership", []):
            raw["links"].append(
                {
                    "source": row["part"],
                    "target": row["whole"],
                    "relation": "part_of",
                    "verdict": row["verdict"],
                    "evidence_ids": row["evidence_ids"],
                    "basis": row["basis"],
                }
            )
        regions = []
        for event in raw["events"]:
            local = event.pop("regions")
            mapping = {}
            for r in local:
                if r["id"] in mapping:
                    # Duplicate region ids inside one event are a model error.
                    # Keep the first mapping and drop later duplicates instead of
                    # failing the entire window.
                    continue
                rid = f"r{len(regions)}"
                mapping[r["id"]] = rid
                regions.append({**r, "id": rid})
            # Undeclared region references drop only that role; the event's other
            # roles survive. A role-less event is then rejected per record by
            # expand_events instead of raising here and losing the whole window.
            event["roles"] = [
                {**role, "region": mapping[role["region"]]}
                for role in event["roles"]
                if role["region"] in mapping
            ]
        for prop in raw["properties"]:
            r = prop.pop("region")
            rid = f"r{len(regions)}"
            regions.append({**r, "id": rid})
            prop["owner_region"] = rid
        raw["regions"] = regions
        # A repeated link or a repeated evidence frame carries no extra
        # information and would later fail the full-schema uniqueItems check.
        # Deduplicate losslessly; distinct links are preserved.
        dedup = {}
        for link in raw["links"]:
            link["evidence_ids"] = list(dict.fromkeys(link["evidence_ids"]))
            dedup[json.dumps(link, sort_keys=True)] = link
        raw["links"] = list(dedup.values())
        return raw

    def decode_output(self, text, audit):
        raw = json.loads(text)
        Draft202012Validator(transport_schema(audit["wire_schema"])).validate(raw)
        flat = self.normalize_wire(raw)
        flat_audit = {**audit, "wire_schema": audit["flat_schema"]}
        result = expand_events(flat, flat_audit)
        audit["rejected_records"] = flat_audit["rejected_records"]
        return result
