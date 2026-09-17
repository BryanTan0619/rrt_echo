"""Small immutable wire contract. Video time and transaction time are distinct."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "rrt_echo.hypergraph.v1"
VERDICTS = frozenset({"supported", "contradicted", "insufficient"})


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite coordinate/time")
    return value


@dataclass(frozen=True)
class Media:
    media_id: str
    uri: str
    pts: int
    time_base: tuple[int, int]
    sha256: str
    kind: str = "frame"

    @property
    def seconds(self) -> float:
        return self.pts * self.time_base[0] / self.time_base[1]

    def __post_init__(self):
        if (
            not self.media_id
            or not self.uri
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("media requires ID, URI and SHA256")
        if (
            not isinstance(self.pts, int)
            or len(self.time_base) != 2
            or not all(isinstance(v, int) for v in self.time_base)
        ):
            raise ValueError("PTS and time base must be integers")
        if self.time_base[0] <= 0 or self.time_base[1] <= 0 or self.pts < 0:
            raise ValueError("invalid source PTS/time base")


@dataclass(frozen=True)
class Region:
    media_id: str
    box: tuple[float, float, float, float]  # normalized xyxy
    source: str = "vlm"

    def __post_init__(self):
        if len(self.box) != 4:
            raise ValueError("region must have four coordinates")
        x1, y1, x2, y2 = map(finite, self.box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("region outside normalized frame")


@dataclass(frozen=True)
class Instance:
    instance_id: str
    kind: str
    regions: tuple[Region, ...]
    description: str = ""  # current appearance, never a global name
    track_ref: str | None = None  # candidate cue, not a global identity
    attributes: tuple[tuple[str, str], ...] = ()  # (dimension, value) pairs


@dataclass(frozen=True)
class Fact:
    fact_id: str
    kind: str  # event/state/attribute/text
    predicate: str
    roles: tuple[tuple[str, str], ...]
    evidence_by_slot: tuple[tuple[str, tuple[str, ...]], ...]
    joint_evidence: tuple[str, ...]
    observed_media_ids: tuple[str, ...]
    value: str | None = None
    carrier: str | None = None  # text facts only: inscribed/screen/overlay/unknown
    unresolved_slots: tuple[str, ...] = ()
    # Bounds are only accepted when explicitly marked by a reviewed proposal.
    time_bounds: tuple[float, float] | None = None
    boundary_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationPacket:
    observation_id: str
    video_id: str
    shot_id: str
    media: tuple[Media, ...]
    instances: tuple[Instance, ...]
    facts: tuple[Fact, ...]
    extractor: str = "unknown"
    family_id: str = ""

    def __post_init__(self):
        if not self.observation_id or not self.video_id or not self.shot_id:
            raise ValueError("observation/video/shot IDs are required")
        mids = [m.media_id for m in self.media]
        ids = [i.instance_id for i in self.instances]
        fids = [f.fact_id for f in self.facts]
        if len(mids) != len(set(mids)) or len(ids) != len(set(ids)) or len(fids) != len(set(fids)):
            raise ValueError("duplicate local record ID")
        media_set, instance_set = set(mids), set(ids)
        for instance in self.instances:
            if not instance.instance_id or not instance.kind:
                raise ValueError("invalid local instance")
            if any(r.media_id not in media_set for r in instance.regions):
                raise ValueError("unknown region media")
        for fact in self.facts:
            if fact.kind not in {"event", "state", "attribute", "text"} or not fact.predicate:
                raise ValueError("invalid fact kind/predicate")
            if len(dict(fact.roles)) != len(fact.roles):
                raise ValueError("duplicate role: use explicit indexed roles")
            if len(dict(fact.evidence_by_slot)) != len(fact.evidence_by_slot):
                raise ValueError("duplicate evidence slot")
            if any(ref not in instance_set for _, ref in fact.roles):
                raise ValueError("unknown local participant")
            refs = (
                *fact.joint_evidence,
                *fact.observed_media_ids,
                *fact.boundary_evidence,
                *(mid for _, values in fact.evidence_by_slot for mid in values),
            )
            if not set(refs) <= media_set:
                raise ValueError("unknown fact evidence")
            if fact.time_bounds is not None:
                start, end = map(finite, fact.time_bounds)
                if not 0 <= start <= end or not fact.boundary_evidence:
                    raise ValueError("time bounds require evidence")
                observed = [m.seconds for m in self.media if m.media_id in fact.boundary_evidence]
                if not observed or start < min(observed) or end > max(observed):
                    raise ValueError("bounds outside boundary evidence")

    def to_dict(self) -> dict:
        result = asdict(self)
        for fact in result["facts"]:
            fact["roles"] = dict(fact["roles"])
            fact["evidence_by_slot"] = dict(fact["evidence_by_slot"])
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ObservationPacket:
        # Unknown top-level data cannot grant identity/commit authority.
        allowed = {
            "observation_id",
            "video_id",
            "shot_id",
            "media",
            "instances",
            "facts",
            "extractor",
            "family_id",
        }
        if set(data) - allowed:
            raise ValueError(f"unexpected observation fields: {sorted(set(data) - allowed)}")
        return cls(
            observation_id=data["observation_id"],
            video_id=data["video_id"],
            shot_id=str(data["shot_id"]),
            extractor=data.get("extractor", "unknown"),
            family_id=data.get("family_id", data["observation_id"]),
            media=tuple(Media(**{**m, "time_base": tuple(m["time_base"])}) for m in data["media"]),
            instances=tuple(
                Instance(
                    instance_id=i["instance_id"],
                    kind=i["kind"],
                    description=i.get("description", ""),
                    track_ref=i.get("track_ref"),
                    regions=tuple(
                        Region(r["media_id"], tuple(r["box"]), r.get("source", "vlm"))
                        for r in i.get("regions", [])
                    ),
                    attributes=tuple(
                        (a["dimension"], a["value"])
                        for a in i.get("attributes", [])
                    ),
                )
                for i in data.get("instances", [])
            ),
            facts=tuple(
                Fact(
                    fact_id=f["fact_id"],
                    kind=f["kind"],
                    predicate=f["predicate"],
                    roles=tuple(sorted(f.get("roles", {}).items())),
                    evidence_by_slot=tuple(
                        (k, tuple(v)) for k, v in sorted(f.get("evidence_by_slot", {}).items())
                    ),
                    joint_evidence=tuple(f.get("joint_evidence", [])),
                    observed_media_ids=tuple(f.get("observed_media_ids", [])),
                    value=f.get("value"),
                    carrier=f.get("carrier"),
                    unresolved_slots=tuple(f.get("unresolved_slots", [])),
                    time_bounds=tuple(f["time_bounds"])
                    if f.get("time_bounds") is not None
                    else None,
                    boundary_evidence=tuple(f.get("boundary_evidence", [])),
                )
                for f in data.get("facts", [])
            ),
        )
