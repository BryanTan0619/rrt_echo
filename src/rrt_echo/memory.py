"""Append-only observations and immutable, rebuildable hypergraph snapshots."""

from __future__ import annotations

from dataclasses import asdict

from .identity import IdentityGraph, IdentityProposal
from .schema import SCHEMA_VERSION, Fact, ObservationPacket, digest


def support(ok: bool, evidence=(), reason="missing_evidence") -> dict:
    return {
        "status": "supported" if ok else "insufficient",
        "evidence_ids": sorted(set(evidence)),
        "gap_reasons": [] if ok else [reason],
        "assessment": "structural",
        "media_audit_required": True,
    }


class Memory:
    def __init__(self, *, source_video_id: str | None = None):
        self.source_video_id = source_video_id
        self._packets: dict[str, ObservationPacket] = {}
        self.identity = IdentityGraph()
        self._video_id: str | None = None

    @property
    def packets(self) -> tuple[ObservationPacket, ...]:
        return tuple(self._packets.values())

    @property
    def instances(self):
        return {i.instance_id: i for p in self.packets for i in p.instances}

    def append(self, packet: ObservationPacket):
        if self._video_id is not None and packet.video_id != self._video_id:
            raise ValueError("one memory cannot silently combine videos")
        if packet.observation_id in self._packets:
            raise ValueError("observations are append-only")
        if set(self.instances).intersection(i.instance_id for i in packet.instances):
            raise ValueError("local instances are packet-scoped; continuity requires a link")
        existing_facts = {f.fact_id for p in self.packets for f in p.facts}
        if existing_facts.intersection(f.fact_id for f in packet.facts):
            raise ValueError("duplicate occurrence/fact ID")
        old_media = {m.media_id: m for p in self.packets for m in p.media}
        for media in packet.media:
            if media.media_id in old_media and media != old_media[media.media_id]:
                raise ValueError("conflicting canonical media ID")
        self._packets[packet.observation_id] = packet
        self._video_id = packet.video_id

    def compare(self, proposal: IdentityProposal):
        return self.identity.propose(proposal, self.instances)

    @staticmethod
    def _dimensions(fact: Fact, instances, identity_mapping) -> dict:
        slots = dict(fact.evidence_by_slot)
        joint = set(fact.joint_evidence)
        pred_ok = bool(joint.intersection(slots.get("predicate", ())))
        roles = dict(fact.roles)
        role_checks = {}
        for role, ref in roles.items():
            visible = {r.media_id for r in instances[ref].regions}
            ids = visible.intersection(slots.get(role, ())).intersection(joint)
            role_checks[role] = support(bool(ids), ids, "missing_role_region_or_joint_evidence")
        local_gaps = [s for s in fact.unresolved_slots if not s.startswith("identity:")]
        if fact.kind != "event":
            pred_ok = (
                pred_ok
                and "owner" in roles
                and fact.value is not None
                and bool(joint.intersection(slots.get("value", ())))
            )
        joint_ok = (
            pred_ok
            and not local_gaps
            and all(v["status"] == "supported" for v in role_checks.values())
        )
        identity_checks = {
            role: support(
                bool(identity_mapping[ref]["identity_decisions"]),
                identity_mapping[ref]["identity_decisions"],
                "global_identity_unresolved",
            )
            for role, ref in roles.items()
        }
        owner = role_checks.get("owner", support(False, reason="owner_not_bound"))
        return {
            "local_joint": support(joint_ok, fact.joint_evidence, "local_joint_incomplete"),
            "roles": role_checks,
            "global_identity": identity_checks,
            "region_owner": owner,
            "text_content": support(
                fact.kind == "text" and bool(fact.value) and bool(slots.get("value")),
                slots.get("value", ()),
                "characters_not_observed",
            ),
            "observed_time": support(bool(fact.observed_media_ids), fact.observed_media_ids),
            "duration": support(
                fact.time_bounds is not None, fact.boundary_evidence, "event_boundaries_unresolved"
            ),
            "persistence": support(False, reason="no_persistence_rule_registered"),
        }

    def snapshot(self) -> dict:
        instances = self.instances
        entities, mapping = self.identity.projection(instances)
        media = {
            m.media_id: asdict(m) | {"seconds": m.seconds} for p in self.packets for m in p.media
        }
        facts = []
        for packet in self.packets:
            for fact in packet.facts:
                roles = dict(fact.roles)
                observed = sorted({media[mid]["seconds"] for mid in fact.observed_media_ids})
                dimensions = self._dimensions(fact, instances, mapping)
                dependencies = sorted(
                    {d for ref in roles.values() for d in mapping[ref]["identity_decisions"]}
                )
                facts.append(
                    {
                        "fact_id": fact.fact_id,
                        "kind": fact.kind,
                        "predicate": fact.predicate,
                        "roles": roles,
                        "value": fact.value,
                        "carrier": fact.carrier,
                        "resolved_roles": {
                            role: {
                                "local_instance": ref,
                                "entity_id": mapping[ref]["entity_id"],
                                "entity_version": mapping[ref]["version"],
                            }
                            for role, ref in roles.items()
                        },
                        "evidence_by_slot": dict(fact.evidence_by_slot),
                        "joint_evidence": fact.joint_evidence,
                        "evidence_family": packet.family_id or packet.observation_id,
                        "observed_times": observed,
                        "extent": [observed[0], observed[-1]] if observed else None,
                        "time_bounds": fact.time_bounds,
                        "boundary_evidence": fact.boundary_evidence,
                        "support": dimensions,
                        "unresolved_slots": fact.unresolved_slots,
                        "dependencies": {
                            "observation": packet.observation_id,
                            "identity": dependencies,
                        },
                    }
                )
        local_tracklets = {}
        for packet in self.packets:
            for instance in packet.instances:
                if instance.track_ref:
                    track = local_tracklets.setdefault(
                        instance.track_ref,
                        {
                            "tracklet_id": instance.track_ref,
                            "shot_id": packet.shot_id,
                            "local_sightings": [],
                            "observed_media_ids": [],
                            "status": "local_correspondence_not_global_identity",
                        },
                    )
                    track["local_sightings"].append(instance.instance_id)
                    track["observed_media_ids"] = sorted(
                        set(track["observed_media_ids"]) | {r.media_id for r in instance.regions}
                    )
        from .rules import temporal_relations

        graph = {
            "schema": SCHEMA_VERSION,
            "video_id": self._video_id,
            "source_video_id": self.source_video_id,
            "protocol": "offline_query_blind_graph_only",
            "identity_revision": self.identity.revision,
            "instances": [asdict(i) for i in instances.values()],
            "local_tracklets": list(local_tracklets.values()),
            "entities": entities,
            "identity_decisions": self.identity.export(),
            "facts": facts,
            "relations": temporal_relations(facts),
            "evidence": media,
            "support_scope": "structural_only_independent_media_audit_required",
        }
        graph["snapshot_id"] = digest(graph)
        return graph
