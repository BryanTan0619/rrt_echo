"""D: immutable observations, revisable correspondence journal, reader-compatible memory."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ..echo_perception.adapter import legacy_packet
from ..echo_perception.graph import compile_graph
from ..identity import IdentityProposal
from ..memory import Memory
from ..schema import ObservationPacket, digest
from .diagnostics import (
    independent_person_anchors,
    replacement_constraints,
    whole_person_separations,
)
from .event_content import attach_event_contents


class RRTMemory:
    def __init__(self, source_video_id=None, ledger_dir=None):
        self.source_video_id = source_video_id
        self.observations = []
        self.journal = []
        self.audio = None
        self.text_revisions = []
        self.ledger_dir = Path(ledger_dir) if ledger_dir else None
        self._head = None
        self._sequence = 0
        self._in_append = False
        if self.ledger_dir:
            self.ledger_dir.mkdir(parents=True, exist_ok=True)
            if (self.ledger_dir / "observations.jsonl").exists():
                raise ValueError("ledger_exists_use_open")

    def _record(self, operation, payload):
        if not self.ledger_dir:
            return
        row = {
            "sequence": self._sequence + 1,
            "previous": self._head,
            "source_video_id": self.source_video_id,
            "operation": operation,
            "payload": payload,
        }
        row["sha256"] = digest(row)
        # A single committed line contains an observation and its initial links.
        # Single writer only. A torn final line is rejected, never silently skipped.
        with (self.ledger_dir / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sequence, self._head = row["sequence"], row["sha256"]

    @classmethod
    def open(cls, ledger_dir):
        directory = Path(ledger_dir)
        mem = cls()
        for line in (directory / "observations.jsonl").read_text().splitlines():
            row = json.loads(line)
            checksum = row.pop("sha256")
            if (
                digest(row) != checksum
                or row["previous"] != mem._head
                or row["sequence"] != mem._sequence + 1
            ):
                raise ValueError("invalid_ledger_chain")
            if mem._sequence and row["source_video_id"] != mem.source_video_id:
                raise ValueError("ledger_source_changed")
            mem.source_video_id = row["source_video_id"]
            op, data = row["operation"], row["payload"]
            if op == "observation":
                before = len(mem.journal)
                mem.append(data["result"])
                generated = mem.journal[before:]
                if [x["proposal"] for x in generated] != [x["proposal"] for x in data["bindings"]]:
                    raise ValueError("ledger_initial_bindings_mismatch")
                mem.journal[before:] = data["bindings"]
            elif op == "binding":
                if data["operation"] == "propose":
                    mem.propose(data["proposal"])
                elif data["operation"] == "revoke":
                    mem.revoke(data["proposal_id"], data["reason"])
                else:
                    raise ValueError("unknown_binding_operation")
                if mem.journal[-1]["revision"] != data["revision"]:
                    raise ValueError("binding_revision_mismatch")
                mem.journal[-1] = data
            elif op == "text_revision":
                current = next(
                    f
                    for r in mem.observations
                    for f in r["observation"]["facts"]
                    if f["fact_id"] == data["fact_id"]
                )
                if digest(current) != data["previous_fact_sha256"]:
                    raise ValueError("text_revision_predecessor_mismatch")
                mem.revise_text(data["fact_id"], data["value"], data["review"])
                if mem.text_revisions[-1] != data:
                    raise ValueError("text_revision_mismatch")
            elif op == "audio":
                mem.set_audio(data)
            else:
                raise ValueError("unknown_ledger_operation")
            mem._sequence, mem._head = row["sequence"], checksum
        mem.ledger_dir = directory
        return mem

    def set_audio(self, audio):
        from .audio import validate_audio

        validate_audio(audio)
        if self.audio is not None:
            raise ValueError("audio_already_registered")
        self._record("audio", audio)
        self.audio = copy.deepcopy(audio)

    def revise_text(self, fact_id, value, review):
        """Version only a literal transcription; roles and original log stay intact."""
        found = [
            (r, f)
            for r in self.observations
            for f in r["observation"]["facts"]
            if f["fact_id"] == fact_id
        ]
        if len(found) != 1:
            raise ValueError("unknown_text_fact")
        result, fact = found[0]
        if fact["kind"] != "text" or not isinstance(value, str) or not value.strip():
            raise ValueError("literal_text_revision_required")
        if (
            review.get("verdict") != "readable"
            or not review.get("model")
            or not review.get("basis")
        ):
            raise ValueError("unaccepted_text_review")
        crops = review.get("crops", [])
        if not crops or not {c["source_media_id"] for c in crops} <= set(fact["joint_evidence"]):
            raise ValueError("text_revision_outside_original_evidence")
        media = {m["media_id"]: m for m in result["observation"]["media"]}
        owner = fact["roles"].get("owner")
        inst = next(i for i in result["observation"]["instances"] if i["instance_id"] == owner)
        for crop in crops:
            mid, box = crop["source_media_id"], crop["source_box"]
            if crop["source_sha256"] != media[mid]["sha256"] or len(crop["sha256"]) != 64:
                raise ValueError("text_crop_source_mismatch")
            owner_box = crop.get("owner_box", box)
            if not (
                0 <= box[0] <= owner_box[0] < owner_box[2] <= box[2] <= 1
                and 0 <= box[1] <= owner_box[1] < owner_box[3] <= box[3] <= 1
            ):
                raise ValueError("text_crop_does_not_contain_owner_region")
            if not any(
                r["media_id"] == mid and list(r["box"]) == list(owner_box) for r in inst["regions"]
            ):
                raise ValueError("text_crop_not_owner_region")
        update = {
            "fact_id": fact_id,
            "version": 2 + sum(r["fact_id"] == fact_id for r in self.text_revisions),
            "previous_fact_sha256": digest(fact),
            "previous_value": fact["value"],
            "value": value,
            "review": copy.deepcopy(review),
        }
        self._record("text_revision", update)
        fact["value"] = value
        self.text_revisions.append(update)

    def materialize(self, out=None):
        from ..echo_perception.types import atomic_json
        from .audio import attach_audio
        from .event_content import attach_coobserved_attributes
        from .hypergraph import TemporalEvidenceHypergraph

        graph = json.loads(json.dumps(self.snapshot()))
        registry = self.registry(graph)
        status = {e["entity_id"]: e["status"] for e in registry["entities"]}
        for entity in graph["entities"]:
            entity["resolution_status"] = status[entity["entity_id"]]
        for fact in graph["facts"]:
            for role in fact["resolved_roles"].values():
                role["identity_status"] = status[role["entity_id"]]
        for instance in graph["instances"]:
            times = sorted(
                {graph["evidence"][r["media_id"]]["seconds"] for r in instance["regions"]}
            )
            instance["extent"] = [times[0], times[-1]] if times else None
            # Samples are direct observations; do not fill the interval between them.
            instance["observed_segments"] = [[t, t] for t in times]
            instance["continuity_links"] = [
                p["proposal_id"]
                for p in graph["binding_links"]
                if p["relation"] == "continues"
                and p["status"] == "accepted"
                and instance["instance_id"] in (p["source"], p["target"])
            ]
        graph["architecture"] = "echo-simple-v1"
        graph.pop("snapshot_id", None)
        graph["snapshot_id"] = digest(graph)
        if self.audio is not None:
            graph = attach_audio(graph, self.audio)
        graph = attach_coobserved_attributes(graph)
        graph = TemporalEvidenceHypergraph(graph).snapshot()
        if out is not None:
            out = Path(out)
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "hypergraph.json", graph)
            atomic_json(out / "storyline.json", narrative_projection(graph))
            atomic_json(out / "entity_registry.json", self.registry(graph))
            atomic_json(out / "gaps.json", graph.get("coverage_gaps", []))
            # Disposable projections, never loaded as a second source of truth.
            for name, rows in (("facts", graph["facts"]), ("bindings", self.journal)):
                target = out / (name + ".jsonl")
                tmp = target.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                tmp.replace(target)
            atomic_json(
                out / "view_manifest.json",
                {
                    "canonical_log": str(self.ledger_dir / "observations.jsonl")
                    if self.ledger_dir
                    else None,
                    "ledger_head": self._head,
                    "snapshot_id": graph["snapshot_id"],
                    "derived": [
                        "facts.jsonl",
                        "bindings.jsonl",
                        "entity_registry.json",
                        "hypergraph.json",
                    ],
                    "semantic_acceptance": None,
                },
            )
        return graph

    @staticmethod
    def registry(graph, max_references=3):
        instances = {i["instance_id"]: i for i in graph["instances"]}
        rows = []
        for entity in graph["entities"]:
            members = entity["local_instances"]
            links = [
                p["proposal_id"]
                for p in graph.get("binding_links", [])
                if p["status"] == "accepted"
                and p["relation"] in {"same_identity", "continues"}
                and p["source"] in members
                and p["target"] in members
            ]
            refs = [
                {"local_instance": key, **r} for key in members for r in instances[key]["regions"]
            ]
            refs.sort(
                key=lambda r: (
                    -(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]),
                    r["media_id"],
                )
            )
            rows.append(
                {
                    "entity_id": entity["entity_id"],
                    "local_instances": members,
                    "status": "linked_candidate" if links else "unresolved_local",
                    "binding_ids": links,
                    "visual_references": refs[:max_references],
                    "voice_references": [],
                    "semantic_support_audited": False,
                }
            )
        return {
            "derived_from": graph["snapshot_id"],
            "entities": rows,
            "reference_selection": "largest_region_area_proxy_not_face_quality",
        }

    def append(self, result):
        oid = result["observation"]["observation_id"]
        if any(r["observation"]["observation_id"] == oid for r in self.observations):
            raise ValueError("duplicate_observation")
        legacy_packet(result, ObservationPacket.from_dict, extractor="rrt-abcd-v1")
        links = [*result["link_proposals"], *replacement_constraints(result)]
        ids = [p["proposal_id"] for p in links]
        existing = {r.get("proposal", {}).get("proposal_id") for r in self.journal}
        if len(ids) != len(set(ids)) or existing.intersection(ids):
            raise ValueError("duplicate_proposal")
        before = len(self.journal)
        self.observations.append(copy.deepcopy(result))
        self._in_append = True
        try:
            for link in links:
                self.propose(link)
            self._record("observation", {"result": result, "bindings": self.journal[before:]})
        except Exception:
            self.observations.pop()
            del self.journal[before:]
            raise
        finally:
            self._in_append = False

    def propose(self, link):
        if any(
            row.get("proposal", {}).get("proposal_id") == link["proposal_id"]
            for row in self.journal
        ):
            raise ValueError("duplicate_proposal")
        self.journal.append(
            {
                "revision": len(self.journal) + 1,
                "transaction_time": datetime.now(timezone.utc).isoformat(),
                "operation": "propose",
                "proposal": copy.deepcopy(link),
            }
        )

        if not self._in_append:
            try:
                self._record("binding", self.journal[-1])
            except Exception:
                self.journal.pop()
                raise

    def revoke(self, proposal_id, reason):
        proposed = {
            r["proposal"]["proposal_id"] for r in self.journal if r["operation"] == "propose"
        }
        retired = {r["proposal_id"] for r in self.journal if r["operation"] == "revoke"}
        if proposal_id not in proposed - retired or not reason:
            raise ValueError("invalid_revocation")
        self.journal.append(
            {
                "revision": len(self.journal) + 1,
                "transaction_time": datetime.now(timezone.utc).isoformat(),
                "operation": "revoke",
                "proposal_id": proposal_id,
                "reason": reason,
            }
        )

        try:
            self._record("binding", self.journal[-1])
        except Exception:
            self.journal.pop()
            raise

    def snapshot(self):
        if not self.observations:
            raise ValueError("no_observations")
        results = copy.deepcopy(self.observations)
        independent = independent_person_anchors(results)
        endpoint_gaps = []
        for r in results:
            for fact in r["observation"]["facts"]:
                for role, key in fact["roles"].items():
                    if key in independent and not independent[key].intersection(
                        fact["evidence_by_slot"].get(role, [])
                    ):
                        if role not in fact["unresolved_slots"]:
                            fact["unresolved_slots"].append(role)
                        endpoint_gaps.append(
                            {
                                "fact_id": fact["fact_id"],
                                "role": role,
                                "instance_id": key,
                                "reason": "person_roles_share_composite_box",
                            }
                        )
        separations = whole_person_separations(results)
        separated_pairs = {frozenset((c["left"], c["right"])) for c in separations}
        for r in results:
            r["link_proposals"] = []
        retired = {r["proposal_id"] for r in self.journal if r["operation"] == "revoke"}
        active = [
            copy.deepcopy(r["proposal"])
            for r in self.journal
            if r["operation"] == "propose" and r["proposal"]["proposal_id"] not in retired
        ]
        aliases = {
            m["media_id"]: m["source_region"]["source_media_id"]
            for r in results
            for m in r.get("reference_evidence", [])
        }
        for p in active:
            mids = {aliases.get(m, m) for m in p["evidence_ids"]}
            if p["verdict"] == "supported" and any(
                key in independent and not mids.intersection(independent[key])
                for key in (p["source"], p["target"])
            ):
                p["source_verdict"] = p["verdict"]
                p["verdict"] = "unresolved"
                p["endpoint_gap"] = "independent_person_anchor_missing"
            if (
                p["relation"] in {"same_identity", "continues"}
                and p["verdict"] == "supported"
                and frozenset((p["source"], p["target"])) in separated_pairs
            ):
                p["source_verdict"] = p["verdict"]
                p["verdict"] = "unresolved"
                p["endpoint_gap"] = "whole_person_event_roles_cannot_collapse"
            if p["relation"] == "continues":
                # Explicit visual continuity entails identity, never persistence of a state.
                p["original_relation"] = "continues"
                p["relation"] = "same_identity"
        results[-1]["link_proposals"] = active
        context = compile_graph(
            results,
            {
                "video_id": results[0]["observation"]["video_id"],
                "version": "rrt-abcd-v1",
                "separation_constraints": separations,
            },
        )
        core = Memory(source_video_id=self.source_video_id)
        for r in results:
            packet, _, _ = legacy_packet(r, ObservationPacket.from_dict, extractor="rrt-abcd-v1")
            core.append(packet)
        aliases = {}
        for r in results:
            for m in r.get("reference_evidence", []):
                aliases[m["media_id"]] = m["source_region"]["source_media_id"]
        # Component conflict resolution is performed before the legacy reader projection.
        for p in context["links"]:
            if p["relation"] != "same_identity" or p["status"] not in {"accepted", "contradicted"}:
                continue
            if not {p["source"], p["target"]} <= core.instances.keys():
                continue
            endpoint_frames = {
                r.media_id
                for key in (p["source"], p["target"])
                for r in core.instances[key].regions
            }
            mids = tuple(
                dict.fromkeys(
                    aliases.get(m, m)
                    for m in p["evidence_ids"]
                    if aliases.get(m, m) in endpoint_frames
                )
            )
            core.compare(
                IdentityProposal(
                    p["proposal_id"],
                    p["source"],
                    p["target"],
                    "same" if p["status"] == "accepted" else "different",
                    mids,
                    p["basis"],
                )
            )
        graph = core.snapshot()
        times = {
            r["proposal"]["proposal_id"]: r["transaction_time"]
            for r in self.journal
            if r["operation"] == "propose"
        }
        for d in graph["identity_decisions"]:
            d["transaction_time"] = times[d["proposal"]["proposal_id"]]
        region_bindings = {
            fid: binding
            for r in results
            for fid, binding in r.get("fact_region_bindings", {}).items()
        }
        owner_links = {}
        for p in context["links"]:
            if p["relation"] == "part_of" and p["status"] == "accepted":
                owner_links.setdefault(p["source"], []).append(p)
        entity_of = {key: e for e in graph["entities"] for key in e["local_instances"]}
        for f in graph["facts"]:
            f["region_bindings"] = region_bindings.get(f["fact_id"], {})
            f["owner_projections"] = {}
            f["dependencies"]["ownership"] = []
            for role, endpoint in f["roles"].items():
                links = owner_links.get(endpoint, [])
                if links:
                    owners = {entity_of[p["target"]]["entity_id"] for p in links}
                    if len(owners) == 1:
                        f["owner_projections"][role] = {
                            "owner_entity": next(iter(owners)),
                            "local_owner_endpoints": sorted({p["target"] for p in links}),
                            "ownership_links": [p["proposal_id"] for p in links],
                            "semantic_support_audited": False,
                        }
                        f["dependencies"]["ownership"].extend(p["proposal_id"] for p in links)
            region_roles = [
                role for role, key in f["roles"].items() if core.instances[key].kind == "region"
            ]
            if region_roles:
                ok = all(role in f["owner_projections"] for role in region_roles)
                f["support"]["region_owner"] = {
                    "status": "supported" if ok else "insufficient",
                    "evidence_ids": f["dependencies"]["ownership"],
                    "assessment": "structural",
                    "media_audit_required": True,
                    "gap_reasons": [] if ok else ["region_owner_unresolved"],
                }
            f["semantic_support_audited"] = False
        graph.update(
            architecture="rrt-abcd-v1",
            binding_revision=len(self.journal),
            binding_journal=copy.deepcopy(self.journal),
            binding_links=context["links"],
            spatial_observations=[s for r in results for s in r.get("spatial_observations", [])],
            coverage_gaps=[g for r in results for g in r.get("coverage_gaps", [])],
            semantic_quality_passed=None,
            endpoint_grounding_gaps=endpoint_gaps,
            acceptance_policy="independent-person-and-role-consistency-v2",
            separation_constraints=separations,
        )
        if self.text_revisions:
            graph["text_revisions"] = copy.deepcopy(self.text_revisions)
            for fact in graph["facts"]:
                updates = [r for r in self.text_revisions if r["fact_id"] == fact["fact_id"]]
                if updates:
                    fact["version"] = updates[-1]["version"]
                    fact["text_review"] = copy.deepcopy(updates[-1]["review"])
        graph.pop("snapshot_id", None)
        graph["snapshot_id"] = digest(graph)
        reviews = [v for r in results for v in r.get("event_content_reviews", [])]
        return attach_event_contents(graph, reviews) if reviews else graph


def narrative_projection(graph):
    """Readable deterministic facts. No generated connective story or inherited captions."""
    people = {i["instance_id"]: i for i in graph["instances"]}
    rows = []
    for f in sorted(
        graph["facts"], key=lambda f: (min(f["observed_times"], default=float("inf")), f["fact_id"])
    ):
        rows.append(
            {
                "fact_id": f["fact_id"],
                "times": f["observed_times"],
                "predicate": f["predicate"],
                "value": f["value"],
                "roles": {
                    role: {**f["resolved_roles"][role], "appearance": people[key]["description"]}
                    for role, key in f["roles"].items()
                },
                "owner_projections": f["owner_projections"],
                "unresolved_slots": f["unresolved_slots"],
                "semantic_support_audited": False,
            }
        )
    return {"snapshot_id": graph["snapshot_id"], "events": rows}
