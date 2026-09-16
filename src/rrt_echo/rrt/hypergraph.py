"""Transactional multimodal evidence graph and scope-preserving proof projection.

Admission checks evidence structure and model review, not infallible perception.
Original observations are immutable; revocation only changes derived bindings.
"""

from __future__ import annotations

import copy
import math

from ..schema import digest
from ..scoped_identity import identity_paths
from .audio import validate_audio

SCHEMA = "rrt.multimodal.v1"


def seal(value):
    value = copy.deepcopy(value)
    value.pop("snapshot_id", None)
    value["snapshot_id"] = digest(value)
    return value


def entity_map(graph):
    return {i: e["entity_id"] for e in graph["entities"] for i in e["local_instances"]}


def validate_submission(graph, packet):
    """Validate a complete transaction before making any state change."""
    if packet.get("schema") != SCHEMA:
        raise ValueError("multimodal_schema_mismatch")
    audio = graph.get("audio_memory")
    if audio is None:
        raise ValueError("audio_memory_required")
    validate_audio(audio)
    if packet["source_sha256"] != audio["source"]["sha256"]:
        raise ValueError("multimodal_source_mismatch")
    if packet["video_id"] != audio["source_video_id"]:
        raise ValueError("multimodal_video_mismatch")
    utterances = {u["utterance_id"]: u for u in audio["utterances"]}
    people = {i["instance_id"]: i for i in graph["instances"] if i["kind"] == "person"}
    bundles = packet["evidence_bundles"]
    for key, b in bundles.items():
        if key != b["evidence_id"] or b["source_sha256"] != packet["source_sha256"]:
            raise ValueError("invalid_multimodal_evidence")
        if not all(math.isfinite(b[k]) for k in ("start", "end")) or b["start"] >= b["end"]:
            raise ValueError("invalid_multimodal_interval")
        if (
            set(b["modalities"]) != {"audio", "video"}
            or b["audio_samples"] <= 0
            or b["video_frames"] < 2
        ):
            raise ValueError("synchronized_audio_video_required")
        if len(b["clip_sha256"]) != 64 or not b["uri"]:
            raise ValueError("source_clip_provenance_required")
    ids = set()
    for p in packet["bindings"]:
        if p["proposal_id"] in ids:
            raise ValueError("duplicate_multimodal_proposal")
        ids.add(p["proposal_id"])
        if p["relation"] != "speaker_of":
            raise ValueError("unsupported_multimodal_relation")
        if p["utterance_id"] not in utterances or p["local_instance"] not in people:
            raise ValueError("dangling_multimodal_endpoint")
        if not p["evidence_ids"] or not set(p["evidence_ids"]) <= bundles.keys():
            raise ValueError("dangling_multimodal_proof")
        u = utterances[p["utterance_id"]]
        for eid in p["evidence_ids"]:
            b = bundles[eid]
            if b["start"] > u["start"] or b["end"] < u["end"]:
                raise ValueError("utterance_outside_evidence")
            if p["local_instance"] not in b["candidate_instances"]:
                raise ValueError("unpresented_speaker_candidate")
        mids = p["visual_media_ids"]
        anchors = {r["media_id"] for r in people[p["local_instance"]]["regions"]}
        if not mids or not set(mids) <= anchors:
            raise ValueError("speaker_visual_endpoint_missing")
        # At least one anchor must belong to the actual inspected interval.
        if not any(
            any(
                bundles[e]["start"]
                <= graph["evidence"][m]["pts"]
                * graph["evidence"][m]["time_base"][0]
                / graph["evidence"][m]["time_base"][1]
                <= bundles[e]["end"]
                for e in p["evidence_ids"]
            )
            for m in mids
        ):
            raise ValueError("speaker_anchor_outside_interval")
        reviews = p.get("reviews", [])
        for r in reviews:
            if (
                r["evidence_id"] not in p["evidence_ids"]
                or r["local_instance"] != p["local_instance"]
            ):
                raise ValueError("review_endpoint_mismatch")
            if r["utterance_id"] != p["utterance_id"]:
                raise ValueError("review_utterance_mismatch")
    claim_ids = set()
    for c in packet.get("claims", []):
        if c["claim_id"] in claim_ids:
            raise ValueError("duplicate_reported_claim")
        claim_ids.add(c["claim_id"])
        if c["source_utterance"] not in utterances or c["kind"] != "reported_claim":
            raise ValueError("claim_requires_utterance_not_world_fact")
        if not c.get("text") or c.get("world_fact") is not False:
            raise ValueError("speech_content_is_not_world_fact")
        if (
            c["text"].strip().casefold()
            != utterances[c["source_utterance"]]["text"].strip().casefold()
        ):
            raise ValueError("reported_claim_not_literal_utterance")


def reviewed(p):
    good = [
        r
        for r in p.get("reviews", [])
        if r.get("verdict") == "supported"
        and r.get("used_modalities") == ["audio", "video"]
        and r.get("story_used_as_evidence") is False
        and r.get("model")
        and r.get("reason")
        and r.get("observed_speaking") is True
    ]
    return {r.get("stage") for r in good} >= {"propose", "verify"} and not any(
        r.get("verdict") == "contradicted" for r in p.get("reviews", [])
    )


class TemporalEvidenceHypergraph:
    """Compose with the existing visual graph rather than maintain a divergent copy."""

    def __init__(self, graph):
        self.base = copy.deepcopy(graph)
        prior = self.base.pop("multimodal", {})
        self.submissions = copy.deepcopy(prior.get("submissions", []))
        self.journal = copy.deepcopy(prior.get("journal", []))
        if "audio_memory" in self.base:
            validate_audio(self.base["audio_memory"])
        seen = set()
        proposal_ids, evidence = set(), {}
        for packet in self.submissions:
            validate_submission(self.base, packet)
            if packet["submission_id"] in seen:
                raise ValueError("duplicate_multimodal_submission")
            seen.add(packet["submission_id"])
            pids = {p["proposal_id"] for p in packet["bindings"]}
            if pids & proposal_ids:
                raise ValueError("duplicate_multimodal_proposal")
            proposal_ids.update(pids)
            for key, value in packet["evidence_bundles"].items():
                if key in evidence and evidence[key] != value:
                    raise ValueError("evidence_id_collision")
                evidence[key] = value
        submitted, retired = set(), set()
        for n, row in enumerate(self.journal, 1):
            if row["revision"] != n:
                raise ValueError("multimodal_journal_revision_mismatch")
            if row["operation"] == "submit":
                matches = [
                    s for s in self.submissions if s["submission_id"] == row["submission_id"]
                ]
                if (
                    len(matches) != 1
                    or row["submission_id"] in submitted
                    or row["digest"] != digest(matches[0])
                ):
                    raise ValueError("multimodal_journal_submission_mismatch")
                submitted.add(row["submission_id"])
            elif row["operation"] == "revoke":
                known = {
                    p["proposal_id"]
                    for s in self.submissions
                    if s["submission_id"] in submitted
                    for p in s["bindings"]
                }
                if row["proposal_id"] not in known - retired or not row.get("reason"):
                    raise ValueError("invalid_multimodal_revocation")
                retired.add(row["proposal_id"])
            else:
                raise ValueError("unknown_multimodal_operation")
        if submitted != seen:
            raise ValueError("multimodal_journal_incomplete")

    def submit(self, packet):
        validate_submission(self.base, packet)
        known = {p["submission_id"] for p in self.submissions}
        if packet["submission_id"] in known:
            raise ValueError("duplicate_multimodal_submission")
        old_ids = {p["proposal_id"] for s in self.submissions for p in s["bindings"]}
        if old_ids & {p["proposal_id"] for p in packet["bindings"]}:
            raise ValueError("duplicate_multimodal_proposal")
        old_evidence = {k: v for s in self.submissions for k, v in s["evidence_bundles"].items()}
        if any(
            k in old_evidence and old_evidence[k] != v
            for k, v in packet["evidence_bundles"].items()
        ):
            raise ValueError("evidence_id_collision")
        self.submissions.append(copy.deepcopy(packet))
        self.journal.append(
            {
                "revision": len(self.journal) + 1,
                "operation": "submit",
                "submission_id": packet["submission_id"],
                "digest": digest(packet),
            }
        )

    def revoke(self, proposal_id, reason):
        known = {p["proposal_id"] for s in self.submissions for p in s["bindings"]}
        retired = {r["proposal_id"] for r in self.journal if r["operation"] == "revoke"}
        if proposal_id not in known - retired or not reason:
            raise ValueError("invalid_multimodal_revocation")
        self.journal.append(
            {
                "revision": len(self.journal) + 1,
                "operation": "revoke",
                "proposal_id": proposal_id,
                "reason": reason,
            }
        )

    def snapshot(self):
        graph = copy.deepcopy(self.base)
        audio = graph.get("audio_memory", {})
        emap = entity_map(graph)
        retired = {r["proposal_id"] for r in self.journal if r["operation"] == "revoke"}
        proposals = [copy.deepcopy(p) for s in self.submissions for p in s["bindings"]]
        contenders = {}
        for p in proposals:
            if p["proposal_id"] not in retired and reviewed(p):
                contenders.setdefault(p["utterance_id"], set()).add(emap[p["local_instance"]])
        admitted = {}
        for p in proposals:
            if p["proposal_id"] in retired:
                p["status"] = "revoked"
            elif len(contenders.get(p["utterance_id"], [])) > 1:
                p["status"] = "conflict"
            elif reviewed(p):
                p["status"] = "accepted"
                admitted.setdefault(p["utterance_id"], []).append(p)
            else:
                p["status"] = "unresolved"
            p["support_scope"] = "model_reviewed_not_independent_media_audit"
        edges = []
        for u in audio.get("utterances", []):
            links = admitted.get(u["utterance_id"], [])
            locals_ = sorted({p["local_instance"] for p in links})
            paths = identity_paths(graph, locals_) if locals_ else {}
            edges.append(
                {
                    "edge_id": u["utterance_id"] + ":says",
                    "kind": "speech",
                    "predicate": "says",
                    "source_utterance": u["utterance_id"],
                    "local_roles": {"speaker": "audio:" + u["utterance_id"]},
                    "audio_evidence_id": "audio-evidence:" + u["utterance_id"],
                    "value": u["text"],
                    "interval": [u["start"], u["end"]],
                    "resolved_roles": {"speaker": emap[locals_[0]]} if locals_ else {},
                    "unresolved_slots": [] if locals_ else ["speaker"],
                    "evidence_ids": sorted({e for p in links for e in p["evidence_ids"]}),
                    "dependencies": {
                        "speaker_bindings": [p["proposal_id"] for p in links],
                        "identity": sorted({d for path in paths.values() for d in path}),
                        "local_instances": locals_,
                    },
                    "assertion_scope": "utterance_only_not_content_truth",
                    "media_verified": False,
                }
            )
        for s in self.submissions:
            for c in s.get("claims", []):
                edges.append(
                    {
                        **copy.deepcopy(c),
                        "edge_id": s["submission_id"] + ":" + c["claim_id"],
                        "local_roles": {"claimant": "audio:" + c["source_utterance"]},
                        "resolved_roles": {},
                        "unresolved_slots": ["subject", "phase"],
                        "dependencies": {"source_utterance": c["source_utterance"]},
                        "assertion_scope": "reported_claim_not_world_fact",
                        "media_verified": False,
                    }
                )
        graph["architecture"] = "multimodal-rrt-temporal-hypergraph-v1"
        graph["multimodal"] = {
            "schema": SCHEMA,
            "revision": len(self.journal),
            "submissions": copy.deepcopy(self.submissions),
            "journal": copy.deepcopy(self.journal),
            "bindings": proposals,
            "hyperedges": edges,
            "nodes": [
                {
                    "node_id": "audio:" + u["utterance_id"],
                    "kind": "audio_tracklet",
                    "interval": [u["start"], u["end"]],
                    "source_utterance": u["utterance_id"],
                    "single_speaker_model_reviewed": bool(admitted.get(u["utterance_id"])),
                }
                for u in audio.get("utterances", [])
            ],
            "audio_evidence": {
                "audio-evidence:" + u["utterance_id"]: {
                    **u["evidence"],
                    "source_sha256": audio["source"]["sha256"],
                    "modality": "audio",
                    "source_utterance": u["utterance_id"],
                }
                for u in audio.get("utterances", [])
            },
            "evidence_bundles": {
                k: v for s in self.submissions for k, v in s["evidence_bundles"].items()
            },
            "support_scope": "structural_and_model_review_not_ground_truth",
            "visual_fact_ids": [f["fact_id"] for f in graph["facts"]],
        }
        return seal(graph)


def validate_multimodal(graph):
    """Recompute projections: prevent stale, forged, or dangling accepted views."""
    if "multimodal" not in graph:
        return
    expected = TemporalEvidenceHypergraph(graph).snapshot()["multimodal"]
    if expected != graph["multimodal"]:
        raise ValueError("multimodal_projection_mismatch")


def proof_closure(graph, utterance_ids, *, entity_ids=(), require_speaker=False):
    """Retrieve whole joint edges, never independently ranked role fragments."""
    mm = graph.get("multimodal")
    if mm is None:
        return None
    edges = []
    gaps = []
    for edge in mm["hyperedges"]:
        if edge.get("source_utterance") not in utterance_ids:
            continue
        if edge["kind"] != "speech":
            continue
        speaker = edge["resolved_roles"].get("speaker")
        if entity_ids and speaker not in entity_ids:
            gaps.append(
                {"edge_id": edge["edge_id"], "reason": "speaker_scope_unresolved_or_mismatch"}
            )
            continue
        if require_speaker and speaker is None:
            gaps.append({"edge_id": edge["edge_id"], "reason": "speaker_unresolved"})
            continue
        edges.append(copy.deepcopy(edge))
    pids = {p for e in edges for p in e["dependencies"]["speaker_bindings"]}
    bindings = [
        copy.deepcopy(p)
        for p in mm["bindings"]
        if p["proposal_id"] in pids and p["status"] == "accepted"
    ]
    dids = {d for e in edges for d in e["dependencies"]["identity"]}
    decisions = [copy.deepcopy(d) for d in graph["identity_decisions"] if d["decision_id"] in dids]
    endpoints = {i for e in edges for i in e["dependencies"]["local_instances"]}
    endpoints.update(d["proposal"][side] for d in decisions for side in ("left", "right"))
    people = [copy.deepcopy(i) for i in graph["instances"] if i["instance_id"] in endpoints]
    mids = {m for p in bindings for m in p["visual_media_ids"]}
    mids.update(m for d in decisions for m in d["proposal"]["media_ids"])
    bids = {b for e in edges for b in e["evidence_ids"]}
    selected = {e["source_utterance"] for e in edges}
    return {
        "hyperedges": edges,
        "bindings": bindings,
        "identity_decisions": decisions,
        "instances": people,
        "evidence": {m: graph["evidence"][m] for m in mids},
        "evidence_bundles": {b: mm["evidence_bundles"][b] for b in bids},
        "audio_evidence": {
            e["audio_evidence_id"]: mm["audio_evidence"][e["audio_evidence_id"]] for e in edges
        },
        "nodes": [n for n in mm["nodes"] if n["source_utterance"] in selected],
        "reported_claims": [
            c
            for c in mm["hyperedges"]
            if c["kind"] == "reported_claim" and c["source_utterance"] in selected
        ],
        "gaps": gaps,
        "support_scope": mm["support_scope"],
    }
