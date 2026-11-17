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


# Base-graph key that opts a snapshot into speech-event alignment. Graphs built
# before it existed carry no key and project exactly as they did, so their
# stored multimodal views still validate.
ALIGNMENT_KEY = "speech_event_alignment_policy"
ALIGNMENT_POLICY = "co_temporal_v1"
# Words too common in narration to say anything about which event is meant.
ALIGNMENT_STOPWORDS = frozenset(
    "about after again also back been before being does doing done from going gonna have "
    "here into just like little make more much really right some that them then there "
    "these they this those very want what when where which while will with your".split()
)


def alignment_policy(margin_seconds=1.5):
    if not (isinstance(margin_seconds, (int, float)) and 0 <= margin_seconds <= 10):
        raise ValueError("invalid_speech_event_margin")
    return {"policy": ALIGNMENT_POLICY, "margin_seconds": float(margin_seconds)}


def _alignment_terms(text):
    words = "".join(c.lower() if c.isalnum() else " " for c in text or "").split()
    return {w[:-1] if w.endswith("s") and len(w) > 4 else w
            for w in words if len(w) >= 4 and w not in ALIGNMENT_STOPWORDS}


def speech_event_alignment(graph):
    """Link each ASR utterance to the visual events it overlaps in time.

    The edge records temporal co-occurrence between two independently observed
    records and nothing more: it does not make the voice a participant of the
    event, does not resolve a mentioned name to a role, and does not turn speech
    content into a world fact. Those remain `speaker_of` / `refers_to_event_role`
    decisions with their own admission policies.
    """
    policy = graph.get(ALIGNMENT_KEY)
    audio = graph.get("audio_memory") or {}
    if not policy or not audio.get("utterances"):
        return [], None
    if policy.get("policy") != ALIGNMENT_POLICY:
        raise ValueError("unsupported_speech_event_alignment_policy")
    margin = alignment_policy(policy["margin_seconds"])["margin_seconds"]
    descriptions = {i["instance_id"]: i.get("description", "") for i in graph["instances"]}
    events = [
        f for f in graph["facts"]
        if f["kind"] == "event" and f.get("observed_times") and not f.get("retrieval_quarantined")
    ]
    edges, aligned_events, aligned_speech = [], set(), set()
    for u in audio["utterances"]:
        spoken = _alignment_terms(u["text"])
        for f in events:
            lo, hi = min(f["observed_times"]), max(f["observed_times"])
            if u["end"] < lo - margin or u["start"] > hi + margin:
                continue
            if u["end"] >= lo and u["start"] <= hi:
                relation, gap = "during", 0.0
            elif u["end"] < lo:
                relation, gap = "just_before", lo - u["end"]
            else:
                relation, gap = "just_after", u["start"] - hi
            seen = _alignment_terms(
                " ".join([f["predicate"], f.get("value") or "",
                          *(descriptions.get(i, "") for i in f["roles"].values())])
            )
            edges.append(
                {
                    "edge_id": u["utterance_id"] + ":co_temporal:" + f["fact_id"],
                    "kind": "co_temporal",
                    "predicate": "co_temporal_with",
                    "source_utterance": u["utterance_id"],
                    "local_roles": {"utterance": "audio:" + u["utterance_id"], "event": f["fact_id"]},
                    "audio_evidence_id": "audio-evidence:" + u["utterance_id"],
                    "temporal_relation": relation,
                    "gap_seconds": round(gap, 3),
                    "speech_interval": [u["start"], u["end"]],
                    "event_interval": [lo, hi],
                    # Navigation hint only: a shared word does not show that the
                    # speech is about this event or names one of its participants.
                    "shared_terms": sorted(spoken & seen),
                    "resolved_roles": {},
                    "unresolved_slots": ["speaker", "referent"],
                    "evidence_ids": sorted(f["joint_evidence"]),
                    "dependencies": {
                        "event_fact_ids": [f["fact_id"]],
                        "speaker_bindings": [],
                        "reference_bindings": [],
                        "identity": [],
                        "local_instances": [],
                    },
                    "assertion_scope": "temporal_cooccurrence_only_not_speaker_referent_or_content_truth",
                    "media_verified": False,
                }
            )
            aligned_events.add(f["fact_id"])
            aligned_speech.add(u["utterance_id"])
    # Silence is recorded, not filled: an event with no speech within the margin
    # keeps an explicit gap rather than borrowing a distant utterance.
    report = {
        "policy": ALIGNMENT_POLICY,
        "margin_seconds": margin,
        "events": len(events),
        "events_with_speech": len(aligned_events),
        "utterances": len(audio["utterances"]),
        "utterances_with_event": len(aligned_speech),
        "event_gaps": [
            {"event_fact_id": f["fact_id"], "reason": "no_asr_within_margin"}
            for f in events if f["fact_id"] not in aligned_events
        ],
        "unaligned_utterances": [
            u["utterance_id"] for u in audio["utterances"] if u["utterance_id"] not in aligned_speech
        ],
    }
    return edges, report


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
    facts = {f["fact_id"]: f for f in graph["facts"]}
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
        if p["relation"] not in {"speaker_of", "refers_to_event_role"}:
            raise ValueError("unsupported_multimodal_relation")
        if p["utterance_id"] not in utterances or p["local_instance"] not in people:
            raise ValueError("dangling_multimodal_endpoint")
        u = utterances[p["utterance_id"]]
        mids = p["visual_media_ids"]
        anchors = {r["media_id"] for r in people[p["local_instance"]]["regions"]}
        if not mids or not set(mids) <= anchors:
            raise ValueError("visual_endpoint_missing")
        reviews = p.get("reviews", [])
        if p["relation"] == "speaker_of":
            if not p["evidence_ids"] or not set(p["evidence_ids"]) <= bundles.keys():
                raise ValueError("dangling_multimodal_proof")
            for eid in p["evidence_ids"]:
                b = bundles[eid]
                if b["start"] > u["start"] or b["end"] < u["end"]:
                    raise ValueError("utterance_outside_evidence")
                if p["local_instance"] not in b["candidate_instances"]:
                    raise ValueError("unpresented_speaker_candidate")
            if not any(
                any(
                    bundles[e]["start"] <= graph["evidence"][m]["pts"]
                    * graph["evidence"][m]["time_base"][0]
                    / graph["evidence"][m]["time_base"][1] <= bundles[e]["end"]
                    for e in p["evidence_ids"]
                ) for m in mids
            ):
                raise ValueError("speaker_anchor_outside_interval")
        else:
            fact = facts.get(p.get("event_fact_id"))
            role = p.get("event_role")
            mention = p.get("mention") or {}
            if not fact or fact.get("kind") != "event":
                raise ValueError("referent_event_missing")
            if fact.get("roles", {}).get(role) != p["local_instance"]:
                raise ValueError("referent_role_endpoint_mismatch")
            if mention.get("utterance_id") != p["utterance_id"] or not mention.get("text"):
                raise ValueError("invalid_named_mention")
            if mention["text"].casefold() not in u["text"].casefold():
                raise ValueError("mention_not_literal_utterance_span")
            event_mids = set(fact.get("joint_evidence", [])) | {
                m for values in fact.get("evidence_by_slot", {}).values() for m in values
            }
            if not set(mids) <= event_mids:
                raise ValueError("referent_visual_evidence_outside_event")
            if p.get("evidence_ids"):
                raise ValueError("referent_uses_visual_and_asr_provenance_not_speaker_bundle")
        for r in reviews:
            if r.get("local_instance") != p["local_instance"]:
                raise ValueError("review_endpoint_mismatch")
            if r.get("utterance_id") != p["utterance_id"]:
                raise ValueError("review_utterance_mismatch")
            if p["relation"] == "speaker_of" and r.get("evidence_id") not in p["evidence_ids"]:
                raise ValueError("review_evidence_mismatch")
            if p["relation"] == "refers_to_event_role" and (
                r.get("event_fact_id") != p["event_fact_id"]
                or r.get("event_role") != p["event_role"]
            ):
                raise ValueError("referent_review_scope_mismatch")
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
    if p.get("admission_policy") == "independent_av_sync_v1":
        # Versioned gate for new Omni-origin speaker proposals. A repeated
        # judgement by the same model is not independent synchronization
        # evidence; an external active-speaker check is required before this
        # policy can admit the proposed link. Preserve it as unresolved.
        return False
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


def referent_reviewed(p):
    if p.get("admission_policy") == "named_referent_v2" and (
        (p.get("mention", {}).get("on_screen_spelling") or {}).get("match") == "phonetic"
    ):
        return False
    good = [
        r for r in p.get("reviews", [])
        if r.get("verdict") == "supported"
        and r.get("used_modalities") == ["asr", "video"]
        and r.get("query_blind") is True
        and r.get("same_occurrence") is True
        and r.get("same_role") is True
        and r.get("candidates_distinguished") is True
        and (p.get("admission_policy") != "named_referent_v2" or (
            r.get("mention_is_person_name") is True
            and r.get("name_refers_to_role") is True
        ))
        and r.get("model") and r.get("reason")
    ]
    return bool(good) and not any(r.get("verdict") == "contradicted" for r in p.get("reviews", []))


def binding_reviewed(p):
    return referent_reviewed(p) if p.get("relation") == "refers_to_event_role" else reviewed(p)

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
            key = (p["relation"], p["utterance_id"])
            if p["relation"] == "refers_to_event_role":
                key += (p["mention"]["text"].casefold(), p["event_fact_id"], p["event_role"])
            if p["proposal_id"] not in retired and binding_reviewed(p):
                contenders.setdefault(key, set()).add(p["local_instance"])
        admitted = {}
        for p in proposals:
            key = (p["relation"], p["utterance_id"])
            if p["relation"] == "refers_to_event_role":
                key += (p["mention"]["text"].casefold(), p["event_fact_id"], p["event_role"])
            if p["proposal_id"] in retired:
                p["status"] = "revoked"
            elif len(contenders.get(key, [])) > 1:
                p["status"] = "conflict"
            elif binding_reviewed(p):
                p["status"] = "accepted"
                admitted.setdefault((p["relation"], p["utterance_id"]), []).append(p)
            else:
                p["status"] = "unresolved"
            p["support_scope"] = "model_reviewed_not_independent_media_audit"
        edges = []
        for u in audio.get("utterances", []):
            links = admitted.get(("speaker_of", u["utterance_id"]), [])
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
        for u in audio.get("utterances", []):
            for p in admitted.get(("refers_to_event_role", u["utterance_id"]), []):
                paths = identity_paths(graph, [p["local_instance"]])
                edges.append(
                    {
                        "edge_id": p["proposal_id"] + ":refers",
                        "kind": "reference",
                        "predicate": "refers_to_event_role",
                        "source_utterance": u["utterance_id"],
                        "mention": copy.deepcopy(p["mention"]),
                        "local_roles": {
                            "referent": p["local_instance"],
                            "event": p["event_fact_id"],
                            "role": p["event_role"],
                        },
                        "resolved_roles": {"referent": emap[p["local_instance"]]},
                        "unresolved_slots": [],
                        "evidence_ids": list(p["visual_media_ids"]),
                        "dependencies": {
                            "reference_bindings": [p["proposal_id"]],
                            "speaker_bindings": [],
                            "identity": sorted({d for path in paths.values() for d in path}),
                            "local_instances": [p["local_instance"]],
                            "event_fact_ids": [p["event_fact_id"]],
                        },
                        "assertion_scope": "event_scoped_named_reference_not_global_identity",
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
        alignment_edges, alignment = speech_event_alignment(graph)
        edges.extend(alignment_edges)
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
                    "single_speaker_model_reviewed": bool(admitted.get(("speaker_of", u["utterance_id"]))),
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
        if alignment is not None:
            graph["multimodal"]["speech_event_alignment"] = alignment
        return seal(graph)


def submit_observation_referents(graph):
    """Commit query-blind local referent proposals through the ECHO journal."""
    memory = TemporalEvidenceHypergraph(graph)
    audio = graph.get("audio_memory")
    if not audio:
        return memory.snapshot()
    utterances = {u["utterance_id"] for u in audio.get("utterances", [])}
    losses = []
    for proposal in graph.get("referent_proposals", []):
        if proposal.get("utterance_id") not in utterances:
            losses.append({
                "stage": "echo_admission",
                "proposal_id": proposal.get("proposal_id"),
                "reason": "referent_utterance_missing",
            })
            continue
        packet = {
            "schema": SCHEMA,
            "submission_id": "observation:" + proposal["proposal_id"],
            "video_id": audio["source_video_id"],
            "source_sha256": audio["source"]["sha256"],
            "evidence_bundles": {},
            "bindings": [copy.deepcopy(proposal)],
            "claims": [],
            "gaps": [],
            "task_id": proposal["proposal_id"],
        }
        try:
            memory.submit(packet)
        except (KeyError, ValueError, TypeError) as error:
            losses.append({
                "stage": "echo_admission",
                "proposal_id": proposal.get("proposal_id"),
                "reason": str(error)[:300],
            })
    snapshot = memory.snapshot()
    snapshot["referent_admission_losses"] = losses
    return seal(snapshot)
