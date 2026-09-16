"""Forced answers and independent support: correct guesses are not evidence."""

from __future__ import annotations

from collections import defaultdict

from .schema import VERDICTS


def audit_assertions(assertions: list[dict], payload: dict) -> dict:
    received = {f["fact_id"] for f in payload.get("facts", [])}
    received.update(
        u["utterance_id"] for u in payload.get("audio_memory", {}).get("utterances", [])
    )
    received.update(d["decision_id"] for d in payload.get("identity_decisions", []))
    received.update(r["relation_id"] for r in payload.get("relations", []))
    received.update(
        p["proposal_id"] for p in payload.get("binding_links", []) if p.get("status") == "accepted"
    )
    received.update(
        p["proposal_id"] for p in payload.get("speaker_bindings", []) if p["status"] == "accepted"
    )
    for edge in payload.get("multimodal_hyperedges", []):
        deps = edge["dependencies"]
        if (
            edge["source_utterance"] in received
            and edge["audio_evidence_id"] in payload.get("speech_audio_evidence", {})
            and set(deps["speaker_bindings"] + deps["identity"]) <= received
            and set(edge["evidence_ids"]) <= payload.get("audio_video_evidence", {}).keys()
        ):
            received.add(edge["edge_id"])
    received.update(
        c["edge_id"]
        for c in payload.get("reported_claims", [])
        if c["source_utterance"] in received and c.get("world_fact") is False
    )
    rows = []
    for assertion in assertions:
        verdict = assertion.get("support_verdict", "insufficient")
        chain = assertion.get("evidence_chain", [])
        valid = (
            isinstance(chain, list)
            and bool(chain)
            and all(isinstance(x, str) and x in received for x in chain)
        )
        claim_type = assertion.get("claim_type")
        if valid and verdict == "supported":
            if claim_type == "world_state":
                # An utterance or a speaker identity chain cannot establish its content as true.
                visual_facts = {f["fact_id"] for f in payload.get("facts", [])}
                if not set(chain) & visual_facts:
                    valid = False
            elif claim_type == "speaker_identity":
                bound_edges = {
                    e["edge_id"]
                    for e in payload.get("multimodal_hyperedges", [])
                    if e["resolved_roles"].get("speaker") and e["edge_id"] in received
                }
                if not set(chain) & bound_edges:
                    valid = False
        if verdict not in VERDICTS or not valid:
            verdict, chain = "insufficient", []
        rows.append(
            {
                "assertion": str(assertion.get("assertion", "")),
                **({"claim_type": claim_type} if claim_type else {}),
                "support_verdict": verdict,
                "evidence_chain": chain,
            }
        )
    values = [r["support_verdict"] for r in rows]
    verdict = (
        "contradicted"
        if "contradicted" in values
        else "supported"
        if values and all(v == "supported" for v in values)
        else "insufficient"
    )
    return {
        "support_verdict": verdict,
        "assertions": rows,
        "scope": "graph_audit_not_media_verified",
        "strict_support": None,
    }


def summarize(rows: list[dict]) -> dict:
    def metrics(group):
        n = len(group)
        audited = all(r.get("media_audit", {}).get("completed") is True for r in group)
        return {
            "questions": n,
            "correct": sum(r.get("correct", False) for r in group),
            "accuracy": sum(r.get("correct", False) for r in group) / n if n else None,
            "execution_failures": sum(bool(r.get("error")) for r in group),
            "correct_with_support": (
                sum(
                    bool(r.get("correct")) and r["media_audit"].get("answer_support") == "supported"
                    for r in group
                )
                / n
                if n and audited
                else None
            ),
            "support_audit_complete": bool(n and audited),
            "answer_support": {
                v: sum(r.get("support_verdict") == v for r in group) for v in sorted(VERDICTS)
            },
        }

    groups = defaultdict(list)
    for row in rows:
        groups[row.get("type", "unspecified")].append(row)
    return {
        **metrics(rows),
        "by_type": {key: metrics(group) for key, group in sorted(groups.items())},
    }
