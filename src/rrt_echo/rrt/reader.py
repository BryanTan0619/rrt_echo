"""Graph-only QA following an explicit media-acceptance artifact."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from perception.types import atomic_json
from ..evaluation import audit_assertions, summarize
from perception.reader_prompts import AUDIT
from perception.reader_vlm import VisionClient
from ..retrieval import Query, retrieve
from ..runtime import Deadline
from ..schema import digest
from ..scoped_identity import identity_paths
from ..storage import read_graph
from .audio import retrieve_audio
from .binding_scope import binding_view, state_sequence_view, video_time_ranges
from .hypergraph import proof_closure


def payload_for(
    graph,
    question,
    top_k=40,
    max_bytes=150000,
    options=None,
    speaker_entity_ids=(),
    require_speaker=False,
):
    # Options are separate retrieval hypotheses, never established graph claims.
    stem = re.split(r"\n\s*[A-Z][).:]\s+", question, maxsplit=1)[0]
    alternatives = options or dict(re.findall(r"^\s*([A-Z])[).:]\s*(.+)$", question, re.MULTILINE))
    if set(alternatives) == {"True", "False"}:
        alternatives = {}
    ranges = video_time_ranges(stem)
    if alternatives:
        quota = max(1, top_k // (len(alternatives) + 1))
        scopes = {
            "question": stem,
            **{f"option_{k}": stem + " Hypothesis: " + v for k, v in alternatives.items()},
        }
        parts = {
            key: retrieve(graph, Query(text, time_ranges=ranges), top_k=quota, max_bytes=max_bytes)
            for key, text in scopes.items()
        }
        ids = tuple(
            dict.fromkeys(f["fact_id"] for part in parts.values() for f in part.get("facts", []))
        )
        payload = (
            retrieve(
                graph,
                Query(stem, time_ranges=ranges, fact_ids=ids),
                top_k=max(1, len(ids)),
                max_bytes=max_bytes,
            )
            if ids
            else {
                "facts": [],
                "instances": [],
                "evidence": {},
                "identity_decisions": [],
                "gap": "no_scope_evidence",
            }
        )
        delivered = {f["fact_id"] for f in payload.get("facts", [])}
        payload["retrieval_hypotheses"] = {
            key: [f["fact_id"] for f in part.get("facts", []) if f["fact_id"] in delivered]
            for key, part in parts.items()
        }
        payload["hypotheses_are_claims"] = False
    else:
        payload = retrieve(graph, Query(stem, time_ranges=ranges), top_k=top_k, max_bytes=max_bytes)
    payload["query_time_windows"] = [list(r) for r in ranges]
    payload["attribute_bindings"] = binding_view(payload)
    payload["state_sequences"] = state_sequence_view(payload, graph)
    payload["state_sequence_rules"] = (
        "A state_sequence lists observed states of one entity ordered by time. "
        "Order is observation order, not proof that one state turned into another. "
        "Do not infer a transition across an unobserved gap."
    )
    payload["binding_rules"] = (
        "Attribute values belong only to their cited local surface/owner at the observed time. Distinct unresolved entity IDs are not evidence of different identities. Do not transfer values between owners or events."
    )
    payload["scope_gaps"] = [
        {"time_window": list(r), "reason": "no_observed_facts_in_window"}
        for r in ranges
        if not any(r[0] <= t <= r[1] for f in payload.get("facts", []) for t in f["observed_times"])
    ]
    owners = {
        x for f in payload.get("facts", []) for x in f.get("dependencies", {}).get("ownership", [])
    }
    payload["binding_links"] = [
        p for p in graph.get("binding_links", []) if p["proposal_id"] in owners
    ]
    endpoints = {p["target"] for p in payload["binding_links"]}
    received = {i["instance_id"] for i in payload.get("instances", [])}
    payload.setdefault("instances", []).extend(
        i for i in graph["instances"] if i["instance_id"] in endpoints - received
    )
    # The owner's identity proof is required as well as the part_of edge.
    dids = {d for path in identity_paths(graph, endpoints).values() for d in path}
    present = {d["decision_id"] for d in payload.get("identity_decisions", [])}
    extra = [d for d in graph["identity_decisions"] if d["decision_id"] in dids - present]
    payload.setdefault("identity_decisions", []).extend(extra)
    proof_endpoints = {d["proposal"][side] for d in extra for side in ("left", "right")}
    received = {i["instance_id"] for i in payload["instances"]}
    payload["instances"].extend(
        i for i in graph["instances"] if i["instance_id"] in proof_endpoints - received
    )
    # Include media metadata for ownership endpoints and proposal source references.
    mids = {m for p in payload["binding_links"] for m in p["evidence_ids"]}
    mids.update(m for d in extra for m in d["proposal"]["media_ids"])
    mids.update(r["media_id"] for i in payload["instances"] for r in i["regions"])
    payload.setdefault("evidence", {}).update(
        {m: graph["evidence"][m] for m in mids if m in graph["evidence"]}
    )
    speech = retrieve_audio(
        graph,
        question + " " + " ".join(alternatives.values()),
        ranges=ranges,
        top_k=min(16, max(2, top_k)),
    )
    if speech is not None:
        closure = proof_closure(
            graph,
            {u["utterance_id"] for u in speech["utterances"]},
            entity_ids=speaker_entity_ids,
            require_speaker=require_speaker,
        )
        if closure is not None:
            delivered = {e["source_utterance"] for e in closure["hyperedges"]}
            speech["utterances"] = [
                u for u in speech["utterances"] if u["utterance_id"] in delivered
            ]
            payload["multimodal_hyperedges"] = closure["hyperedges"]
            payload["speech_audio_evidence"] = closure["audio_evidence"]
            payload["audio_nodes"] = closure["nodes"]
            payload["speaker_bindings"] = closure["bindings"]
            payload["audio_video_evidence"] = closure["evidence_bundles"]
            payload["reported_claims"] = closure["reported_claims"]
            payload["multimodal_scope_gaps"] = closure["gaps"]
            for key, field in [("instances", "instance_id"), ("identity_decisions", "decision_id")]:
                present = {r[field] for r in payload.setdefault(key, [])}
                payload[key].extend(r for r in closure[key] if r[field] not in present)
            payload.setdefault("evidence", {}).update(closure["evidence"])
            # Old voice clusters are not authoritative person identities.
            for u in speech["utterances"]:
                u["voice_cluster_candidate"] = u.pop("speaker_id", None)
            speech["speakers"] = []
        payload["audio_memory"] = speech
    payload.pop("payload_id", None)
    if len(json.dumps(payload, ensure_ascii=False).encode()) > max_bytes:
        return {
            "facts": [],
            "instances": [],
            "binding_links": [],
            "gap": "ownership_closure_budget_exhausted",
        }
    payload["payload_id"] = digest(payload)
    return payload


def compact_payload(payload):
    """Lossless ID aliases for the reader; canonical proof stays in saved payload."""
    aliases = {}

    def add(values, prefix):
        for value in sorted(set(values)):
            if value not in aliases:
                aliases[value] = prefix + str(len(aliases))

    add([u["utterance_id"] for u in payload.get("audio_memory", {}).get("utterances", [])], "S")
    add([s["speaker_id"] for s in payload.get("audio_memory", {}).get("speakers", [])], "V")
    add([e["edge_id"] for e in payload.get("multimodal_hyperedges", [])], "H")
    add([e["edge_id"] for e in payload.get("reported_claims", [])], "C")
    add([p["proposal_id"] for p in payload.get("speaker_bindings", [])], "P")
    add(list(payload.get("audio_video_evidence", {})), "B")
    add(list(payload.get("speech_audio_evidence", {})), "A")
    add([n["node_id"] for n in payload.get("audio_nodes", [])], "N")
    add(
        [v for e in payload.get("multimodal_hyperedges", []) for v in e["resolved_roles"].values()],
        "E",
    )
    add([f["fact_id"] for f in payload.get("facts", [])], "F")
    add([i["instance_id"] for i in payload.get("instances", [])], "I")
    add(list(payload.get("evidence", {})), "M")
    add([d["decision_id"] for d in payload.get("identity_decisions", [])], "D")
    add(
        [d["proposal"]["proposal_id"] for d in payload.get("identity_decisions", [])]
        + [p["proposal_id"] for p in payload.get("binding_links", [])],
        "L",
    )
    add(
        [r["entity_id"] for f in payload.get("facts", []) for r in f["resolved_roles"].values()]
        + [
            p["owner_entity"]
            for f in payload.get("facts", [])
            for p in f.get("owner_projections", {}).values()
        ],
        "E",
    )
    add([f["dependencies"]["observation"] for f in payload.get("facts", [])], "O")

    def remap(value):
        if isinstance(value, str):
            return aliases.get(value, value)
        if isinstance(value, list):
            return [remap(x) for x in value]
        if isinstance(value, dict):
            return {aliases.get(k, k): remap(v) for k, v in value.items()}
        return value

    data = copy.deepcopy(
        {k: v for k, v in payload.items() if k not in {"snapshot_id", "payload_id"}}
    )

    # Preserve typed claims and all proof edges, summarize repetitive structural
    # status metadata. The complete canonical payload is saved with each answer.
    def support_status(value):
        if isinstance(value, dict):
            return {
                k: support_status(v)
                for k, v in value.items()
                if k not in {"assessment", "media_audit_required", "evidence_ids"}
            }
        return value

    for f in data.get("facts", []):
        f["support"] = support_status(f["support"])
        f.pop("source_fact_digest", None)
        f.pop("proof_view", None)
    data["support_scope"] = "structural statuses only; not independent video verification"

    # Source file paths/hashes are trace metadata, not facts supplied to the reader.
    data["evidence"] = {
        mid: {"seconds": m.get("seconds", m["pts"] * m["time_base"][0] / m["time_base"][1])}
        for mid, m in payload.get("evidence", {}).items()
    }
    data["audio_video_evidence"] = {
        k: {f: v[f] for f in ("start", "end", "modalities", "candidate_instances")}
        for k, v in payload.get("audio_video_evidence", {}).items()
    }
    return remap(data), {v: k for k, v in aliases.items()}


def restore_audit_ids(audit, aliases):
    return [
        {**a, "evidence_chain": [aliases.get(x, x) for x in a.get("evidence_chain", [])]}
        for a in audit.get("assertions", [])
    ]


def question_options(question):
    if str(question.get("question_type", "")).casefold() in {"true/false", "true_false", "boolean"}:
        return {"True": "True", "False": "False"}
    options = question.get("options") or dict(
        re.findall(r"^\s*([A-Z])[).:]\s*(.+)$", question["question"], re.MULTILINE)
    )
    if isinstance(options, list):
        options = {chr(65 + i): x for i, x in enumerate(options)}
    if not options:
        raise ValueError("question_options_missing")
    return options


def evaluate(
    graph_path,
    questions_path,
    acceptance_path,
    *,
    out,
    model,
    base_url,
    diagnostic=False,
    workers=1,
):
    graph = read_graph(Path(graph_path))
    acceptance = json.loads(Path(acceptance_path).read_text())
    if acceptance.get("snapshot_id") != graph["snapshot_id"]:
        raise ValueError("acceptance_snapshot_mismatch")
    if not diagnostic and acceptance.get("passed") is not True:
        raise ValueError("media_acceptance_required_for_this_snapshot")
    target = graph.get("source_video_id") or graph["video_id"]
    rows = [
        json.loads(line) for line in Path(questions_path).read_text().splitlines() if line.strip()
    ]
    rows = [
        r
        for r in rows
        if str(r.get("video_id", r.get("video", ""))).removesuffix(".mp4").split("/")[-1] == target
    ]
    if not rows:
        raise ValueError("no_matching_video_questions")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if not 1 <= workers <= 4:
        raise ValueError("reader_workers_must_be_1_to_4")
    results = []
    all_calls = []

    def process(item):
        n, q = item
        client = VisionClient(model=model, base_url=base_url, max_tokens=1800, request_timeout=300)
        options = question_options(q)
        row = {
            "question_id": q.get("question_id", str(n)),
            "type": q.get("hallucination_type", q.get("type", "unspecified")),
            "correct": False,
        }
        try:
            for top_k in (40, 32, 24, 16, 8, 4):
                payload = payload_for(
                    graph, q["question"], top_k=top_k, max_bytes=1000000, options=options
                )
                compact, aliases = compact_payload(payload)
                if len(json.dumps(compact, ensure_ascii=False)) <= 65000:
                    break
            else:
                raise ValueError("reader_context_budget_exhausted")
            row["reader_payload"] = payload
            row["reader_aliases"] = aliases
            row["retrieval_top_k"] = top_k
            body = {"question": q["question"], "options": options, "graph": compact}
            answer = client.complete(
                'Answer using only the supplied graph. Preserve event roles, time, identity and ownership. Select exactly one option even if evidence is insufficient; do not assert it is supported merely because selected. Return JSON {"answer":"option ID"}.\n'
                + json.dumps(body),
                deadline=Deadline(300),
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string", "enum": list(options)}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            )
            selected = str(answer.get("answer", "")).strip()
            if set(options) == {"True", "False"}:
                selected = selected.capitalize()
            if selected not in options:
                raise ValueError("invalid_option")
            expected = str(q.get("ans", q.get("answer"))).strip()
            if set(options) == {"True", "False"}:
                expected = expected.capitalize()
            row.update(answer=selected, correct=selected == expected)
            audit = client.complete(
                AUDIT
                + "\nClassify each assertion: utterance_content (what was said), speaker_identity (who said it), or world_state (what actually happened/is true). Spoken claims are not world facts. Speaker identity must cite the corresponding bound speech hyperedge, not an anonymous voice cluster or the utterance alone.\n"
                + "\n"
                + json.dumps(
                    {
                        "question": re.split(r"\n\s*[A-Z][).:]\s+", q["question"], maxsplit=1)[0],
                        "output_answer": options[selected],
                        "graph": compact,
                    }
                ),
                deadline=Deadline(300),
                output_schema={
                    "type": "object",
                    "properties": {
                        "assertions": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "assertion": {"type": "string", "maxLength": 400},
                                    "claim_type": {
                                        "type": "string",
                                        "enum": [
                                            "utterance_content",
                                            "speaker_identity",
                                            "world_state",
                                        ],
                                    },
                                    "support_verdict": {
                                        "type": "string",
                                        "enum": ["supported", "contradicted", "insufficient"],
                                    },
                                    "evidence_chain": {
                                        "type": "array",
                                        "maxItems": 8,
                                        "items": {"type": "string", "maxLength": 32},
                                    },
                                },
                                "required": [
                                    "assertion",
                                    "claim_type",
                                    "support_verdict",
                                    "evidence_chain",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["assertions"],
                    "additionalProperties": False,
                },
            )
            row.update(audit_assertions(restore_audit_ids(audit, aliases), payload))
        except Exception as e:
            row["error"] = str(e)
        atomic_json(out / "question_calls" / f"{n:03d}.json", client.calls)
        return row, client.calls

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, calls in pool.map(process, enumerate(rows)):
            results.append(row)
            all_calls.extend(calls)
            atomic_json(out / "results.json", results)
    atomic_json(out / "summary.json", summarize(results))
    atomic_json(out / "calls.json", all_calls)
    atomic_json(
        out / "manifest.json",
        {
            "snapshot_id": graph["snapshot_id"],
            "model": model,
            "reader_video_access": False,
            "caption_access": False,
            "questions": len(rows),
            "media_acceptance": acceptance,
            "diagnostic": diagnostic,
            "workers": workers,
            "snapshot_media_accepted": acceptance.get("passed") is True,
            "quality_claim": "diagnostic_only" if diagnostic else "accepted_snapshot_evaluation",
        },
    )
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "questions", "acceptance", "out", "model"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help="Evaluate an unaccepted snapshot; never implies media acceptance",
    )
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    evaluate(
        a.graph,
        a.questions,
        a.acceptance,
        out=a.out,
        model=a.model,
        base_url=a.base_url,
        diagnostic=a.diagnostic,
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
