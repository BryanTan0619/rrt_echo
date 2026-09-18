"""Apply explicit evidence-backed review records; never infer labels from QA."""

from __future__ import annotations

import copy

from ..schema import digest


def quarantine_endpoints(graph, reviews):
    """Keep raw facts in the snapshot while excluding known invalid bindings in retrieval."""
    graph = copy.deepcopy(graph)
    known = {i["instance_id"] for i in graph["instances"]}
    invalid = {}
    for row in reviews:
        if row["instance_id"] not in known or not row.get("evidence_ids") or not row.get("reason"):
            raise ValueError("invalid_endpoint_review")
        if not set(row["evidence_ids"]) <= set(graph["evidence"]):
            raise ValueError("unknown_review_evidence")
        if row["verdict"] == "invalid":
            invalid[row["instance_id"]] = row
    for fact in graph["facts"]:
        bad = {role: invalid[key] for role, key in fact["roles"].items() if key in invalid}
        if bad:
            fact["retrieval_quarantined"] = True
            fact["endpoint_review"] = bad
    graph["endpoint_reviews"] = copy.deepcopy(reviews)
    graph["review_scope"] = (
        "assistant_visual_binding_review; not exhaustive fact truth or automatic benchmark"
    )
    graph.pop("snapshot_id", None)
    graph["snapshot_id"] = digest(graph)
    return graph


def main():
    import argparse
    import json
    from pathlib import Path

    from perception.types import atomic_json
    from .memory import RRTMemory, narrative_projection

    parser = argparse.ArgumentParser(
        description="Replay an explicit visual review journal without model calls."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--endpoint-reviews", type=Path, required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    memory = RRTMemory(args.source_video_id)
    observations = json.loads(args.observations.read_text())
    for result in observations:
        memory.append({**result, "link_proposals": []})
    for record in json.loads(args.journal.read_text()):
        if record["operation"] == "propose":
            memory.propose(record["proposal"])
        elif record["operation"] == "revoke":
            memory.revoke(record["proposal_id"], record["reason"])
        else:
            raise ValueError("unknown_journal_operation")
        if record["revision"] != memory.journal[-1]["revision"]:
            raise ValueError("nonsequential_journal")
        memory.journal[-1] = copy.deepcopy(record)
    graph = quarantine_endpoints(memory.snapshot(), json.loads(args.endpoint_reviews.read_text()))
    atomic_json(args.out / "hypergraph.json", graph)
    atomic_json(args.out / "binding_journal.json", memory.journal)
    atomic_json(args.out / "storyline.json", narrative_projection(graph))
    atomic_json(
        args.out / "diagnostic_acceptance.json",
        {
            "snapshot_id": graph["snapshot_id"],
            "passed": False,
            "reason": "Journal replay is not an exhaustive media acceptance.",
        },
    )
    print(graph["snapshot_id"])


if __name__ == "__main__":
    main()
