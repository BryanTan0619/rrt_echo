"""Graph-only retrieval with explicit proof closure and delivery budgets."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from .rrt.binding_scope import text_companions
from .rrt.event_content import event_content_closure
from .schema import digest
from .scoped_identity import identity_paths


@dataclass(frozen=True)
class Query:
    text: str
    required_dimensions: tuple[str, ...] = ("local_joint",)
    entity_ids: tuple[str, ...] = ()
    time_ranges: tuple[tuple[float, float], ...] = ()
    fact_ids: tuple[str, ...] = ()


def terms(text):
    """Search normalization only; never rewrites the stored claim or identity."""
    aliases = {
        "infant": "baby",
        "babies": "baby",
        "automobile": "car",
        "vehicle": "car",
        "holding": "hold",
        "held": "hold",
        "holds": "hold",
        "carrying": "carry",
        "carried": "carry",
        "carries": "carry",
        "writing": "write",
        "written": "write",
        "wrote": "write",
        "looking": "look",
        "looked": "look",
        "exiting": "exit",
        "exited": "exit",
        "attached": "attach",
        "attaches": "attach",
        "attaching": "attach",
        "hanging": "hang",
        "hangs": "hang",
    }
    stop = {
        "a",
        "an",
        "the",
        "is",
        "was",
        "are",
        "were",
        "of",
        "in",
        "on",
        "to",
        "and",
        "who",
        "what",
        "which",
        "does",
        "did",
        "do",
        "with",
        "at",
        "it",
        "his",
        "her",
        "he",
        "she",
        "they",
        "him",
        "them",
        "their",
        "its",
        "himself",
        "herself",
        "itself",
        "themselves",
        "how",
        "when",
        "where",
        "why",
        "for",
        "by",
        "from",
        "as",
        "be",
        "being",
        "been",
        "a",
        "an",
    }
    return {aliases.get(w, w) for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in stop}


def _checks(fact, required):
    missing = []
    for dimension in required:
        value = fact["support"].get(dimension, {})
        statuses = [value] if "status" in value else list(value.values())
        if not statuses or any(v.get("status") != "supported" for v in statuses):
            missing.append(dimension)
    return {
        "missing_dimensions": missing,
        "structurally_supported": not missing,
        "media_audit_required": True,
    }


def retrieve(graph: dict, query: Query, *, top_k=8, max_bytes=100_000) -> dict:
    if top_k < 1 or max_bytes < 256:
        raise ValueError("invalid retrieval budget")
    # Options are reader input, not a list of alternative retrieval targets.
    question_stem = re.split(r"\n\s*[A-Z][).:]\s+", query.text, maxsplit=1)[0]
    words = terms(question_stem)
    available_facts = [f for f in graph["facts"] if not f.get("retrieval_quarantined")]
    descriptions = {i["instance_id"]: i.get("description", "") for i in graph["instances"]}
    documents = {}
    for f in available_facts:
        local = " ".join(descriptions.get(i, "") for i in f["roles"].values())
        documents[f["fact_id"]] = terms(
            " ".join([f["predicate"], f.get("value") or "", *f["roles"], local])
        )
    entity_words = {}
    alias_support = {}
    named_entities = set()
    for entity in graph["entities"]:
        entity_words[entity["entity_id"]] = terms(
            " ".join(descriptions.get(i, "") for i in entity["local_instances"])
        )
    by_instance = {i["instance_id"]: i for i in graph["instances"]}
    for f in available_facts:
        if f["kind"] not in {"text", "attribute"} or "owner" not in f["roles"]:
            continue
        owner = f["roles"]["owner"]
        projection = f.get("owner_projections", {}).get("owner")
        if projection:
            entity = projection["owner_entity"]
        elif by_instance[owner]["kind"] in {"person", "object"}:
            entity = f["resolved_roles"]["owner"]["entity_id"]
        else:
            continue
        value = f.get("value") or ""
        name_match = re.search(r"\bmy name is\s+([a-z][a-z \-]*)", value, re.I)
        literal_name = (
            name_match.group(1)
            if name_match
            else value
            if f["predicate"] in {"name", "person_name", "name_tag"}
            else ""
        )
        if terms(literal_name) & terms(question_stem.split(" Hypothesis:", 1)[0]):
            named_entities.add(entity)
        value_words = terms(value)
        entity_words.setdefault(entity, set()).update(value_words)
        if words & value_words:
            alias_support.setdefault(entity, []).append(f)
    frequency = Counter(w for doc in documents.values() for w in doc)
    weights = {w: math.log(1 + len(documents) / (1 + n)) for w, n in frequency.items()}
    ranked = []
    late = bool(
        re.search(
            r"\bat the end\b|\bend of (?:the )?(?:video|film|story)\b|\blast seen\b",
            question_stem,
            re.I,
        )
    )
    early = bool(
        re.search(
            r"\bat the beginning\b|\bstart of (?:the )?(?:video|film|story)\b", question_stem, re.I
        )
    )
    last_observed = (
        max((max(f["observed_times"], default=0) for f in available_facts), default=1) or 1
    )
    for fact in available_facts:
        if query.fact_ids and fact["fact_id"] not in query.fact_ids:
            continue
        if query.time_ranges and not any(
            lo <= t <= hi for lo, hi in query.time_ranges for t in fact["observed_times"]
        ):
            continue
        entities = {r["entity_id"] for r in fact["resolved_roles"].values()}
        entities.update(p["owner_entity"] for p in fact.get("owner_projections", {}).values())
        if named_entities and not entities.intersection(named_entities):
            continue
        if query.entity_ids and not entities.intersection(query.entity_ids):
            continue
        score = sum(weights.get(w, 0) for w in words & documents[fact["fact_id"]])
        score += 0.7 * sum(weights.get(w, 0) for w in words & terms(fact["predicate"]))
        inherited = set().union(*(entity_words.get(e, set()) for e in entities))
        score += 0.35 * sum(weights.get(w, 0) for w in words & inherited)
        if late or early:
            position = max(fact["observed_times"], default=0) / last_observed
            score += 3 * (position if late else 1 - position)
        ranked.append((-score, fact["fact_id"], fact))
    ranked.sort(key=lambda row: row[:2])
    chosen = []
    gaps = []

    def bundle(facts):
        paths = identity_paths(graph, {i for f in facts for i in f["roles"].values()})
        projected = []
        for original in facts:
            f = copy.deepcopy(original)
            f["source_fact_digest"] = digest(original)
            f["proof_view"] = "paths_to_frozen_entity_representatives"
            f["dependencies"]["identity"] = sorted(
                {d for i in f["roles"].values() for d in paths[i]}
            )
            for role, endpoint in f["roles"].items():
                if role in f["support"].get("global_identity", {}):
                    f["support"]["global_identity"][role]["evidence_ids"] = paths[endpoint]
            projected.append(f)
        facts = projected
        fids = {f["fact_id"] for f in facts}
        refs = {ref for f in facts for ref in f["roles"].values()}
        dids = {d for f in facts for d in f["dependencies"]["identity"]}
        decisions = [d for d in graph["identity_decisions"] if d["decision_id"] in dids]
        # Identity paths may involve intermediate instances, which need grounding.
        for d in decisions:
            refs.update((d["proposal"]["left"], d["proposal"]["right"]))
        instances = [i for i in graph["instances"] if i["instance_id"] in refs]
        mids = {m for f in facts for values in f["evidence_by_slot"].values() for m in values}
        mids.update(
            m
            for f in facts
            for m in (*f["joint_evidence"], *f["boundary_evidence"])
            if isinstance(m, str)
        )
        mids.update(m for d in decisions for m in d["proposal"]["media_ids"])
        mids.update(r["media_id"] for i in instances for r in i["regions"])
        hyperedges = [
            e for e in graph.get("event_hyperedges", []) if set(e["source_fact_ids"]) <= fids
        ]
        mids.update(m for e in hyperedges for m in e["evidence_ids"])
        missing = sorted(mids - set(graph["evidence"]))
        return {
            "snapshot_id": graph["snapshot_id"],
            "facts": facts,
            "event_hyperedges": hyperedges,
            "instances": instances,
            "identity_decisions": decisions,
            "relations": [r for r in graph["relations"] if {r["source"], r["target"]} <= fids],
            "evidence": {m: graph["evidence"][m] for m in sorted(mids) if m in graph["evidence"]},
            "checks": {f["fact_id"]: _checks(f, query.required_dimensions) for f in facts},
            "missing_references": missing,
        }

    for _, _, fact in ranked[:top_k]:
        # If a name/value retrieves an event through an entity, deliver the
        # actual attribute/text fact as evidence too, not just the index hint.
        entity_ids = {r["entity_id"] for r in fact["resolved_roles"].values()}
        companions = [f for e in entity_ids for f in alias_support.get(e, [])]
        companions.extend(text_companions(fact, available_facts))
        candidate_facts = list({f["fact_id"]: f for f in [*chosen, fact, *companions]}.values())
        occurrence_facts = event_content_closure([fact], available_facts, graph)
        candidate_facts = list(
            {f["fact_id"]: f for f in [*candidate_facts, *occurrence_facts]}.values()
        )
        candidate = bundle(candidate_facts)
        # Reserve space for budget/gap metadata and hash, not only raw facts.
        if len(json.dumps(candidate, ensure_ascii=False).encode()) + 256 > max_bytes:
            gaps.append(fact["fact_id"])
        else:
            chosen = candidate_facts
    result = bundle(chosen)
    result["omitted_for_budget"] = gaps
    result["closure_complete"] = not result["missing_references"]
    result["delivery_complete"] = not gaps and result["closure_complete"]
    result["payload_id"] = digest(result)
    # Gap IDs can themselves be large. Never silently exceed a byte budget.
    if len(json.dumps(result, ensure_ascii=False).encode()) > max_bytes:
        return {
            "snapshot_id": graph["snapshot_id"],
            "facts": [],
            "delivery_complete": False,
            "closure_complete": False,
            "gap": "payload_budget_exhausted",
        }
    return result
