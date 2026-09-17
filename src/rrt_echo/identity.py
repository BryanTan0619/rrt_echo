"""Revisable endpoint correspondences; descriptions and names never merge people."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Mapping

from .schema import Instance, digest

# Discriminative inscribed identifiers (vest/bib/jersey numbers, tags, plates).
# These are physically inscribed, designed to be unique per entity, and far more
# reliable than shared appearance attributes (e.g. everyone wearing the same
# uniform). Leading zeros are normalized so "05" and "5" compare equal.
_IDENTIFIER_PATTERN = re.compile(
    r"(?:number|vest|bib|jersey|player|contestant|racer)\s*#?\s*(\d+)"
    r"|\b#\s*(\d+)"
    r"|no\.?\s*(\d+)",
    re.IGNORECASE,
)


def extract_identifiers(description: str) -> frozenset[str]:
    """Extract normalized inscribed identifiers from an appearance description.

    Only identifiers preceded by an explicit label are kept, so unrelated numbers
    (e.g. "18-inch weapon") are not treated as identity evidence. Returns an empty
    set when no explicit identifier was observed, which must never be used as
    evidence of distinctness. Leading zeros are normalized ("05" == "5").
    """
    ids = set()
    for group in _IDENTIFIER_PATTERN.findall(description or ""):
        value = next((g for g in group if g), None)
        if value is not None:
            ids.add(str(int(value)))
    return frozenset(ids)


def discriminative_identifiers(instance: Instance) -> frozenset[str]:
    """Identifiers for a local instance; delegates to extract_identifiers."""
    return extract_identifiers(instance.description)


@dataclass(frozen=True)
class IdentityProposal:
    proposal_id: str
    left: str
    right: str
    verdict: str
    media_ids: tuple[str, ...]
    visual_basis: str
    competing_instances: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    decision_id: str
    revision: int
    transaction_time: str
    proposal: IdentityProposal
    status: str
    reason: str
    invalidates: tuple[str, ...] = ()


class IdentityGraph:
    def __init__(self):
        self._decisions: list[Decision] = []

    @classmethod
    def restore(cls, rows, instances):
        """Replay/check a journal and preserve its original transaction times."""
        graph = cls()
        for row in rows:
            raw = dict(row["proposal"])
            raw["media_ids"] = tuple(raw["media_ids"])
            raw["competing_instances"] = tuple(raw.get("competing_instances", ()))
            proposal = IdentityProposal(**raw)
            if row["status"] == "revoked":
                result = graph.revoke(row["invalidates"][0], reason=row["reason"])
            else:
                result = graph.propose(proposal, instances)
            for field in ("decision_id", "revision", "status", "reason"):
                if getattr(result, field) != row[field]:
                    raise ValueError("identity journal failed semantic replay")
            if tuple(result.invalidates) != tuple(row["invalidates"]):
                raise ValueError("identity invalidation journal differs")
            datetime.fromisoformat(row["transaction_time"])
            graph._decisions[-1] = replace(result, transaction_time=row["transaction_time"])
        return graph

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(self._decisions)

    @property
    def revision(self) -> int:
        return len(self._decisions)

    def active(self) -> tuple[Decision, ...]:
        retired = {target for d in self._decisions for target in d.invalidates}
        return tuple(
            d for d in self._decisions if d.status == "accepted" and d.decision_id not in retired
        )

    def components(self, instances: Mapping[str, Instance]) -> tuple[frozenset[str], ...]:
        neighbors = {key: set() for key in instances}
        for decision in self.active():
            p = decision.proposal
            if p.verdict == "same":
                neighbors[p.left].add(p.right)
                neighbors[p.right].add(p.left)
        components = []
        remaining = set(instances)
        while remaining:
            seed = min(remaining)
            found, pending = set(), [seed]
            while pending:
                node = pending.pop()
                if node not in found:
                    found.add(node)
                    pending.extend(neighbors[node] - found)
            remaining -= found
            components.append(frozenset(found))
        return tuple(components)

    def _append(self, proposal, status, reason, invalidates=()):
        revision = self.revision + 1
        result = Decision(
            f"identity:{revision}",
            revision,
            datetime.now(timezone.utc).isoformat(),
            proposal,
            status,
            reason,
            tuple(invalidates),
        )
        self._decisions.append(result)
        return result

    def propose(self, proposal: IdentityProposal, instances: Mapping[str, Instance]) -> Decision:
        if any(d.proposal.proposal_id == proposal.proposal_id for d in self._decisions):
            raise ValueError("duplicate identity proposal")
        p = proposal
        if p.verdict not in {"same", "different", "unresolved"}:
            raise ValueError("invalid identity verdict")
        if p.left not in instances or p.right not in instances or p.left == p.right:
            raise ValueError("identity comparison requires two known local endpoints")
        if not set(p.competing_instances) <= set(instances):
            raise ValueError("unknown competing identity")
        if p.verdict == "unresolved":
            return self._append(p, "deferred", "insufficient_visual_evidence")
        if not p.visual_basis.strip():
            return self._append(p, "deferred", "missing_visual_basis")
        # Both endpoints must be grounded in the actual compared media.
        for ref in (p.left, p.right):
            if not {r.media_id for r in instances[ref].regions}.intersection(p.media_ids):
                return self._append(p, "deferred", "missing_endpoint_region")
        available = {r.media_id for i in instances.values() for r in i.regions}
        if not set(p.media_ids) <= available:
            raise ValueError("unknown identity evidence")
        groups = self.components(instances)
        union = set().union(*(g for g in groups if p.left in g or p.right in g))
        if p.verdict == "same":
            if instances[p.left].kind != instances[p.right].kind:
                return self._append(p, "deferred", "incompatible_instance_types")
            # Explicit inscribed identifiers are the strongest identity evidence.
            # When both endpoints carry different ones, the SAME interpretation is
            # vetoed; absence of an identifier never counts as distinctness.
            left_ids = discriminative_identifiers(instances[p.left])
            right_ids = discriminative_identifiers(instances[p.right])
            if left_ids and right_ids and left_ids.isdisjoint(right_ids):
                return self._append(p, "deferred", "inscribed_identifier_conflict")
            conflict = any(
                d.proposal.verdict == "different" and {d.proposal.left, d.proposal.right} <= union
                for d in self.active()
            )
            if conflict:
                return self._append(p, "deferred", "cluster_conflict")
            return self._append(p, "accepted", "visual_endpoints_and_cluster_checks_passed")
        # Later different evidence invalidates affected SAME interpretations.
        # Conservative: quarantine internal links rather than pick a path to keep.
        same_group = next((g for g in groups if p.left in g and p.right in g), None)
        retired = [
            d.decision_id
            for d in self.active()
            if same_group
            and d.proposal.verdict == "same"
            and {d.proposal.left, d.proposal.right} <= same_group
        ]
        return self._append(p, "accepted", "explicit_distinctness", retired)

    def revoke(self, decision_id: str, *, reason: str) -> Decision:
        previous = next((d for d in self.active() if d.decision_id == decision_id), None)
        if previous is None or not reason.strip():
            raise ValueError("revoke requires an active decision and reason")
        return self._append(previous.proposal, "revoked", reason, (decision_id,))

    def projection(self, instances: Mapping[str, Instance]) -> tuple[list[dict], dict[str, dict]]:
        entities, mapping = [], {}
        for group in self.components(instances):
            entity_id = "entity:" + digest(min(group))[:16]
            links = [
                d.decision_id
                for d in self.active()
                if d.proposal.verdict == "same" and {d.proposal.left, d.proposal.right} <= group
            ]
            entity = {
                "entity_id": entity_id,
                "version": self.revision,
                "local_instances": sorted(group),
                "identity_decisions": links,
                "status": "linked" if links else "local_only",
            }
            entities.append(entity)
            for ref in group:
                mapping[ref] = entity
        return entities, mapping

    def export(self) -> list[dict]:
        return [asdict(d) for d in self._decisions]
