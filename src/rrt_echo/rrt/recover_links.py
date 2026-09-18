"""Recover independent valid rows from stored correspondence responses, without inference."""

import json
from pathlib import Path

from perception.correspondence import validate_correspondences
from perception.types import atomic_json
from .association import inventory


def recover_links(results, association_dir, out):
    instances, _, _ = inventory(results)
    proposals = {}
    errors = []
    for request in sorted((Path(association_dir) / "calls").glob("*/request.json")):
        data = json.loads(request.read_text())
        response = request.parent / "response.json"
        if not response.exists():
            continue
        saved = json.loads(response.read_text())
        choice = saved["choices"][0]
        if choice.get("finish_reason") == "length":
            errors.append({"call": str(request.parent), "error": "truncated_response"})
            continue
        try:
            raw = json.loads(choice["message"]["content"])
        except Exception as exc:
            errors.append({"call": str(request.parent), "error": str(exc)})
            continue
        aliases = {f"f{n}": m["media_id"] for n, m in enumerate(data["media"])}
        for pair in data["pairs"]:
            rows = [
                r for r in raw.get("correspondences", []) if r.get("pair_id") == pair["pair_id"]
            ]
            row = dict(rows[0]) if len(rows) == 1 else {}
            initial = row.get("verdict", "unresolved")
            try:
                if not row:
                    raise ValueError("missing_or_duplicate_pair")
                row["evidence_ids"] = list(dict.fromkeys(aliases[m] for m in row["evidence_ids"]))
                row.setdefault(
                    "gap_reason", "missing_connection" if initial == "unresolved" else "none"
                )
                checked = validate_correspondences(
                    {"correspondences": [row]}, [pair], instances, list(aliases.values())
                )[0]
                verdict = {
                    "same": "supported",
                    "belongs": "supported",
                    "different": "contradicted",
                    "not_belongs": "contradicted",
                    "unresolved": "unresolved",
                }[checked["verdict"]]
                reason = checked["basis"]
                mids = checked["evidence_ids"]
                error = None
            except Exception as exc:
                verdict = "unresolved"
                reason = "Invalid pair evidence; requires focused review"
                mids = []
                error = str(exc)[:500]
                errors.append({"pair_id": pair["pair_id"], "error": error})
            proposals[pair["pair_id"]] = {
                "proposal_id": pair["pair_id"],
                "source": pair["left"],
                "target": pair["right"],
                "relation": pair["relation"],
                "verdict": verdict,
                "evidence_ids": mids,
                "basis": reason,
                "status": "proposed",
                "semantic_support_audited": False,
                "prior_model_verdict": initial,
                "recovered_from": str(response),
                "row_error": error,
            }
    result = list(proposals.values())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "recovered_links.json", result)
    atomic_json(out / "row_errors.json", errors)
    return result
