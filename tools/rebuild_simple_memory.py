"""Import frozen observations once, or rebuild only from the canonical simple ledger."""

import argparse
import json
from pathlib import Path

from rrt_echo.rrt.memory import RRTMemory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ledger")
    src.add_argument("--observations")
    p.add_argument("--journal", help="Frozen binding journal; imported as new ledger revisions")
    p.add_argument("--audio")
    p.add_argument("--source-video-id")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.ledger:
        if a.audio or a.journal:
            p.error("rebuild reads only its ledger")
        mem = RRTMemory.open(a.ledger)
    else:
        mem = RRTMemory(a.source_video_id, out / "ledger")
        rows = json.loads(Path(a.observations).read_text())
        rows = rows if isinstance(rows, list) else [rows]
        # Frozen journal already contains initial and derived constraints.
        journal = json.loads(Path(a.journal).read_text()) if a.journal else None
        for row in rows:
            mem.append(row if journal is None else {**row, "link_proposals": []})
        if journal is not None:
            existing = {
                r["proposal"]["proposal_id"] for r in mem.journal if r["operation"] == "propose"
            }
            for row in journal:
                if row["operation"] == "propose":
                    if row["proposal"]["proposal_id"] not in existing:
                        mem.propose(row["proposal"])
                else:
                    mem.revoke(row["proposal_id"], row["reason"])
        if a.audio:
            mem.set_audio(json.loads(Path(a.audio).read_text()))
    graph = mem.materialize(out)
    replay = RRTMemory.open(mem.ledger_dir).materialize()
    if replay != graph:
        raise RuntimeError("non_deterministic_rebuild")
    print(
        json.dumps(
            {
                "snapshot_id": graph["snapshot_id"],
                "facts": len(graph["facts"]),
                "bindings": len(mem.journal),
                "rebuild_equal": True,
                "semantic_acceptance": None,
            }
        )
    )


if __name__ == "__main__":
    main()
