"""Export the same scoped payload used by QA, without making model calls."""

import argparse
import json
from pathlib import Path

from rrt_echo.rrt.reader import payload_for
from rrt_echo.storage import atomic_json, read_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-bytes", type=int, default=150000)
    args = parser.parse_args()
    graph = read_graph(args.graph)
    args.out.mkdir(parents=True, exist_ok=False)
    payload = payload_for(graph, args.question, top_k=args.top_k, max_bytes=args.max_bytes)
    atomic_json(args.out / "payload.json", payload)
    atomic_json(
        args.out / "acceptance.json",
        {
            "snapshot_id": graph["snapshot_id"],
            "passed": False,
            "reason": "Diagnostic artifact only; independent media acceptance not performed",
        },
    )
    print(
        json.dumps(
            {
                "snapshot_id": graph["snapshot_id"],
                "facts": len(payload.get("facts", [])),
                "payload": str(args.out / "payload.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
