"""Public CLI: build, replay, inspect and graph-only QA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .evaluation import audit_assertions, summarize
from .identity import IdentityProposal
from .memory import Memory
from .pipeline import build, load_config
from .retrieval import Query, retrieve
from .runtime import Deadline, supervise, worker_command
from .schema import ObservationPacket, digest
from .storage import atomic_json, export_run


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def new_run(path):
    # Exclusive creation prevents concurrent writers or accidental overwrites.
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_graph(path):
    graph = json.loads(Path(path).read_text())
    if graph.get("snapshot_id") != digest({k: v for k, v in graph.items() if k != "snapshot_id"}):
        raise ValueError("snapshot hash mismatch")
    return graph


def replay(args):
    out = new_run(args.out)
    memory = Memory()
    for value in rows(args.observations):
        memory.append(ObservationPacket.from_dict(value))
    if args.identity:
        for value in rows(args.identity):
            memory.compare(
                IdentityProposal(
                    **{
                        **value,
                        "media_ids": tuple(value["media_ids"]),
                        "competing_instances": tuple(value.get("competing_instances", [])),
                    }
                )
            )
    export_run(memory, out, status="complete")
    atomic_json(
        out / "manifest.json",
        {"mode": "observation_replay", "cold_video_build": False, "quality_passed": None},
    )
    return 0


def evaluate(args):
    from .perception.prompts import ANSWER, AUDIT
    from .perception.vlm import VisionClient

    graph = read_graph(args.graph)
    target = graph.get("source_video_id") or graph["video_id"]
    questions = [q for q in rows(args.questions) if q.get("video_id", target) == target]
    if not questions:
        raise ValueError(f"no questions match graph video ID: {target}")
    out = new_run(args.out)
    atomic_json(
        out / "manifest.json",
        {
            "snapshot_id": graph["snapshot_id"],
            "video_id": target,
            "questions": len(questions),
            "reader_video_access": False,
        },
    )
    client = VisionClient(model=args.model)
    results = []
    for index, question in enumerate(questions):
        row = {
            "question_id": question.get("question_id", str(index)),
            "type": question.get("hallucination_type", question.get("type", "unspecified")),
            "correct": False,
            "strict_support": None,
        }
        try:
            options = question.get("options") or dict(
                re.findall(r"^([A-Z])[).:]\s*(.+)$", question["question"], re.MULTILINE)
            )
            if not options:
                raise ValueError("question needs explicit multiple-choice options")
            query = Query(
                question["question"], tuple(question.get("required_dimensions", ["local_joint"]))
            )
            payload = retrieve(graph, query, top_k=args.top_k, max_bytes=args.max_bytes)
            row["reader_payload"] = payload
            deadline = Deadline(args.question_seconds)
            answer_input = {"question": question["question"], "options": options, "graph": payload}
            response = client.complete(ANSWER + "\n" + json.dumps(answer_input), deadline=deadline)
            answer = str(response.get("answer", ""))
            if answer not in options:
                raise ValueError("answer is not one of the required options")
            row.update(
                answer=answer, correct=answer == str(question.get("ans", question.get("answer")))
            )
            # No gold answer, captions, or video reach the support auditor.
            audit = client.complete(
                AUDIT + "\n" + json.dumps({"output_answer": options[answer], "graph": payload}),
                deadline=deadline,
            )
            row.update(audit_assertions(audit.get("assertions", []), payload))
        except Exception as exc:
            row["error"] = str(exc)
        results.append(row)
        atomic_json(out / "results.json", results)
    atomic_json(out / "summary.json", summarize(results))
    atomic_json(out / "calls.json", client.calls)
    return 0 if not any(r.get("error") for r in results) else 2


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("replay", help="build from immutable local observations")
    p.add_argument("observations", type=Path)
    p.add_argument("--identity", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("build", help="raw video to offline memory under a hard deadline")
    p.add_argument("video", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("video", type=Path)
    p.add_argument("config", type=Path)
    p.add_argument("out", type=Path)
    p = commands.add_parser("inspect", help="show the exact graph-only reader payload")
    p.add_argument("graph", type=Path)
    p.add_argument("--question", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=100_000)
    p = commands.add_parser("qa", help="forced MCQ answers and separate graph support audit")
    p.add_argument("graph", type=Path)
    p.add_argument("questions", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=100_000)
    p.add_argument("--question-seconds", type=float, default=180)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "replay":
        return replay(args)
    if args.command == "build":
        config = load_config(args.config)
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        out = new_run(args.out)
        config_path = out / "config.json"
        atomic_json(config_path, config)
        return supervise(
            worker_command(args.video.resolve(), config_path.resolve(), out.resolve()),
            out,
            seconds=config["budget_seconds"],
        )
    if args.command == "_worker":
        return build(args.video, args.out, load_config(args.config))
    if args.command == "inspect":
        print(
            json.dumps(
                retrieve(
                    read_graph(args.graph),
                    Query(args.question),
                    top_k=args.top_k,
                    max_bytes=args.max_bytes,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
