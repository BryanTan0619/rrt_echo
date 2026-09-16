"""Export literal text and owner evidence from a frozen graph for visual inspection."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image

from rrt_echo.storage import read_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Remap a source-media path prefix after moving artifacts",
    )
    args = parser.parse_args()
    graph, out = read_graph(Path(args.graph)), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    instances = {i["instance_id"]: i for i in graph["instances"]}
    cards, index = [], []
    for number, fact in enumerate(f for f in graph["facts"] if f["kind"] == "text"):
        owner = fact["roles"].get("owner")
        refs = []
        for offset, mid in enumerate(fact["joint_evidence"][:2]):
            media = graph["evidence"][mid]
            source = Path(media["uri"])
            if not source.exists():
                for mapping in args.path_map:
                    old, new = mapping.split("=", 1)
                    if str(source).startswith(old):
                        source = Path(new + str(source)[len(old) :])
                        break
            destination = out / "frames" / f"{number}_{offset}.jpg"
            with Image.open(source) as frame:
                frame.thumbnail((960, 640))
                frame.save(destination, quality=92)
            refs.append(
                {
                    "media_id": mid,
                    "seconds": media["seconds"],
                    "display_file": str(destination.relative_to(out)),
                }
            )
        record = {
            "fact_id": fact["fact_id"],
            "text": fact["value"],
            "owner": owner,
            "appearance": instances[owner]["description"] if owner in instances else None,
            "owner_projection": fact.get("owner_projections", {}),
            "frames": refs,
        }
        index.append(record)
        pictures = "".join(
            f'<figure><img src="{r["display_file"]}"><figcaption>{r["seconds"]:.2f}s</figcaption></figure>'
            for r in refs
        )
        cards.append(
            "<article><h2>"
            + html.escape(str(fact["value"]))
            + "</h2><p>局部承载物："
            + html.escape(str(record["appearance"]))
            + "</p>"
            + pictures
            + "<details><summary>原始端点与归属</summary><pre>"
            + html.escape(json.dumps(record, ensure_ascii=False, indent=2))
            + "</pre></details></article>"
        )
    (out / "text_evidence.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>文字与承载物证据</title><style>body{font:16px system-ui;background:#eef2f6;max-width:1300px;margin:24px auto}article{background:white;padding:24px;margin:20px;border-radius:12px}figure{display:inline-block;width:46%;margin:1%}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>文字与承载物：原始帧核验</h1><p>标题来自模型识别，可能错误。下方原始帧用于核验，不能把 schema 合法当作视频正确。</p>'
        + "".join(cards)
    )
    print(json.dumps({"text_facts": len(index), "output": str(out)}))


if __name__ == "__main__":
    main()
