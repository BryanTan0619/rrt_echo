"""Read-only graph census and source-linked identity contact sheets.

The census measures representation, not identity accuracy. A human must inspect
regions to label false merges, splits and missed people.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from .storage import atomic_json, read_graph


def census(graph):
    instances = {i["instance_id"]: i for i in graph["instances"]}
    people = {key for key, i in instances.items() if i["kind"].casefold() in {"person", "human"}}
    person_entities = [e for e in graph["entities"] if people.intersection(e["local_instances"])]
    decisions = graph["identity_decisions"]
    retired = {key for d in decisions for key in d["invalidates"]}
    active = [d for d in decisions if d["status"] == "accepted" and d["decision_id"] not in retired]
    return {
        "snapshot_id": graph["snapshot_id"],
        "local_instances_all": len(instances),
        "local_person_instances": len(people),
        "registered_entities_all": len(graph["entities"]),
        "registered_person_entities": len(person_entities),
        "linked_person_entities": sum(e["status"] == "linked" for e in person_entities),
        "singleton_person_entities": sum(e["status"] != "linked" for e in person_entities),
        "local_tracklets": len(graph["local_tracklets"]),
        "tracked_person_instances": sum(bool(instances[i]["track_ref"]) for i in people),
        "facts": len(graph["facts"]),
        "fact_kinds": dict(Counter(f["kind"] for f in graph["facts"])),
        "structurally_joint_supported_facts": sum(
            f["support"]["local_joint"]["status"] == "supported" for f in graph["facts"]
        ),
        "active_identity_verdicts": dict(Counter(d["proposal"]["verdict"] for d in active)),
        "identity_decision_statuses": dict(Counter(d["status"] for d in decisions)),
        "identity_accuracy": None,
        "media_audit_required": True,
    }


def render(graph, output: Path, *, path_prefix=None, replacement=None):
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "census.json", census(graph))
    instances = {i["instance_id"]: i for i in graph["instances"]}
    sections = []
    index = []
    for number, entity in enumerate(graph["entities"]):
        members = [instances[key] for key in entity["local_instances"]]
        if not any(i["kind"].casefold() in {"person", "human"} for i in members):
            continue
        sightings = [
            (graph["evidence"][r["media_id"]]["seconds"], i, r)
            for i in members
            for r in i["regions"]
        ]
        sightings.sort(key=lambda row: row[0])
        selected = (
            [sightings[k] for k in sorted({round(n * (len(sightings) - 1) / 7) for n in range(8)})]
            if sightings
            else []
        )
        sheet = Image.new("RGB", (4 * 280, 2 * 220), "white")
        draw = ImageDraw.Draw(sheet)
        records = []
        for n, (seconds, instance, region) in enumerate(selected):
            media = graph["evidence"][region["media_id"]]
            uri = media["uri"]
            if path_prefix and uri.startswith(path_prefix):
                uri = replacement + uri[len(path_prefix) :]
            x, y = n % 4 * 280, n // 4 * 220
            try:
                with Image.open(uri) as image:
                    image = image.convert("RGB")
                    w, h = image.size
                    box = tuple(round(v * size) for v, size in zip(region["box"], (w, h, w, h)))
                    ImageDraw.Draw(image).rectangle(box, outline="red", width=max(2, w // 300))
                    image.thumbnail((275, 165))
                    sheet.paste(image, (x, y))
            except OSError:
                draw.text((x + 4, y + 10), "SOURCE UNAVAILABLE", fill="red")
            draw.text((x + 4, y + 168), f"{seconds:.2f}s {instance['instance_id']}", fill="black")
            draw.text((x + 4, y + 186), instance["description"][:38], fill="black")
            records.append(
                {
                    "seconds": seconds,
                    "instance_id": instance["instance_id"],
                    "media_id": region["media_id"],
                    "box": region["box"],
                }
            )
        name = f"person_{number:04d}.jpg"
        sheet.save(output / name)
        row = {
            "entity_id": entity["entity_id"],
            "status": entity["status"],
            "members": entity["local_instances"],
            "extent": [sightings[0][0], sightings[-1][0]] if sightings else None,
            "sheet": name,
            "sampled_regions": records,
        }
        index.append(row)
        sections.append(
            f'<section><h2>{html.escape(entity["entity_id"])} — {entity["status"]}</h2><p>{html.escape(str(row["extent"]))} · {len(members)} local instances</p><img width="1120" src="{name}"><pre>{html.escape(json.dumps(row, indent=2))}</pre></section>'
        )
    atomic_json(output / "person_index.json", index)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RRT identity audit</title><h1>Identity audit: visual verification required</h1><p>Boxes are model observations, not ground truth. Extent does not imply continuous visibility.</p>'
        + "".join(sections)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--path-prefix")
    parser.add_argument("--replacement")
    args = parser.parse_args()
    if bool(args.path_prefix) != bool(args.replacement):
        parser.error("path-prefix and replacement must be supplied together")
    render(
        read_graph(args.graph), args.out, path_prefix=args.path_prefix, replacement=args.replacement
    )


if __name__ == "__main__":
    main()
