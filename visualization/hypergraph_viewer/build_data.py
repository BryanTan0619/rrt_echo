#!/usr/bin/env python3
"""Hypergraph viewer data for a finished rrt_echo run (the author's new code, 09-16).

Reads ONLY the author's outputs, never rewrites them:
    <run>/hypergraph.json        facts (kind=event -> one ring each), instances, entities, evidence
    <run>/entity_registry.json   the 3 reference boxes the author picked per entity (used for the avatars)
    <run>/manifest.json          run parameters (input video path, local-seconds)
    <run>/prepared/frames/*.jpg  source frames (avatars are cropped from these with the author's boxes)
    <run>/clips.json             time extent of the run

    python build_data.py <run dir> <run name> [--out DIR] [--path-map OLD=NEW] [--ffmpeg PATH]

--path-map follows tools/render_text_evidence.py: the run records the absolute paths of the machine that produced it,
so frames/video are found under a different mount here. May be repeated. --out defaults to ./runs next to the pages.

No post-processing of our own: no identity merging, no object merging by name, no translation. People / objects are
exactly the author's `entities` (linked local instances are one node), events are exactly `facts[kind=event]` with
the author's roles resolved to entities. The page (index.html) is the one from echo_hypergraph_mockup; the data.json
keys are kept compatible with it, fields that do not exist in the new design are filled with neutral values.
"""

import argparse
import base64
import io
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PATH_MAP = {}  # filled from --path-map; the run stores the producing machine's absolute paths
OUT_ROOT = HERE / "runs"
FFMPEG = "ffmpeg"  # only used to add a webm fallback for the preview; skipped when not found


def local_path(p):
    p = str(p or "")
    for a, b in PATH_MAP.items():
        if p.startswith(a):
            return b + p[len(a) :]
    return p


# --no-avatars: skip the base64 crops (the page still works, nodes just have no picture)
NO_AVATARS = False
AVATAR = 52  # px, square, like the old page
MAX_FACES = 6  # per local instance in the panel


def load(run, name, default=None):
    p = run / name
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def crop_b64(uri, box, size=AVATAR):
    if NO_AVATARS:
        return None
    """Square avatar from a frame: the author's normalized xyxy box, padded to a square around its centre."""
    try:
        im = Image.open(local_path(uri))
    except Exception:
        return None
    W, H = im.size
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0, 8) * 1.15
    left, top = max(0, cx - side / 2), max(0, cy - side / 2)
    right, bottom = min(W, cx + side / 2), min(H, cy + side / 2)
    im = (
        im.convert("RGB")
        .crop((int(left), int(top), int(right), int(bottom)))
        .resize((size, size), Image.LANCZOS)
    )
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def short(instance_id):
    """`video:<hash>:action:<hash>:i0` -> `i0`; keeps whatever the author named the instance."""
    return instance_id.split(":")[-1] if ":" in instance_id else instance_id


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description="Build viewer data from a finished rrt_echo run.")
    ap.add_argument("run", help="the run directory (holds hypergraph.json)")
    ap.add_argument("name", help="name to show in the run selector")
    ap.add_argument(
        "--out",
        default=None,
        help="where runs/<name>/ is written (default: runs/ next to the pages)",
    )
    ap.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="rewrite a stored absolute path prefix, e.g. /data/=/mnt/data/ (repeatable)",
    )
    ap.add_argument(
        "--no-avatars",
        action="store_true",
        help="do not embed the cropped reference boxes; ~5x smaller data.json, for sharing a runnable demo",
    )
    ap.add_argument(
        "--expect-video",
        action="store_true",
        help="write video: segment.mp4 even when the file is not here yet (the viewer will look for it)",
    )
    ap.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg for the optional webm preview; skipped if missing",
    )
    a = ap.parse_args()
    global OUT_ROOT, FFMPEG, NO_AVATARS
    for pair in a.path_map:
        old, _, new = pair.partition("=")
        if not old or not new:
            raise SystemExit(f"--path-map wants OLD=NEW, got {pair!r}")
        PATH_MAP[old] = new
    if a.out:
        OUT_ROOT = Path(a.out).resolve()
    FFMPEG = a.ffmpeg
    NO_AVATARS = a.no_avatars
    run, name = Path(a.run).resolve(), a.name
    H = load(run, "hypergraph.json")
    reg = load(run, "entity_registry.json", {"entities": []})
    manifest = load(run, "manifest.json", {})
    clips = load(run, "clips.json", [])
    evidence = H.get("evidence", {})
    instances = {i["instance_id"]: i for i in H.get("instances", [])}
    facts = H.get("facts", [])

    # ---- time extent of the run
    t0, t1 = 0.0, 0.0
    if clips:
        t1 = max(float(c.get("target_end", 0)) for c in clips)
    if not t1:
        t1 = max([float(m.get("seconds", 0)) for m in evidence.values()] + [1.0])
    window = float(
        manifest.get("local_seconds") or manifest.get("args", {}).get("local_seconds") or 12
    )

    # ---- entities: one node per author entity; its members are the linked local instances
    inst_to_ent = {}
    for e in H.get("entities", []):
        for li in e.get("local_instances", []):
            inst_to_ent[li] = e["entity_id"]
    for iid in instances:  # instance without an entity row: its own node
        inst_to_ent.setdefault(iid, iid)
    ref_boxes = defaultdict(list)  # entity -> author-picked reference boxes
    for r in reg.get("entities", []):
        for v in r.get("visual_references", []):
            ref_boxes[r["entity_id"]].append(v)
    ent_members = defaultdict(list)
    for iid, eid in inst_to_ent.items():
        ent_members[eid].append(iid)

    # attributes / states / text facts hang on their owner instance
    attrs = defaultdict(list)
    for f in facts:
        if f.get("kind") == "event":
            continue
        owner = f.get("roles", {}).get("owner") or next(iter(f.get("roles", {}).values()), None)
        if not owner:
            continue
        ext = f.get("extent") or [0, 0]
        attrs[inst_to_ent.get(owner, owner)].append(
            {
                "prop": f.get("predicate") or f.get("kind"),
                "value": str(f.get("value") if f.get("value") is not None else ""),
                "start": float(ext[0]),
                "end": float(ext[1] if len(ext) > 1 else ext[0]),
                "via": [short(owner)],
            }
        )

    def inst_extent(iid):
        ext = instances.get(iid, {}).get("extent")
        if ext:
            return float(ext[0]), float(ext[1])
        ts = [
            float(evidence[m]["seconds"])
            for m in {r["media_id"] for r in instances.get(iid, {}).get("regions", [])}
            if m in evidence
        ]
        return (min(ts), max(ts)) if ts else (None, None)

    entities, kinds = [], {}
    for eid, members in ent_members.items():
        kind0 = instances.get(members[0], {}).get("kind", "person")
        kind = "person" if kind0 == "person" else "object"
        kinds[eid] = kind
        mem_rows, faces_all, aliases = [], [], []
        for iid in sorted(members, key=lambda i: (inst_extent(i)[0] or 0, i)):
            inst = instances.get(iid, {})
            first, last = inst_extent(iid)
            faces = []
            boxes = [v for v in ref_boxes.get(eid, []) if v.get("local_instance") == iid] or [
                {"media_id": r["media_id"], "box": r["box"]}
                for r in inst.get("regions", [])[:MAX_FACES]
            ]
            for v in boxes[:MAX_FACES]:
                ev = evidence.get(v["media_id"])
                if ev and ev.get("uri"):
                    b = crop_b64(ev["uri"], v["box"])
                    if b:
                        faces.append(b)
            desc = inst.get("description") or ""
            if desc and desc not in aliases:
                aliases.append(desc)
            mem_rows.append(
                {
                    "id": iid,
                    "first": first,
                    "last": last,
                    "face_embeds": len(inst.get("regions", [])),
                    "faces": faces,
                    "aliases": [desc] if desc else [],
                }
            )
            faces_all += faces
        firsts = [m["first"] for m in mem_rows if m["first"] is not None]
        entities.append(
            {
                # instance ids repeat per window (i0, i1, …), so the label carries the first-seen time
                "key": eid,
                "kind": kind,
                "name": short(members[0])
                + "@"
                + mmss(min(firsts) if firsts else 0)
                + (f" +{len(members) - 1}" if len(members) > 1 else ""),
                "look": aliases[0] if aliases else "",
                "members": mem_rows,
                "pairs": [],
                "faces": faces_all[:1],
                "unresolved_mention": False,
                "attributes": sorted(attrs.get(eid, []), key=lambda a: a["start"]),
                "first": min(firsts) if firsts else 0.0,
                # object-panel fields of the old page: here they just describe the author's entity
                "reason": f"author entity {eid} ({len(members)} local instances, status={next((e.get('status') for e in H.get('entities', []) if e['entity_id'] == eid), 'local')})",
                "notes": [f"local instances: {', '.join(members)}"],
                "occurrences": [],
            }
        )

    # ---- events: one ring per fact of kind=event
    bubbles, no_roles = [], 0
    for f in sorted(facts, key=lambda f: (f.get("extent") or [0])[0]):
        if f.get("kind") != "event":
            continue
        rr = f.get("resolved_roles") or {}
        roles, text = {}, {}
        for role, iid in (f.get("roles") or {}).items():
            eid = (rr.get(role) or {}).get("entity_id") or inst_to_ent.get(iid, iid)
            roles[eid] = role if eid not in roles else roles[eid] + "/" + role
            text[eid] = short(iid)
        members = list(roles)
        if not members:
            no_roles += 1
            continue
        ext = f.get("extent") or [0, 0]
        start, end = float(ext[0]), float(ext[1] if len(ext) > 1 else ext[0])
        if end <= start:
            end = start + 1.0
        pred = f.get("predicate") or "?"
        desc = (
            pred
            + "("
            + ", ".join(f"{r}={short(i)}" for r, i in (f.get("roles") or {}).items())
            + ")"
        )
        sup = (f.get("support") or {}).get("local_joint", {}).get("status", "")
        bubbles.append(
            {
                "id": f["fact_id"],
                "members": members,
                "label": f"{pred}: " + " → ".join(text[m] for m in members),
                "single": len(members) < 2,
                "start": start,
                "intervals": [[start, end]],
                "actions": [
                    {
                        "start": start,
                        "end": end,
                        "pred": pred,
                        "desc": desc,
                        "desc_en": desc,
                        "roles": roles,
                        "text": text,
                        "status": "full" if sup == "supported" else "partial",
                        "evidence": len(f.get("joint_evidence") or []),
                        "unresolved": f.get("unresolved_slots") or [],
                        "event_id": f["fact_id"],
                        "version": H.get("identity_revision", 1),
                        "continued": False,
                        "offline_rewritten": False,
                        "merged_members": [],
                    }
                ],
            }
        )
        for m in members:
            e = next(x for x in entities if x["key"] == m)
            if kinds.get(m) == "object":
                e["occurrences"].append(
                    {
                        "start": start,
                        "end": end,
                        "slot": roles[m],
                        "text": text[m],
                        "pred": pred,
                        "holder": next(
                            (text[k] for k in members if kinds.get(k) == "person"), None
                        ),
                    }
                )

    linked = [e for e in entities if len(e["members"]) > 1]
    stats = {
        "events": sum(1 for f in facts if f.get("kind") == "event"),
        "drawn": len(bubbles),
        "no_roles": no_roles,
        "absorbed": 0,
        "continued": 0,
        "identity_merges": len(linked),
        "event_merges": 0,
        "people": sum(1 for e in entities if e["kind"] == "person"),
        "objects": sum(1 for e in entities if e["kind"] == "object"),
    }
    view = {
        "entities": entities,
        "bubbles": bubbles,
        "stats": stats,
        "identity_merges": [
            {"canonical": e["key"], "entities": [m["id"] for m in e["members"]], "pairs": []}
            for e in linked
        ],
    }

    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    video = local_path(manifest.get("video") or (manifest.get("args") or {}).get("video"))
    link = out / "segment.mp4"
    if video and Path(video).exists():
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(video)
    # a VP8 copy for browsers without H.264 (headless Chromium, Firefox); same as the old build_data did
    webm, ffmpeg = out / "segment.webm", shutil.which(FFMPEG)
    if link.exists() and not webm.exists() and ffmpeg:
        os.system(
            f"'{ffmpeg}' -hide_banner -loglevel error -y -i '{link}' -c:v libvpx -deadline realtime -cpu-used 8 -b:v 1M "
            f"-vf scale=-2:480 -c:a libvorbis '{webm}'"
        )
    data = {
        "name": name,
        "source": str(run / "hypergraph.json"),
        "t0": t0,
        "t1": t1,
        "window": window,
        "video": "segment.mp4" if (link.exists() or a.expect_video) else None,
        "video_webm": "segment.webm" if webm.exists() else None,
        "zh": False,
        "views": {"graph": view},
        "default_view": "graph",
        "rules": {"same_holder_gap": 0, "generic": [], "merge_same_name": False},
        "snapshot_id": H.get("snapshot_id"),
        "architecture": H.get("architecture"),
    }
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    idx_p = OUT_ROOT / "index.json"
    idx = [
        r
        for r in (json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else [])
        if r["name"] != name
    ]
    idx.append({"name": name, "t0": t0, "t1": t1, "views": ["graph"]})
    idx_p.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"{name}: {stats} | entities {len(entities)} | snapshot {str(H.get('snapshot_id'))[:12]} | video {'ok' if link.exists() else 'none'}"
    )


if __name__ == "__main__":
    main()
