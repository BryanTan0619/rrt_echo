"""B/C: sparse motion candidates and bidirectional visual identity resolution.

Motion, color and language are retrieval cues only. No cue threshold commits identity.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ..echo_perception.correspondence import CorrespondenceClient
from ..echo_perception.tracking import LocalTracker
from ..echo_perception.types import atomic_json


def inventory(results):
    instances = {}
    media = {}
    observation_of = {}
    for result in results:
        obs = result["observation"]
        for m in obs["media"]:
            media[m["media_id"]] = m
        for instance in obs["instances"]:
            instances[instance["instance_id"]] = instance
            observation_of[instance["instance_id"]] = obs["observation_id"]
    return instances, media, observation_of


def seconds(m):
    return m["pts"] * m["time_base"][0] / m["time_base"][1]


def motion_candidates(results, media_shots=None):
    """Associate available spatial anchors within shots; not dense tracking.

    Every candidate preserves the two source boxes. Tracking cannot certify identity.
    """
    instances, media, _ = inventory(results)
    grouped = defaultdict(list)
    for key, inst in instances.items():
        if inst["kind"] not in {"person", "object"}:
            continue
        for r in inst["regions"]:
            grouped[(r["media_id"], inst["kind"])].append((key, r["box"]))
    trackers = {kind: LocalTracker() for kind in ("person", "object")}
    latest = {}
    rows = []
    pairs = {}
    for (mid, kind), items in sorted(
        grouped.items(), key=lambda row: (seconds(media[row[0][0]]), row[0])
    ):
        shot = (media_shots or {}).get(mid)
        if shot is None:
            shot = mid  # Unknown cut provenance: never infer continuity across frames.
        unique = list(dict.fromkeys((key, tuple(box)) for key, box in items))
        assigned = trackers[kind].update(
            [box for _, box in unique], seconds=seconds(media[mid]), shot_id=shot
        )
        for (key, box), track in zip(unique, assigned):
            tid = f"{kind}:{shot}:{track['track_ref']}"
            row = {
                "instance_id": key,
                "media_id": mid,
                "box": list(box),
                "track_id": tid,
                "status": "candidate_only",
            }
            previous = latest.get(tid)
            if previous and previous["instance_id"] != key:
                pair = tuple(sorted((previous["instance_id"], key)))
                pairs[pair] = {
                    "left": pair[0],
                    "right": pair[1],
                    "relation": "continues",
                    "evidence_ids": [previous["media_id"], mid],
                    "source": "sparse_motion",
                }
            latest[tid] = row
            rows.append(row)
    return {
        "observations": rows,
        "pairs": list(pairs.values()),
        "method": "sparse_anchor_motion_iou; not_dense_MOT",
        "identity_accepted": False,
    }


def crop_feature(instance, media):
    """Small RGB histogram baseline, explicitly only a candidate-ranking feature."""
    vectors = []
    for r in instance["regions"][:: max(1, len(instance["regions"]) // 3)][:3]:
        with Image.open(media[r["media_id"]]["uri"]) as im:
            w, h = im.size
            a, b, c, d = r["box"]
            crop = (
                im.convert("RGB")
                .crop(
                    (
                        int(a * w),
                        int(b * h),
                        max(int(a * w) + 1, int(c * w)),
                        max(int(b * h) + 1, int(d * h)),
                    )
                )
                .resize((32, 32))
            )
            hist = crop.histogram()
            vector = [
                sum(hist[ch * 256 + k : ch * 256 + k + 32]) / 1024
                for ch in range(3)
                for k in range(0, 256, 32)
            ]
            norm = math.sqrt(sum(x * x for x in vector))
            vectors.append([x / norm for x in vector] if norm else vector)
    return vectors


def candidate_pairs(results, motion=None, top_k=3):
    instances, media, obs_of = inventory(results)
    eligible = {
        k: i for k, i in instances.items() if i["kind"] in {"person", "object"} and i["regions"]
    }
    features = {k: crop_feature(i, media) for k, i in eligible.items()}
    times = {
        k: sum(seconds(media[r["media_id"]]) for r in i["regions"]) / len(i["regions"])
        for k, i in eligible.items()
    }
    pairs = {}
    for p in (motion or {}).get("pairs", []):
        pairs[tuple(sorted((p["left"], p["right"])))] = dict(p, priority=2.0)
    # Inscribed identifier index: same-number buckets pair directly at high priority,
    # so identity recall no longer depends on the top-k appearance ranking.
    from ..identity import extract_identifiers

    id_of = {}
    buckets = defaultdict(list)
    for key, inst in eligible.items():
        ids = extract_identifiers(inst["description"])
        id_of[key] = ids
        for nid in ids:
            buckets[nid].append(key)
    for nid, keys in buckets.items():
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                pair = tuple(sorted((keys[a], keys[b])))
                pairs.setdefault(
                    pair,
                    {
                        "left": pair[0],
                        "right": pair[1],
                        "relation": "same_identity",
                        "source": "inscribed_identifier_match",
                        "priority": 5.0,
                    },
                )
    for key, inst in eligible.items():
        ranks = []
        words = set(re.findall(r"\w+", inst["description"].lower()))
        for other, oi in eligible.items():
            if other == key or oi["kind"] != inst["kind"]:
                continue
            # Explicitly different inscribed identifiers cannot merge; skip the pair
            # rather than spend comparison budget on an identity that will be vetoed.
            if id_of[key] and id_of[other] and id_of[key].isdisjoint(id_of[other]):
                continue
            appearance = max(
                (sum(a * b for a, b in zip(x, y)) for x in features[key] for y in features[other]),
                default=0,
            )
            ow = set(re.findall(r"\w+", oi["description"].lower()))
            lexical = len(words & ow) / max(1, len(words | ow))
            # Cues affect ranking only and are recorded; neither grants same/different.
            shared_frames = {r["media_id"] for r in inst["regions"]} & {
                r["media_id"] for r in oi["regions"]
            }
            # Same-observation endpoints can be repeated sightings of one person.
            # They were previously excluded, leaving duplicates disconnected.
            score = appearance + 0.3 * lexical + 0.25 * bool(shared_frames)
            ranks.append((score, -abs(times[key] - times[other]), other))
        selected = sorted(ranks, reverse=True)[:top_k]
        # Preserve a cross-observation candidate even when local duplicates rank first.
        distant = [r for r in ranks if obs_of[r[2]] != obs_of[key]]
        if distant:
            selected.append(max(distant))
        # Ensure temporal directions can contribute despite similar distant appearances.
        for side in (-1, 1):
            near = [r for r in ranks if (times[r[2]] - times[key]) * side > 0]
            if near:
                selected.append(max(near, key=lambda r: r[1]))
        for score, _, other in selected:
            pair = tuple(sorted((key, other)))
            pairs.setdefault(
                pair,
                {
                    "left": pair[0],
                    "right": pair[1],
                    "relation": "same_identity",
                    "source": "rgb_histogram_description_and_temporal_gallery",
                    "priority": score,
                },
            )
    return sorted(pairs.values(), key=lambda p: (-p["priority"], p["left"], p["right"]))


def ownership_candidates(results, top_k=3):
    """Retrieve possible owners in both temporal directions; never certify ownership.

    Local co-visibility ranks ahead of adjacent segments, but is not proof of
    ownership. A region stays unresolved when none of these comparisons succeeds.
    """
    instances, media, obs_of = inventory(results)
    owners = {
        k: i for k, i in instances.items() if i["kind"] in {"person", "object"} and i["regions"]
    }
    rows = []
    for key, region in instances.items():
        if region["kind"] != "region" or not region["regions"]:
            continue
        mids = {r["media_id"] for r in region["regions"]}
        t = sum(seconds(media[mid]) for mid in mids) / len(mids)
        ranked = []
        for other, owner in owners.items():
            omids = {r["media_id"] for r in owner["regions"]}
            ot = sum(seconds(media[mid]) for mid in omids) / len(omids)
            ranked.append(
                (
                    int(bool(mids & omids)),
                    int(obs_of[key] == obs_of[other]),
                    -abs(t - ot),
                    other,
                    ot,
                )
            )
        selected = sorted(ranked, reverse=True)[:top_k]
        for side in (-1, 1):
            adjacent = [r for r in ranked if (r[4] - t) * side > 0]
            if adjacent:
                selected.append(max(adjacent, key=lambda r: r[2]))
        for row in {r[3]: r for r in selected}.values():
            rows.append(
                {
                    "left": key,
                    "right": row[3],
                    "relation": "part_of",
                    "source": "co_visible_or_bidirectional_owner_candidate",
                    "priority": 3.0 + row[0] + row[1] * 0.1,
                }
            )
    return sorted(rows, key=lambda p: (-p["priority"], p["left"], p["right"]))


def representative_regions(instance, media, limit=3):
    """Temporally spread anchors (first/middle/last), not area-max crops.

    Continuity comparisons need a person's pose and attire across their whole
    sighting, not the largest box (which may be only a back view). Spread evenly
    in time so a turn or a frontal frame in between is not dropped.
    """
    regions = sorted(instance["regions"], key=lambda r: seconds(media[r["media_id"]]))
    if len(regions) <= limit:
        return regions
    idx = [round(i * (len(regions) - 1) / (limit - 1)) for i in range(limit)]
    return [regions[i] for i in dict.fromkeys(idx)]


def resolve(
    results,
    *,
    model,
    base_url,
    out,
    deadline,
    motion=None,
    top_k=3,
    max_batches=100,
    sender=None,
    workers=1,
    skip_observed_pairs=False,
    previous_pairs=(),
):
    from ..echo_perception.types import Media

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    instances, media, _ = inventory(results)
    pairs = ownership_candidates(results, top_k) + candidate_pairs(results, motion, top_k)

    def pair_key(p):
        relation = "part_of" if p["relation"] == "part_of" else "same_identity"
        endpoints = (p["left"], p["right"])
        return (relation, endpoints if relation == "part_of" else tuple(sorted(endpoints)))

    seen_pairs = {pair_key(p) for p in previous_pairs}
    pairs = [p for p in pairs if pair_key(p) not in seen_pairs]
    skipped = []
    if skip_observed_pairs:
        previous = [p for r in results for p in r.get("link_proposals", [])]
        pending = []
        for pair in pairs:
            endpoints = (pair["left"], pair["right"])
            mids = {
                r["media_id"]
                for key in endpoints
                for r in representative_regions(instances[key], media)
            }
            relation = "part_of" if pair["relation"] == "part_of" else "same_identity"
            seen = any(
                (
                    p["relation"] == relation
                    or relation == "same_identity"
                    and p["relation"] == "continues"
                )
                and (
                    (p["source"], p["target"]) == endpoints
                    if relation == "part_of"
                    else {p["source"], p["target"]} == set(endpoints)
                )
                and mids <= set(p["evidence_ids"])
                for p in previous
            )
            (skipped if seen else pending).append(pair)
        pairs = pending
    atomic_json(out / "skipped_unchanged_evidence.json", skipped)
    atomic_json(out / "candidates.json", pairs)
    client = CorrespondenceClient(
        model=model, base_url=base_url, max_tokens=2000, partial_rows=skip_observed_pairs
    )
    proposals = []
    failures = []
    processed = 0
    # Calls may overlap on the shared inference server; journal writes stay ordered.
    if workers < 1 or workers > 4:
        raise ValueError("association_workers_must_be_1_to_4")
    if skip_observed_pairs:
        # Do not mix ownership and identity verdict vocabularies in one call.
        # Give cross-clip people a slot before spending the budget on objects.
        groups = [[], [], []]
        for pair in pairs:
            bucket = (
                1
                if pair["relation"] == "part_of"
                else (0 if instances[pair["left"]]["kind"] == "person" else 2)
            )
            groups[bucket].append(pair)
        # Round-robin prevents many person pairs from exhausting every owner/object slot.
        chunks = [[group[i : i + 4] for i in range(0, len(group), 4)] for group in groups]
        interleaved = [
            chunks[k][i]
            for i in range(max(map(len, chunks), default=0))
            for k in range(len(chunks))
            if i < len(chunks[k])
        ]
        pairs = [p for batch in interleaved for p in batch]
        batches, offset = [], 0
        for batch in interleaved[:max_batches]:
            batches.append((offset, batch))
            offset += len(batch)
    else:
        batches = [
            (offset, pairs[offset : offset + 4])
            for offset in range(0, min(len(pairs), max_batches * 4), 4)
        ]

    def compare_batch(item):
        offset, batch = item
        if deadline.remaining() < 30:
            return [], [], 0
        local_proposals = []
        local_failures = []
        selected = {}
        wire = []
        for p in batch:
            for key in (p["left"], p["right"]):
                inst = instances[key]
                if skip_observed_pairs:
                    anchors = representative_regions(inst, media)
                else:
                    ordered = sorted(inst["regions"], key=lambda r: seconds(media[r["media_id"]]))
                    anchors = list(
                        {
                            (r["media_id"], tuple(r["box"])): r for r in (ordered[0], ordered[-1])
                        }.values()
                    )
                selected[key] = {**inst, "regions": anchors}
            pid = (
                "association:"
                + hashlib.sha256(
                    (p["relation"] + "|" + p["left"] + "|" + p["right"]).encode()
                ).hexdigest()[:20]
            )
            wire.append(
                {
                    "pair_id": pid,
                    "left": p["left"],
                    "right": p["right"],
                    "relation": "part_of" if p["relation"] == "part_of" else "same_identity",
                }
            )
        mids = {r["media_id"] for i in selected.values() for r in i["regions"]}
        try:
            rows = client.compare(
                pairs=wire,
                instances=selected,
                media=[
                    Media(**media[mid]) for mid in sorted(mids, key=lambda m: seconds(media[m]))
                ],
                out=out / "calls",
                deadline=deadline,
                sender=sender,
            )
            source = {w["pair_id"]: p for w, p in zip(wire, batch)}
            for row in rows:
                p = source[row["pair_id"]]
                local_proposals.append(
                    {
                        "proposal_id": row["pair_id"],
                        "source": row["left"],
                        "target": row["right"],
                        "relation": row["relation"],
                        "candidate_relation": p["relation"],
                        "verdict": {
                            "same": "supported",
                            "belongs": "supported",
                            "different": "contradicted",
                            "not_belongs": "contradicted",
                            "unresolved": "unresolved",
                        }[row["verdict"]],
                        "evidence_ids": row["evidence_ids"],
                        "basis": row["basis"],
                        "status": "proposed",
                        "retrieval_source": p["source"],
                        "semantic_support_audited": False,
                    }
                )
        except Exception as e:
            local_failures.append({"offset": offset, "pairs": wire, "error": str(e)})
        return local_proposals, local_failures, len(batch)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for new_proposals, new_failures, count in pool.map(compare_batch, batches):
            proposals.extend(new_proposals)
            failures.extend(new_failures)
            processed += count
            atomic_json(out / "links.json", proposals)
            atomic_json(out / "failures.json", failures)
            atomic_json(out / "progress.json", {"processed": processed, "total": len(pairs)})
    atomic_json(
        out / "report.json",
        {
            "candidate_pairs": len(pairs),
            "processed": processed,
            "unprocessed": pairs[processed:],
            "failures": failures,
            "semantic_quality_passed": None,
        },
    )
    return proposals
