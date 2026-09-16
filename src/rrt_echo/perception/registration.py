"""Ground model-local sightings to detector tracklets without trusting model IDs."""

from __future__ import annotations

from dataclasses import replace

from .tracking import iou


def register_packet(packet, detections):
    """One-to-one region association. Composite/ambiguous matches stay untracked.

    The persistent tracklet key is scoped to video and shot, independent of the
    scheduling window. Global identity is still a separate decision.
    """
    proposed = {}
    for instance in packet.instances:
        matches = set()
        valid = bool(instance.regions)
        for region in instance.regions:
            candidates = {
                d["track_ref"]
                for d in detections
                if d["media_id"] == region.media_id and iou(region.box, d["box"]) >= 0.8
            }
            if len(candidates) != 1:
                valid = False
                break
            matches.update(candidates)
        if valid and len(matches) == 1:
            proposed[instance.instance_id] = next(iter(matches))
    counts = {track: list(proposed.values()).count(track) for track in proposed.values()}
    instances = []
    for instance in packet.instances:
        track = proposed.get(instance.instance_id)
        trusted = (
            f"{packet.video_id}:shot:{packet.shot_id}:{track}"
            if track and counts[track] == 1
            else None
        )
        instances.append(replace(instance, track_ref=trusted))
    return replace(packet, instances=tuple(instances))
