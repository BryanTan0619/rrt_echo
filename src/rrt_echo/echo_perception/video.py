"""Shot-proposed target intervals + adjacent context; original PTS preserved.
Adapted from the uploaded video.py. Sampling is uniform BEFORE budget reduction:
high-motion frames cannot displace all low-motion portions of the target.
"""

import math
from bisect import bisect_left
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from .types import Clip, Deadline, Frame, Media, atomic_json, digest


def assign_shots(records, threshold=0.10, minimum_gap=0.25):
    candidates = []
    for i, row in enumerate(records):
        neighborhood = [r.motion for r in records[max(0, i - 8) : i + 9]]
        if i and row.motion >= threshold and row.motion >= 2 * median(neighborhood):
            candidates.append(row)
    accepted = []
    for row in sorted(candidates, key=lambda r: (-r.motion, r.index)):
        if all(abs(row.seconds - other.seconds) >= minimum_gap for other in accepted):
            accepted.append(row)
    cuts = {r.index for r in accepted}
    shot, out = 0, []
    for row in records:
        if row.index in cuts:
            shot += 1
        out.append(replace(row, shot_id=shot))
    return out


def scan(video, deadline, threshold=0.10):
    import av
    import numpy as np

    previous, records = None, []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            deadline.require()
            if frame.pts is None or frame.time_base is None:
                raise ValueError("source_frame_missing_PTS")
            seconds = float(frame.pts * frame.time_base)
            if records and seconds <= records[-1].seconds:
                raise ValueError("non_increasing_source_PTS")
            thumb = (
                frame.reformat(width=32, height=32, format="rgb24").to_ndarray().astype(float) / 255
            )
            motion = float(np.abs(thumb - previous).mean()) if previous is not None else 0.0
            records.append(
                Frame(
                    index,
                    frame.pts,
                    (frame.time_base.numerator, frame.time_base.denominator),
                    seconds,
                    0,
                    motion,
                )
            )
            previous = thumb
    if len(records) < 2:
        raise ValueError("need_at_least_two_video_frames")
    return assign_shots(records, threshold)


def uniform(rows, count):
    """Nearest source frames to equally spaced *times*, including both endpoints."""
    if len(rows) <= count:
        return list(rows)
    times = [r.seconds for r in rows]
    chosen = set()
    for k in range(count):
        t = times[0] + (times[-1] - times[0]) * k / (count - 1)
        i = bisect_left(times, t)
        i = min(i, len(rows) - 1)
        if i and abs(times[i - 1] - t) < abs(times[i] - t):
            i -= 1
        chosen.add(i)
    return [rows[i] for i in sorted(chosen)]


def plan(records, *, max_seconds=12.0, sample_fps=4.0, context_seconds=1.5, max_frames=64):
    if max_seconds <= 0 or sample_fps <= 0 or context_seconds < 0 or max_frames < 4:
        raise ValueError("invalid_clip_budget")
    dt = median(b.seconds - a.seconds for a, b in zip(records, records[1:]))
    end = records[-1].seconds + dt
    times = [r.seconds for r in records]
    groups = []
    for row in records:
        if (
            not groups
            or row.shot_id != groups[-1][0].shot_id
            or row.seconds - groups[-1][0].seconds >= max_seconds
        ):
            groups.append([])
        groups[-1].append(row)
    result = []
    for i, group in enumerate(groups):
        start = group[0].seconds
        stop = groups[i + 1][0].seconds if i + 1 < len(groups) else end
        # Allocate target first. Context cannot crowd it out.
        wanted_target = min(len(group), max(2, math.ceil((stop - start) * sample_fps)))
        target = uniform(group, min(wanted_target, max_frames))
        before = records[bisect_left(times, start - context_seconds) : bisect_left(times, start)]
        after = records[bisect_left(times, stop) : bisect_left(times, stop + context_seconds)]
        wanted_context = []
        for side in [before, after]:
            if side:
                wanted_context.extend(
                    uniform(side, max(2, math.ceil(context_seconds * sample_fps)))
                )
        room = max_frames - len(target)
        context = uniform(wanted_context, room) if room >= 2 and wanted_context else []
        rows = sorted(target + context, key=lambda r: r.index)
        result.append(
            (
                group[0].shot_id,
                start,
                stop,
                rows,
                {
                    "requested_target_frames": wanted_target,
                    "selected_target_frames": len(target),
                    "requested_context_frames": len(wanted_context),
                    "selected_context_frames": len(context),
                    "target_max_sample_gap_seconds": max(
                        (b.seconds - a.seconds for a, b in zip(target, target[1:])), default=0
                    ),
                    "target_budget_limited": len(target) < wanted_target,
                    "shot_boundary_is_event_boundary": False,
                },
            )
        )
    return result


def prepare(video, out, *, deadline=None, **config):
    import av

    deadline = deadline or Deadline()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = scan(video, deadline, config.pop("shot_threshold", 0.10))
    windows = plan(records, **config)
    atomic_json(out / "scan.json", [asdict(r) for r in records])
    video_id = "video:" + digest(video)[:20]
    wanted = {r.index: r for _, _, _, rows, _ in windows for r in rows}
    images = {}
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            deadline.require()
            if index not in wanted:
                continue
            r = wanted[index]
            if frame.pts != r.pts:
                raise ValueError("source_PTS_changed")
            path = frames_dir / f"{index:08d}.jpg"
            frame.to_image().save(path, quality=92)
            images[index] = Media(
                f"{video_id}:frame:{index}",
                str(path.resolve()),
                r.pts,
                r.time_base,
                digest(path),
                index,
            )
    if set(images) != set(wanted):
        raise ValueError("missing_decoded_frames")
    dt = median(b.seconds - a.seconds for a, b in zip(records, records[1:]))
    fps = 1 / dt
    cfr = all(
        abs((r.seconds - records[0].seconds) - r.index / fps) <= max(1e-4, dt * 0.01)
        for r in records
    )
    duration = records[-1].seconds - records[0].seconds + dt
    clips = [
        Clip(
            f"{video_id}:clip:{i:05d}",
            video_id,
            shot,
            start,
            stop,
            tuple(images[r.index] for r in rows),
            fps,
            len(records),
            duration,
            records[0].seconds,
            cfr,
            report,
        )
        for i, (shot, start, stop, rows, report) in enumerate(windows)
    ]
    atomic_json(out / "clips.json", [asdict(c) for c in clips])
    return clips
