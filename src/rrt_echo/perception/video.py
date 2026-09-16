"""Offline shot/motion scan and bounded, query-blind sampling with source PTS."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median

from ..runtime import Deadline
from ..schema import Media
from ..storage import atomic_json


@dataclass(frozen=True)
class FrameIndex:
    index: int
    pts: int
    time_base: tuple[int, int]
    seconds: float
    shot_id: int
    motion: float


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan(video: Path, deadline: Deadline, *, shot_threshold=0.10) -> list[FrameIndex]:
    import av
    import numpy as np

    records, previous, shot = [], None, 0
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            deadline.require()
            if frame.pts is None or frame.time_base is None or frame.pts < 0:
                raise ValueError("video frame lacks usable nonnegative source PTS")
            thumb = (
                frame.reformat(width=32, height=32, format="rgb24").to_ndarray().astype(float) / 255
            )
            motion = float(np.abs(thumb - previous).mean()) if previous is not None else 0.0
            if previous is not None and motion >= shot_threshold:
                shot += 1
            previous = thumb
            records.append(
                FrameIndex(
                    index,
                    frame.pts,
                    (frame.time_base.numerator, frame.time_base.denominator),
                    float(frame.pts * frame.time_base),
                    shot,
                    motion,
                )
            )
    if not records:
        raise ValueError("video has no decoded frames")
    return assign_shots(records, threshold=shot_threshold)


def assign_shots(records, *, threshold=0.10, minimum_gap=0.25):
    """Offline cut proposals: absolute change plus local contrast and suppression.

    This remains a lightweight cut detector; cuts are proposals, not event IDs.
    """
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
    shot = 0
    result = []
    for row in records:
        if row.index in cuts:
            shot += 1
        result.append(replace(row, shot_id=shot))
    return result


def select_windows(
    index: list[FrameIndex], *, window_seconds=8.0, base_fps=1.0, high_fps=8.0, max_frames=16
) -> tuple[list[list[FrameIndex]], list[dict]]:
    if min(window_seconds, base_fps, high_fps) <= 0 or max_frames < 2:
        raise ValueError("invalid sampling settings")
    groups = []
    for row in index:
        if (
            not groups
            or row.shot_id != groups[-1][0].shot_id
            or row.seconds - groups[-1][0].seconds >= window_seconds
        ):
            groups.append([])
        groups[-1].append(row)
    windows, reports = [], []
    for group in groups:
        selected = {group[0].index: group[0], group[-1].index: group[-1]}
        last = group[0].seconds
        for row in group:
            if row.seconds - last >= 1 / base_fps:
                selected[row.index] = row
                last = row.seconds
        peak = max(group, key=lambda r: r.motion)
        if peak.motion > 0.03:  # generic motion trigger, not a question/video rule
            last = float("-inf")
            for row in group:
                if abs(row.seconds - peak.seconds) <= 0.5 and row.seconds - last >= 1 / high_fps:
                    selected[row.index] = row
                    last = row.seconds
        wanted = len(selected)
        if wanted > max_frames:
            middle = sorted(
                (r for r in selected.values() if r not in (group[0], group[-1])),
                key=lambda r: (-r.motion, r.index),
            )[: max_frames - 2]
            selected = {r.index: r for r in [group[0], *middle, group[-1]]}
        windows.append(sorted(selected.values(), key=lambda r: r.index))
        reports.append(
            {
                "shot_id": group[0].shot_id,
                "range": [group[0].seconds, group[-1].seconds],
                "requested_frames": wanted,
                "selected_frames": len(selected),
                "omitted_for_frame_budget": wanted - len(selected),
            }
        )
    return windows, reports


def decode_selected(video: Path, out: Path, video_id: str, windows, deadline: Deadline):
    import av

    wanted = {r.index: r for window in windows for r in window}
    media = {}
    out.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            deadline.require()
            if i not in wanted:
                continue
            row = wanted[i]
            if frame.pts != row.pts:
                raise ValueError("source PTS changed between scan and extraction")
            path = out / f"frame_{i:08d}.jpg"
            frame.to_image().save(path, quality=90)
            media[i] = Media(
                f"{video_id}:frame:{i}",
                str(path.resolve()),
                row.pts,
                row.time_base,
                file_hash(path),
            )
    if set(media) != set(wanted):
        raise ValueError("not all selected source frames decoded")
    return [tuple(media[r.index] for r in window) for window in windows]


def prepare(video: Path, out: Path, deadline: Deadline, config: dict):
    records = scan(video, deadline, shot_threshold=config["shot_threshold"])
    atomic_json(out / "scan.json", [asdict(r) for r in records])
    windows, report = select_windows(
        records,
        window_seconds=config["window_seconds"],
        base_fps=config["base_fps"],
        high_fps=config["high_fps"],
        max_frames=config["max_frames"],
    )
    atomic_json(out / "sampling.json", report)
    video_id = "video:" + file_hash(video)[:20]
    media = decode_selected(video, out / "frames", video_id, windows, deadline)
    return video_id, [(str(rows[0].shot_id), frames) for rows, frames in zip(windows, media)]
