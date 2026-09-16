import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    index: int
    pts: int
    time_base: tuple[int, int]
    seconds: float
    shot_id: int
    motion: float


@dataclass(frozen=True)
class Media:
    media_id: str
    uri: str
    pts: int
    time_base: tuple[int, int]
    sha256: str
    index: int

    @property
    def seconds(self):
        return self.pts * self.time_base[0] / self.time_base[1]


@dataclass(frozen=True)
class Clip:
    clip_id: str
    video_id: str
    shot_id: int
    target_start: float
    target_end: float  # exclusive scheduling boundary, NOT an event boundary
    media: tuple[Media, ...]
    fps: float
    total_frames: int
    duration: float
    start_seconds: float
    constant_frame_rate: bool
    sampling: dict

    @property
    def target_ids(self):
        return {m.media_id for m in self.media if self.target_start <= m.seconds < self.target_end}


@dataclass(frozen=True)
class Reference:
    ref_id: str
    instance_id: str  # versioned external endpoint; NOT a name inferred by this package
    kind: str
    media: tuple[Media, ...]  # cropped or marked full-frame views, with source mappings below
    sources: tuple[dict, ...]


class Deadline:
    def __init__(self, seconds=3600):
        self.ends = time.monotonic() + seconds

    def remaining(self):
        return max(0.0, self.ends - time.monotonic())

    def require(self):
        if self.remaining() <= 0:
            raise TimeoutError("budget_exhausted")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def clip_from_dict(d):
    return Clip(**{**d, "media": tuple(Media(**m) for m in d["media"])})
