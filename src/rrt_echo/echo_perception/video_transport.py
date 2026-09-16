"""Explicit metadata envelope for legacy vLLM. Not a standard video MIME format."""

import base64
import json
import math

MIME = "video/x-rrt-jpeg-v1"


def validate_metadata(metadata, count):
    fps = metadata.get("fps")
    indices = metadata.get("frames_indices", [])
    total = metadata.get("total_num_frames")
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid_source_fps")
    if not isinstance(total, int) or total < 1 or count < 2 or count > 512:
        raise ValueError("invalid_frame_count")
    if len(indices) != count or any(type(i) is not int or not 0 <= i < total for i in indices):
        raise ValueError("frame_index_mismatch")
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("non_increasing_frame_indices")
    if metadata.get("do_sample_frames") is not False:
        raise ValueError("resampling_not_allowed")
    duration = metadata.get("duration")
    if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid_duration")
    if abs(total / fps - duration) > max(1e-4, 0.02 / fps):
        raise ValueError("source_duration_mismatch")
    return metadata


def pack(frames, metadata):
    validate_metadata(metadata, len(frames))
    data = json.dumps(
        {"version": 1, "frames": frames, "metadata": metadata}, separators=(",", ":")
    ).encode()
    return "data:" + MIME + ";base64," + base64.b64encode(data).decode()


def unpack(data):
    value = json.loads(base64.b64decode(data, validate=True))
    if value.get("version") != 1 or not isinstance(value.get("frames"), list):
        raise ValueError("invalid_rrt_video_envelope")
    validate_metadata(value["metadata"], len(value["frames"]))
    return value["frames"], value["metadata"]
