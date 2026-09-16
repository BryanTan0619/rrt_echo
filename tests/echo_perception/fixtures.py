import json

from PIL import Image

from rrt_echo.echo_perception.types import Clip, Media, digest


def clips_at(tmp_path):
    media = []
    for i in range(6):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (100, 100), (20 * i, 30, 40)).save(path)
        media.append(Media(f"v:frame:{i}", str(path), i, (1, 1), digest(path), i))
    return [
        Clip("v:clip:0", "v", 0, 1, 3, tuple(media[:4]), 1, 6, 6, 0, True, {}),
        Clip("v:clip:1", "v", 1, 3, 5, tuple(media[2:]), 1, 6, 6, 0, True, {}),
    ]


def response(raw):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]}
