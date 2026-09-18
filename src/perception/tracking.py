"""Shot-local motion association. Tracks propose continuity, never global identity."""

from __future__ import annotations

from dataclasses import dataclass


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union else 0.0


@dataclass
class Track:
    track_id: str
    box: tuple[float, ...]
    time: float
    velocity: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


class LocalTracker:
    def __init__(self, *, max_gap=1.5, minimum_iou=0.25):
        self.max_gap = max_gap
        self.minimum_iou = minimum_iou
        self.shot = None
        self.active: dict[str, Track] = {}
        self.serial = 0

    def update(self, boxes, *, seconds, shot_id):
        if shot_id != self.shot:
            self.active.clear()
            self.shot = shot_id
        self.active = {
            k: v for k, v in self.active.items() if 0 <= seconds - v.time <= self.max_gap
        }
        candidates = []
        for key, track in self.active.items():
            dt = seconds - track.time
            predicted = tuple(x + v * dt for x, v in zip(track.box, track.velocity))
            for index, box in enumerate(boxes):
                score = iou(predicted, box)
                if score >= self.minimum_iou:
                    candidates.append((-score, key, index))
        assigned, used = {}, set()
        for _, key, index in sorted(candidates):
            if index not in assigned and key not in used:
                assigned[index] = key
                used.add(key)
        result = []
        for index, box in enumerate(boxes):
            key = assigned.get(index)
            if key is None:
                self.serial += 1
                key = f"body_track_{self.serial:06d}"
            previous = self.active.get(key)
            dt = seconds - previous.time if previous else 0
            velocity = (
                tuple((x - y) / dt for x, y in zip(box, previous.box)) if dt > 0 else (0.0,) * 4
            )
            self.active[key] = Track(key, tuple(box), seconds, velocity)
            result.append({"track_ref": key, "box": list(box), "status": "candidate"})
        return result


class TorchvisionDetector:
    """Optional backend. Requires explicitly supplied weights; never downloads."""

    def __init__(self, weights, device="cpu", threshold=0.7, architecture="v1"):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2

        self.torch = torch
        self.device = device
        self.threshold = threshold
        constructors = {"v1": fasterrcnn_resnet50_fpn, "v2": fasterrcnn_resnet50_fpn_v2}
        if architecture not in constructors:
            raise ValueError("unknown detector architecture")
        self.model = constructors[architecture](weights=None, weights_backbone=None)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(device).eval()

    def detect(self, path):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        image = Image.open(path).convert("RGB")
        w, h = image.size
        with self.torch.inference_mode():
            result = self.model([to_tensor(image).to(self.device)])[0]
        return [
            tuple(float(x) / scale for x, scale in zip(box, (w, h, w, h)))
            for box, label, score in zip(
                result["boxes"].cpu(), result["labels"].cpu(), result["scores"].cpu()
            )
            if int(label) == 1 and float(score) >= self.threshold
        ]
