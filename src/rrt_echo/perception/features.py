"""Optional face features for candidate ranking, never identity acceptance."""

from __future__ import annotations

import math
from pathlib import Path


def cosine(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def attach_face_candidates(instances, faces):
    """Only rank uniquely contained regions; compound-owner ambiguity stays open."""
    features, unresolved = {}, []
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        area = (x2 - x1) * (y2 - y1)
        owners = set()
        for instance in instances:
            for region in instance.regions:
                if region.media_id != face["media_id"]:
                    continue
                a, b, c, d = region.box
                overlap = max(0, min(c, x2) - max(a, x1)) * max(0, min(d, y2) - max(b, y1))
                if area > 0 and overlap / area >= 0.9:
                    owners.add(instance.instance_id)
        if len(owners) == 1:
            features.setdefault(next(iter(owners)), []).append(tuple(face["embedding"]))
        else:
            unresolved.append(
                {
                    "media_id": face["media_id"],
                    "box": face["box"],
                    "candidate_owners": sorted(owners),
                }
            )
    return features, unresolved


class InsightFaceFeatures:
    """Use preinstalled weights and explicitly report real ONNX providers."""

    def __init__(self, root: str, *, provider="CPUExecutionProvider", model_pack="buffalo_l"):
        model_dir = Path(root) / "models" / model_pack
        if not model_dir.is_dir() or not list(model_dir.glob("*.onnx")):
            raise FileNotFoundError("install InsightFace ONNX model pack under root/models first")
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        if provider not in ort.get_available_providers():
            raise RuntimeError(f"requested ONNX provider unavailable: {provider}")
        self.app = FaceAnalysis(name=model_pack, root=root, providers=[provider])
        self.app.prepare(
            ctx_id=0 if provider == "CUDAExecutionProvider" else -1, det_size=(640, 640)
        )
        self.providers = {
            name: model.session.get_providers() for name, model in self.app.models.items()
        }
        if any(provider not in values for values in self.providers.values()):
            raise RuntimeError(f"ONNX provider fallback detected: {self.providers}")

    def extract(self, media):
        import cv2

        image = cv2.imread(media.uri)
        if image is None:
            raise ValueError("cannot decode face evidence frame")
        h, w = image.shape[:2]
        result = []
        for face in self.app.get(image):
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            box = [float(x) / scale for x, scale in zip(face.bbox, (w, h, w, h))]
            result.append(
                {
                    "media_id": media.media_id,
                    "box": box,
                    "embedding": [float(x) for x in embedding],
                    "score": float(face.det_score),
                    "role": "candidate_feature_only",
                }
            )
        return result
