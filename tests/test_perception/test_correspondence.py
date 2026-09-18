import copy

import pytest
from jsonschema import ValidationError

from perception.correspondence import validate_correspondences


def data():
    instances = {
        "a": {"instance_id": "a", "kind": "person", "regions": [{"media_id": "f1"}]},
        "b": {"instance_id": "b", "kind": "person", "regions": [{"media_id": "f2"}]},
    }
    pairs = [{"pair_id": "p", "left": "a", "right": "b", "relation": "same_identity"}]
    row = {
        "pair_id": "p",
        "verdict": "same",
        "evidence_ids": ["f1", "f2"],
        "basis": "visual correspondence",
        "gap_reason": "none",
    }
    return instances, pairs, {"correspondences": [row]}


def test_context_proposal_cannot_write_local_facts_or_commit_identity():
    instances, pairs, raw = data()
    original = copy.deepcopy(instances)
    result = validate_correspondences(raw, pairs, instances, ["f1", "f2"])
    assert result[0]["status"] == "proposed" and instances == original
    raw["facts"] = []
    with pytest.raises(ValidationError):
        validate_correspondences(raw, pairs, instances, ["f1", "f2"])


def test_missing_endpoint_is_not_supported():
    instances, pairs, raw = data()
    raw["correspondences"][0]["evidence_ids"] = ["f1"]
    with pytest.raises(ValueError, match="missing_endpoint"):
        validate_correspondences(raw, pairs, instances, ["f1", "f2"])


def test_omitted_correspondence_stays_unresolved():
    instances, pairs, _ = data()
    result = validate_correspondences({"correspondences": []}, pairs, instances, ["f1", "f2"])
    assert result[0]["verdict"] == "unresolved" and result[0]["gap_reason"] == "omitted_pair"


def test_comparison_aliases_roundtrip_and_cache(tmp_path):
    from PIL import Image

    from perception.correspondence import CorrespondenceClient
    from perception.types import Media, digest

    frames = []
    instances = {}
    for n in range(2):
        p = tmp_path / f"{n}.jpg"
        Image.new("RGB", (100, 100), "black").save(p)
        frames.append(Media(f"original-frame-{n}", str(p), n, (1, 1), digest(p), n))
        instances[f"entity-{n}"] = {
            "instance_id": f"entity-{n}",
            "kind": "person",
            "regions": [{"media_id": frames[-1].media_id, "box": [0.1, 0.1, 0.8, 0.9]}],
        }
    pairs = [{"pair_id": "p", "left": "entity-0", "right": "entity-1", "relation": "same_identity"}]
    calls = []

    def sender(body):
        calls.append(body)
        assert "P0" in body["messages"][0]["content"][-1]["text"]
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"correspondences":[{"pair_id":"p","verdict":"different","evidence_ids":["f0","f1"],"basis":"distinct visible people","gap_reason":"none"}]}'
                    },
                }
            ]
        }

    client = CorrespondenceClient(model="test", base_url="http://test")
    kwargs = dict(
        pairs=pairs, instances=instances, media=frames, out=tmp_path / "calls", sender=sender
    )
    result = client.compare(**kwargs)
    assert result[0]["evidence_ids"] == ["original-frame-0", "original-frame-1"]
    assert client.compare(**kwargs) == result and len(calls) == 1
