import json

from rrt_echo.rrt.recover_links import recover_links


def test_invalid_pair_does_not_discard_valid_sibling(tmp_path):
    instances = [
        {"instance_id": name, "kind": "person", "regions": [{"media_id": mid}]}
        for name, mid in [("a", "m0"), ("b", "m1"), ("c", "m2")]
    ]
    results = [
        {
            "observation": {
                "observation_id": "o",
                "instances": instances,
                "media": [{"media_id": m} for m in ["m0", "m1", "m2"]],
            }
        }
    ]
    pairs = [
        {"pair_id": "valid", "left": "a", "right": "b", "relation": "same_identity"},
        {"pair_id": "invalid", "left": "a", "right": "c", "relation": "same_identity"},
    ]
    call = tmp_path / "association/calls/one"
    call.mkdir(parents=True)
    (call / "request.json").write_text(
        json.dumps({"pairs": pairs, "media": [{"media_id": m} for m in ["m0", "m1", "m2"]]})
    )
    raw = {
        "correspondences": [
            {
                "pair_id": "valid",
                "verdict": "same",
                "evidence_ids": ["f0", "f0", "f1"],
                "basis": "same visible person",
            },
            {
                "pair_id": "invalid",
                "verdict": "same",
                "evidence_ids": ["f0"],
                "basis": "missing other endpoint",
            },
        ]
    }
    (call / "response.json").write_text(
        json.dumps(
            {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]}
        )
    )
    recovered = {
        p["proposal_id"]: p
        for p in recover_links(results, tmp_path / "association", tmp_path / "recovered")
    }
    assert recovered["valid"]["verdict"] == "supported"
    assert recovered["valid"]["evidence_ids"] == ["m0", "m1"]
    assert recovered["invalid"]["verdict"] == "unresolved"
    assert recovered["invalid"]["row_error"]
