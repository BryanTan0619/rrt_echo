from perception.local_stage import reference_profile


def test_reference_profile_reads_instance_attributes():
    observation = {
        "media": [{"media_id": "f0", "pts": 0, "time_base": [1, 1]}],
        "facts": [],
    }
    instance = {
        "instance_id": "i0",
        "description": "person in leather jacket",
        "regions": [{"media_id": "f0", "box": [0.1, 0.1, 0.5, 0.5]}],
        "attributes": [{"dimension": "attire", "value": "leather jacket"}],
    }
    profile = reference_profile(observation, instance)
    assert any(
        c["property"] == "attire" and c["value"] == "leather jacket"
        for c in profile["cues"]
    )
    assert profile["gap"] is None


def test_reference_profile_handles_tuple_attributes():
    observation = {"media": [], "facts": []}
    instance = {
        "instance_id": "i0",
        "description": "food",
        "regions": [{"media_id": "f0", "box": [0, 0, 1, 1]}],
        "attributes": (("shape", "umbrella"), ("color", "brown")),
    }
    profile = reference_profile(observation, instance)
    vals = {(c["property"], c["value"]) for c in profile["cues"]}
    assert ("shape", "umbrella") in vals
    assert ("color", "brown") in vals
