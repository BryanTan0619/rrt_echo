from PIL import Image

from rrt_echo.echo_perception.types import digest
from rrt_echo.rrt.association import candidate_pairs, motion_candidates


def observations(tmp_path):
    results = []
    for n in range(3):
        p = tmp_path / f"{n}.jpg"
        Image.new("RGB", (64, 64), (100, 30, 10)).save(p)
        mid = f"f{n}"
        results.append(
            {
                "observation": {
                    "observation_id": f"o{n}",
                    "media": [
                        {
                            "media_id": mid,
                            "uri": str(p),
                            "pts": n,
                            "time_base": [1, 1],
                            "sha256": digest(p),
                            "index": n,
                        }
                    ],
                    "instances": [
                        {
                            "instance_id": f"i{n}",
                            "kind": "person",
                            "description": "person in red",
                            "regions": [{"media_id": mid, "box": [0.1, 0.1, 0.8, 0.9]}],
                        }
                    ],
                    "facts": [],
                },
                "link_proposals": [],
            }
        )
    return results


def test_gallery_retrieves_future_endpoints_without_committing(tmp_path):
    results = observations(tmp_path)
    pairs = candidate_pairs(results, top_k=1)
    assert any(p["left"] == "i0" and p["right"] == "i1" for p in pairs)
    assert all("verdict" not in p for p in pairs)


def test_motion_does_not_cross_unknown_or_different_shots(tmp_path):
    results = observations(tmp_path)
    assert not motion_candidates(results)["pairs"]
    assert not motion_candidates(results, {"f0": 0, "f1": 1, "f2": 2})["pairs"]
    motion = motion_candidates(results, {"f0": 0, "f1": 0, "f2": 0})
    assert len(motion["pairs"]) == 2 and motion["identity_accepted"] is False


def test_owner_candidates_include_local_competitors_and_future(tmp_path):
    from rrt_echo.rrt.association import ownership_candidates

    results = observations(tmp_path)
    results[0]["observation"]["instances"].append(
        {
            "instance_id": "arm",
            "kind": "region",
            "description": "forearm",
            "regions": [{"media_id": "f0", "box": [0.2, 0.2, 0.4, 0.5]}],
        }
    )
    pairs = ownership_candidates(results, top_k=1)
    assert any(p["right"] == "i0" for p in pairs)
    assert any(p["right"] == "i1" for p in pairs)
    assert all(p["left"] == "arm" and p["relation"] == "part_of" for p in pairs)
    assert all("verdict" not in p for p in pairs)


def test_same_observation_duplicates_are_visual_candidates(tmp_path):
    import copy

    results = observations(tmp_path)
    duplicate = copy.deepcopy(results[0]["observation"]["instances"][0])
    duplicate["instance_id"] = "same_clip_other_fact"
    results[0]["observation"]["instances"].append(duplicate)
    pairs = candidate_pairs(results, top_k=1)
    assert any(set((p["left"], p["right"])) == {"i0", "same_clip_other_fact"} for p in pairs)
    assert any("i0" in (p["left"], p["right"]) and "i1" in (p["left"], p["right"]) for p in pairs)
    assert all("verdict" not in p for p in pairs)


def test_bounded_batches_reserve_ownership_and_skip_previously_compared(tmp_path, monkeypatch):
    import copy

    from rrt_echo.echo_perception.types import Deadline
    from rrt_echo.rrt import association

    rows = observations(tmp_path)
    for n in range(3, 9):
        inst = copy.deepcopy(rows[0]["observation"]["instances"][0])
        inst["instance_id"] = f"i{n}"
        rows[0]["observation"]["instances"].append(inst)
    arm = copy.deepcopy(rows[0]["observation"]["instances"][0])
    arm.update(instance_id="arm", kind="region")
    rows[0]["observation"]["instances"].append(arm)
    people = [
        {"left": "i0", "right": f"i{n}", "relation": "same_identity", "source": "test"}
        for n in range(1, 9)
    ]
    owners = [{"left": "arm", "right": "i0", "relation": "part_of", "source": "test"}]
    monkeypatch.setattr(association, "candidate_pairs", lambda *args: people)
    monkeypatch.setattr(association, "ownership_candidates", lambda *args: owners)
    seen = []

    def compare(self, **kwargs):
        seen.append(kwargs["pairs"])
        return []

    monkeypatch.setattr(association.CorrespondenceClient, "compare", compare)
    association.resolve(
        rows,
        model="test",
        base_url="unused",
        out=tmp_path / "run",
        deadline=Deadline(100),
        max_batches=2,
        skip_observed_pairs=True,
        previous_pairs=[people[0]],
    )
    assert len(seen) == 2
    assert all(p["relation"] == "same_identity" for p in seen[0])
    assert all(p["relation"] == "part_of" for p in seen[1])
    assert not any(p["left"] == "i0" and p["right"] == "i1" for batch in seen for p in batch)


def test_representative_regions_temporal_spread():
    from rrt_echo.rrt.association import representative_regions

    instance = {
        "regions": [
            {"media_id": "f0", "box": [0.0, 0.0, 0.2, 0.2]},  # t=0, 面积小
            {"media_id": "f1", "box": [0.0, 0.0, 0.9, 0.9]},  # t=10, 面积最大
            {"media_id": "f2", "box": [0.1, 0.1, 0.3, 0.3]},  # t=20
            {"media_id": "f3", "box": [0.1, 0.1, 0.2, 0.2]},  # t=30
            {"media_id": "f4", "box": [0.0, 0.0, 0.2, 0.2]},  # t=40
        ]
    }
    media = {f"f{n}": {"pts": n * 10, "time_base": [1, 1]} for n in range(5)}
    selected = representative_regions(instance, media, limit=3)
    # 时间均匀分布选第一/中间/最后，而非面积最大的 f1
    assert [r["media_id"] for r in selected] == ["f0", "f2", "f4"]


def test_candidate_pairs_identifier_index(tmp_path):
    from rrt_echo.rrt.association import candidate_pairs

    results = observations(tmp_path)
    results[0]["observation"]["instances"][0]["description"] = "man, number 398"
    results[1]["observation"]["instances"][0]["description"] = "man, number 398"
    results[2]["observation"]["instances"][0]["description"] = "man, number 323"
    pairs = candidate_pairs(results, top_k=1)
    indexed = [p for p in pairs if p["source"] == "inscribed_identifier_match"]
    assert any(set((p["left"], p["right"])) == {"i0", "i1"} for p in indexed)
    assert all(p["priority"] == 5.0 for p in indexed)
    # 不同编号的 i0-i2 / i1-i2 被跳过
    assert not any(set((p["left"], p["right"])) == {"i0", "i2"} for p in pairs)
    assert not any(set((p["left"], p["right"])) == {"i1", "i2"} for p in pairs)
