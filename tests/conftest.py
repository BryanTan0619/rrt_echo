import hashlib

import pytest

from rrt_echo.schema import ObservationPacket


def make_packet(name="o1", time=0, predicate="carry", value=None, kind="event"):
    mid = name + ":frame"
    owner_role = "agent" if kind == "event" else "owner"
    return ObservationPacket.from_dict(
        {
            "observation_id": name,
            "video_id": "v",
            "shot_id": name,
            "media": [
                {
                    "media_id": mid,
                    "uri": "synthetic://" + mid,
                    "pts": time,
                    "time_base": [1, 1],
                    "sha256": hashlib.sha256(mid.encode()).hexdigest(),
                }
            ],
            "instances": [
                {
                    "instance_id": name + ":p",
                    "kind": "person",
                    "regions": [{"media_id": mid, "box": [0.1, 0.1, 0.4, 0.9]}],
                }
            ],
            "facts": [
                {
                    "fact_id": name + ":event",
                    "kind": kind,
                    "predicate": predicate,
                    "roles": {owner_role: name + ":p"},
                    "evidence_by_slot": {"predicate": [mid], owner_role: [mid], "value": [mid]},
                    "joint_evidence": [mid],
                    "observed_media_ids": [mid],
                    "value": value,
                }
            ],
        }
    )


@pytest.fixture
def packet():
    return make_packet()
