from copy import deepcopy

import pytest
from conftest import make_packet

from rrt_echo.identity import IdentityProposal
from rrt_echo.memory import Memory
from rrt_echo.schema import ObservationPacket, Region


def proposal(i, left, right, verdict="same"):
    return IdentityProposal(
        i,
        left + ":p",
        right + ":p",
        verdict,
        (left + ":frame", right + ":frame"),
        "explicit visual comparison",
    )


def test_wire_roundtrip_is_immutable(packet):
    assert ObservationPacket.from_dict(packet.to_dict()) == packet
    data = packet.to_dict()
    data["facts"][0]["roles"]["agent"] = "missing"
    assert dict(packet.facts[0].roles)["agent"] == "o1:p"
    with pytest.raises(ValueError):
        ObservationPacket.from_dict(data)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, float("nan"), 1]])
def test_invalid_regions(box):
    with pytest.raises(ValueError):
        Region("frame", tuple(box))


def test_unknown_fields_cannot_grant_commit_authority(packet):
    data = packet.to_dict() | {"accepted_identity": True}
    with pytest.raises(ValueError):
        ObservationPacket.from_dict(data)


def test_append_rejects_duplicate_instances_and_video_mix(packet):
    memory = Memory()
    memory.append(packet)
    with pytest.raises(ValueError):
        memory.append(packet)
    data = make_packet("o2").to_dict()
    data["video_id"] = "other"
    with pytest.raises(ValueError):
        memory.append(ObservationPacket.from_dict(data))


def test_identity_does_not_rewrite_local_event_and_revoke_restores_split():
    memory = Memory()
    memory.append(make_packet("a"))
    memory.append(make_packet("b", 10))
    observations = [p.to_dict() for p in memory.packets]
    before = memory.snapshot()
    decision = memory.compare(proposal("ab", "a", "b"))
    merged = memory.snapshot()
    assert len(merged["entities"]) == 1
    assert merged["facts"][0]["roles"] == before["facts"][0]["roles"]
    memory.identity.revoke(decision.decision_id, reason="reviewed incorrect correspondence")
    split = memory.snapshot()
    assert len(split["entities"]) == 2
    assert split["facts"][0]["support"]["global_identity"]["agent"]["status"] == "insufficient"
    assert observations == [p.to_dict() for p in memory.packets]
    assert len(merged["entities"]) == 1  # previous snapshot remains immutable


def test_transitive_conflict_defers_merge():
    memory = Memory()
    for name in "abc":
        memory.append(make_packet(name))
    memory.compare(proposal("ac", "a", "c", "different"))
    memory.compare(proposal("ab", "a", "b"))
    decision = memory.compare(proposal("bc", "b", "c"))
    assert decision.status == "deferred" and decision.reason == "cluster_conflict"
    assert len(memory.snapshot()["entities"]) == 2


def test_later_distinctness_invalidates_existing_cluster():
    memory = Memory()
    for name in "abc":
        memory.append(make_packet(name))
    memory.compare(proposal("ab", "a", "b"))
    memory.compare(proposal("bc", "b", "c"))
    change = memory.compare(proposal("ac", "a", "c", "different"))
    assert len(change.invalidates) == 2
    assert len(memory.snapshot()["entities"]) == 3
    assert all(not f["dependencies"]["identity"] for f in memory.snapshot()["facts"])


def test_body_visual_evidence_does_not_require_face():
    memory = Memory()
    memory.append(make_packet("a"))
    memory.append(make_packet("b"))
    assert memory.compare(proposal("ab", "a", "b")).status == "accepted"
    bad = IdentityProposal("bad", "a:p", "b:p", "same", ("a:frame",), "same shirt")
    assert memory.compare(bad).status == "deferred"


def test_unresolved_is_not_different():
    memory = Memory()
    memory.append(make_packet("a"))
    memory.append(make_packet("b"))
    memory.compare(proposal("ab", "a", "b", "unresolved"))
    assert not memory.identity.active()


def test_joint_roles_require_own_region_evidence(packet):
    data = packet.to_dict()
    data["facts"][0]["evidence_by_slot"]["agent"] = []
    memory = Memory()
    memory.append(ObservationPacket.from_dict(data))
    assert memory.snapshot()["facts"][0]["support"]["local_joint"]["status"] == "insufficient"


def test_local_event_support_is_not_global_identity_support(packet):
    memory = Memory()
    memory.append(packet)
    support = memory.snapshot()["facts"][0]["support"]
    assert support["local_joint"]["status"] == "supported"
    assert support["global_identity"]["agent"]["status"] == "insufficient"
    assert support["duration"]["status"] == "insufficient"
    assert support["persistence"]["status"] == "insufficient"


def test_fact_evidence_cannot_reference_another_packet(packet):
    data = deepcopy(packet.to_dict())
    data["facts"][0]["joint_evidence"] = ["other:frame"]
    with pytest.raises(ValueError):
        ObservationPacket.from_dict(data)
