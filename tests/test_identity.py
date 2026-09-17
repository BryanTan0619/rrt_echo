from rrt_echo.identity import (
    IdentityGraph,
    IdentityProposal,
    discriminative_identifiers,
    extract_identifiers,
)
from rrt_echo.schema import Instance, Region


def mk(instance_id, description, kind="person"):
    return Instance(instance_id, kind, (Region("m0", (0.1, 0.1, 0.4, 0.4)),), description)


def test_discriminative_identifiers_extraction():
    cases = {
        "man in green tracksuit, number 398": {"398"},
        "person with number 05 on front": {"5"},
        "man with number 044 on his chest": {"44"},
        "player in green tracksuit, number 456": {"456"},
        "man in a dark coat, pointing": set(),
        "18-inch weapon under jacket": set(),
        "contestant in green tracksuit, number 02": {"2"},
    }
    for desc, expected in cases.items():
        assert discriminative_identifiers(mk("i", desc)) == expected, desc


def test_extract_identifiers_accepts_plain_string():
    assert extract_identifiers("number 398 on back") == {"398"}
    assert extract_identifiers("number 05") == {"5"}
    assert extract_identifiers("no explicit label") == set()
    # discriminative_identifiers delegates to the same extraction
    assert discriminative_identifiers(mk("i", "vest 067")) == extract_identifiers("vest 067")


def test_different_identifiers_veto_same():
    graph = IdentityGraph()
    instances = {
        "i0": mk("i0", "man in green tracksuit, number 398"),
        "i1": mk("i1", "man in green tracksuit, number 323"),
    }
    decision = graph.propose(
        IdentityProposal("p1", "i0", "i1", "same", ("m0",), "same green tracksuit"), instances
    )
    assert decision.status == "deferred"
    assert decision.reason == "inscribed_identifier_conflict"


def test_same_identifier_allows_same():
    graph = IdentityGraph()
    instances = {
        "i0": mk("i0", "number 398 curly hair"),
        "i1": mk("i1", "number 398 on back"),
    }
    decision = graph.propose(
        IdentityProposal("p1", "i0", "i1", "same", ("m0",), "same person"), instances
    )
    assert decision.status == "accepted"


def test_missing_identifier_never_vetoes():
    graph = IdentityGraph()
    instances = {
        "i0": mk("i0", "man in green tracksuit"),
        "i1": mk("i1", "man in green tracksuit"),
    }
    decision = graph.propose(
        IdentityProposal("p1", "i0", "i1", "same", ("m0",), "same person"), instances
    )
    assert decision.status == "accepted"


def test_one_sided_identifier_never_vetoes():
    graph = IdentityGraph()
    instances = {
        "i0": mk("i0", "number 398"),
        "i1": mk("i1", "man in green tracksuit"),
    }
    decision = graph.propose(
        IdentityProposal("p1", "i0", "i1", "same", ("m0",), "same person"), instances
    )
    assert decision.status == "accepted"


def test_leading_zero_normalization():
    graph = IdentityGraph()
    instances = {
        "i0": mk("i0", "number 05"),
        "i1": mk("i1", "number 5"),
    }
    decision = graph.propose(
        IdentityProposal("p1", "i0", "i1", "same", ("m0",), "same person"), instances
    )
    assert decision.status == "accepted"
