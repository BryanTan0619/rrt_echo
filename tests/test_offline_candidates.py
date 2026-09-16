from dataclasses import replace

from conftest import make_packet

from rrt_echo.memory import Memory
from rrt_echo.reconcile import rank_candidates


def test_future_candidate_and_temporal_cue_do_not_commit_identity():
    m = Memory()
    for name, time in [("early", 10), ("far", 100), ("later", 11)]:
        m.append(replace(make_packet(name, time), shot_id="same-shot"))
    refs = rank_candidates(m.instances["early:p"], list(m.instances), m, {})
    assert refs[0] == "later:p"
    assert m.identity.revision == 0
    assert len(m.identity.components(m.instances)) == 3


def test_seen_pair_filtered_before_candidate_limit():
    m = Memory()
    for name in ["a", "b", "c", "d", "e"]:
        m.append(make_packet(name))
    seen = {("a:p", "b:p"), ("a:p", "c:p"), ("a:p", "d:p")}
    assert rank_candidates(
        m.instances["a:p"], list(m.instances), m, {}, limit=1, seen_pairs=seen
    ) == ["e:p"]
