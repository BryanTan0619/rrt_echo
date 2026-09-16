import copy
import json

import pytest
from test_memory import link, result

from rrt_echo.rrt.memory import RRTMemory
from rrt_echo.storage import read_graph


def test_disk_rebuild_revoke_and_disposable_views(tmp_path):
    mem = RRTMemory("v", tmp_path / "ledger")
    original = result("a")
    mem.append(original)
    mem.append(result("b"))
    mem.propose(link("ab", "a", "b"))
    accepted = mem.materialize(tmp_path / "views")
    loaded = RRTMemory.open(tmp_path / "ledger")
    assert loaded.materialize() == accepted
    assert read_graph(tmp_path / "views/hypergraph.json") == accepted
    (tmp_path / "views/facts.jsonl").write_text("garbage")
    loaded.revoke("ab", "wrong person")
    revoked = loaded.materialize(tmp_path / "views")
    assert len(revoked["entities"]) == 2
    assert accepted["facts"][0]["roles"] == revoked["facts"][0]["roles"]
    assert original == result("a")
    assert RRTMemory.open(tmp_path / "ledger").materialize() == revoked
    assert all(e["status"] == "unresolved_local" for e in loaded.registry(revoked)["entities"])


def test_initial_bindings_are_atomic_and_replayed(tmp_path):
    mem = RRTMemory("v", tmp_path)
    a = result("a")
    b = result("b")
    a["observation"]["instances"] += b["observation"]["instances"]
    a["observation"]["media"] += b["observation"]["media"]
    a["link_proposals"] = [link("ab", "a", "b")]
    mem.append(a)
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 1
    assert RRTMemory.open(tmp_path).snapshot() == mem.snapshot()
    before = copy.deepcopy(mem.observations)
    bad = result("c")
    bad["link_proposals"] = [link("ab", "a", "b")]
    with pytest.raises(ValueError, match="duplicate_proposal"):
        mem.append(bad)
    assert mem.observations == before


def test_tampered_or_torn_log_rejected(tmp_path):
    mem = RRTMemory("v", tmp_path)
    mem.append(result("a"))
    path = tmp_path / "observations.jsonl"
    original = path.read_text()
    row = json.loads(original)
    row["payload"]["result"]["observation"]["facts"][0]["predicate"] = "run"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid_ledger_chain"):
        RRTMemory.open(tmp_path)
    path.write_text(original + "{")
    with pytest.raises(json.JSONDecodeError):
        RRTMemory.open(tmp_path)


def test_owner_revocation_survives_restart(tmp_path):
    mem = RRTMemory("v", tmp_path)
    mem.append(result("arm", "region"))
    mem.append(result("person"))
    mem.propose(link("owner", "arm", "person", "part_of"))
    assert mem.snapshot()["facts"][0]["owner_projections"]
    loaded = RRTMemory.open(tmp_path)
    loaded.revoke("owner", "insufficient connection")
    assert not RRTMemory.open(tmp_path).snapshot()["facts"][0]["owner_projections"]


def test_simple_pipeline_one_observation_and_ledger(monkeypatch, tmp_path):
    from dataclasses import asdict

    from rrt_echo.echo_perception.types import Clip, Media
    from rrt_echo.rrt import pipeline

    sample = result("a")
    clip = Clip(
        "a", "v", 0, 0, 1, (Media(**sample["observation"]["media"][0]),), 1, 1, 1, 0, True, {}
    )
    source = tmp_path / "clips.json"
    source.write_text(json.dumps([asdict(clip)]))
    calls = []

    class Observer:
        def __init__(self, **kwargs):
            assert kwargs["max_events"] == 4

        def observe(self, *args, **kwargs):
            calls.append("observe")
            return copy.deepcopy(sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("multistage path must be disabled")

    monkeypatch.setattr(pipeline, "OccurrenceVisionClient", Observer)
    monkeypatch.setattr(pipeline, "GlobalIndexClient", forbidden)
    monkeypatch.setattr(pipeline, "VerifiedPerceptionClient", forbidden)
    monkeypatch.setattr(pipeline, "review_links", forbidden)
    out = tmp_path / "run"
    pipeline.run(
        clips_path=source,
        out=out,
        model="fake",
        base_url="unused",
        identity=False,
        budget=500,
        request_seconds=1,
    )
    assert calls == ["observe"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["profile"] == "simple"
    assert not manifest["index"] and not manifest["binding_review"]
    assert manifest["max_local_repairs"] == 0
    assert read_graph(out / "hypergraph.json") == RRTMemory.open(out / "ledger").materialize()


def test_simple_does_not_silently_enable_omni(tmp_path):
    from rrt_echo.rrt.pipeline import run

    with pytest.raises(ValueError, match="require_legacy_profile"):
        run(video="unused", out=tmp_path / "run", model="x", base_url="unused", av_checkpoint="x")


def test_bad_binding_row_does_not_erase_valid_independent_row():
    from rrt_echo.echo_perception.correspondence import validate_partial_correspondences

    instances = {x + ":i": result(x)["observation"]["instances"][0] for x in "abc"}
    pairs = [
        {"pair_id": "ab", "left": "a:i", "right": "b:i", "relation": "same_identity"},
        {"pair_id": "ac", "left": "a:i", "right": "c:i", "relation": "same_identity"},
    ]
    raw = {
        "correspondences": [
            {
                "pair_id": "ab",
                "verdict": "same",
                "evidence_ids": ["a:f", "b:f"],
                "basis": "visible",
                "gap_reason": "none",
            },
            {
                "pair_id": "ac",
                "verdict": "belongs",
                "evidence_ids": ["a:f", "c:f"],
                "basis": "wrong relation",
                "gap_reason": "none",
            },
        ]
    }
    rows, rejected = validate_partial_correspondences(raw, pairs, instances, ["a:f", "b:f", "c:f"])
    assert rows[0]["verdict"] == "same"
    assert rows[1]["verdict"] == "unresolved"
    assert rejected[0]["pair_id"] == "ac"


def test_text_revision_is_versioned_and_cannot_change_owner(tmp_path):
    r = result("text", "object")
    f = r["observation"]["facts"][0]
    f.update(
        kind="text",
        predicate="inscription",
        value="12",
        roles={"owner": "text:i"},
        evidence_by_slot={"predicate": ["text:f"], "owner": ["text:f"]},
    )
    mem = RRTMemory("v", tmp_path)
    mem.append(r)
    original_line = (tmp_path / "observations.jsonl").read_text()
    old = mem.materialize()
    review = {
        "verdict": "readable",
        "model": "test",
        "basis": "higher detail crop",
        "crops": [
            {
                "source_media_id": "text:f",
                "source_sha256": "a" * 64,
                "source_box": [0, 0, 1, 1],
                "sha256": "b" * 64,
            }
        ],
    }
    mem.revise_text("text:event", "17", review)
    graph = mem.materialize()
    assert graph["facts"][0]["value"] == "17" and graph["facts"][0]["version"] == 2
    assert old["facts"][0]["value"] == "12"
    assert graph["facts"][0]["roles"] == old["facts"][0]["roles"]
    assert (tmp_path / "observations.jsonl").read_text().startswith(original_line)
    assert RRTMemory.open(tmp_path).materialize() == graph
    bad = copy.deepcopy(review)
    bad["crops"][0]["source_box"] = [0, 0, 0.5, 0.5]
    with pytest.raises(ValueError, match="not_owner_region"):
        mem.revise_text("text:event", "99", bad)


def test_copied_person_box_does_not_certify_identity_or_directed_roles():
    r = result("scene")
    other = copy.deepcopy(r["observation"]["instances"][0])
    other["instance_id"] = "scene:other"
    r["observation"]["instances"].append(other)
    f = r["observation"]["facts"][0]
    f.update(
        predicate="hold",
        roles={"holder": "scene:i", "held": "scene:other"},
        evidence_by_slot={"predicate": ["scene:f"], "holder": ["scene:f"], "held": ["scene:f"]},
    )
    r["link_proposals"] = [
        {
            "proposal_id": "bad_merge",
            "source": "scene:i",
            "target": "scene:other",
            "relation": "same_identity",
            "verdict": "supported",
            "evidence_ids": ["scene:f"],
            "basis": "model assertion using the copied box",
        }
    ]
    original = copy.deepcopy(r)
    mem = RRTMemory()
    mem.append(r)
    graph = mem.snapshot()
    assert graph["binding_links"][0]["status"] == "unresolved"
    assert len(graph["facts"]) == 1 and graph["facts"][0]["predicate"] == "hold"
    assert set(graph["facts"][0]["unresolved_slots"]) == {"holder", "held"}
    assert len(graph["entities"]) == 2
    assert mem.observations[0] == original
    assert graph["endpoint_grounding_gaps"]


def test_carry_roles_cannot_collapse_directly_or_through_third_instance():
    a = result("a")
    b = result("b")
    b["observation"]["instances"][0]["regions"][0]["box"] = [0.6, 0.1, 0.9, 0.4]
    a["observation"]["instances"] += b["observation"]["instances"]
    a["observation"]["media"] += b["observation"]["media"]
    f = a["observation"]["facts"][0]
    f.update(
        predicate="carry",
        roles={"carrier": "a:i", "carried": "b:i"},
        evidence_by_slot={"predicate": ["a:f", "b:f"], "carrier": ["a:f"], "carried": ["b:f"]},
        joint_evidence=["a:f", "b:f"],
        observed_media_ids=["a:f", "b:f"],
    )
    mem = RRTMemory()
    mem.append(a)
    mem.append(result("c"))
    mem.propose(link("ab", "a", "b"))
    direct = mem.snapshot()
    assert direct["binding_links"][0]["status"] == "unresolved"
    mem.propose(link("ac", "a", "c"))
    mem.propose(link("bc", "b", "c"))
    graph = mem.snapshot()
    assert len(graph["entities"]) == 3
    event = graph["facts"][0]
    assert (
        event["resolved_roles"]["carrier"]["entity_id"]
        != event["resolved_roles"]["carried"]["entity_id"]
    )
    assert event["roles"] == f["roles"] and not event["unresolved_slots"]
    assert graph["separation_constraints"][0]["source_fact_id"] == f["fact_id"]
