from rrt_echo.rrt.link_review import endpoint_verdict


def answer(a, b, verdict="supported"):
    return dict(
        source_category=a,
        target_category=b,
        verdict=verdict,
        source_localized=True,
        target_localized=True,
    )


def test_scene_copresence_cannot_merge_adult_and_child():
    assert (
        endpoint_verdict("same_identity", "person", "person", answer("adult", "child"))
        == "contradicted"
    )


def test_bad_person_anchor_cannot_create_different_identity_constraint():
    assert (
        endpoint_verdict(
            "same_identity", "person", "person", answer("body_part", "adult", "contradicted")
        )
        == "unresolved"
    )


def test_anatomical_difference_does_not_reject_ownership():
    assert (
        endpoint_verdict("part_of", "region", "person", answer("body_part", "child")) == "supported"
    )


def test_background_is_not_body_part():
    assert (
        endpoint_verdict("part_of", "region", "person", answer("background", "child"))
        == "contradicted"
    )
