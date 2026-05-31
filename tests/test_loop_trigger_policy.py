"""Tests for the cause-side misreader trigger policy."""

from __future__ import annotations

from whose_agent.loop_trigger_policy import should_fire_misreader_skill


def _state(**kwargs) -> dict:
    return kwargs


def test_returns_true_when_framework_specified_and_skill_set() -> None:
    state = _state(framework_specified=True, selected_skill_id="safety_framework_escape_hatch")
    assert should_fire_misreader_skill(state) is True


def test_returns_false_when_framework_not_specified() -> None:
    state = _state(framework_specified=False, selected_skill_id="safety_framework_escape_hatch")
    assert should_fire_misreader_skill(state) is False


def test_returns_false_when_no_skill_id() -> None:
    state = _state(framework_specified=True, selected_skill_id=None)
    assert should_fire_misreader_skill(state) is False


def test_returns_false_when_both_absent() -> None:
    assert should_fire_misreader_skill({}) is False


def test_returns_false_when_both_false_and_none() -> None:
    state = _state(framework_specified=False, selected_skill_id=None)
    assert should_fire_misreader_skill(state) is False


def test_ignores_checker_observed_bypass() -> None:
    # Observation-side field must not rescue a failing cause-side condition.
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        checker_observed_bypass=True,
    )
    assert should_fire_misreader_skill(state) is False


def test_ignores_guarantee_bypass_observed() -> None:
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        guarantee_bypass_observed=True,
    )
    assert should_fire_misreader_skill(state) is False


def test_ignores_checker_matches_expected() -> None:
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        checker_matches_expected=True,
    )
    assert should_fire_misreader_skill(state) is False


def test_ignores_observation_outcome() -> None:
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        observation_outcome="observation_succeeded",
    )
    assert should_fire_misreader_skill(state) is False


def test_all_observation_side_fields_present_but_cause_side_fails() -> None:
    # Even with every observation-side field set to a "true" value, the policy
    # returns False when the cause-side preconditions are not met.
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        checker_observed_bypass=True,
        guarantee_bypass_observed=True,
        checker_matches_expected=True,
        observation_outcome="observation_succeeded",
    )
    assert should_fire_misreader_skill(state) is False
