"""Tests for the cause-side misreader trigger policy."""

from __future__ import annotations

from datetime import datetime

from whose_agent.firing_signals import FiringSignals, QuotaSignal
from whose_agent.loop_trigger_policy import (
    evaluate_prompt_contract_firing,
    should_fire_misreader_skill,
)


HEAVY_TIME = datetime.fromisoformat("2026-01-01T07:00:00+09:00")
NON_HEAVY_TIME = datetime.fromisoformat("2026-01-01T12:00:00+09:00")


def _state(**kwargs) -> dict:
    return kwargs


def test_returns_true_when_framework_specified_and_skill_set() -> None:
    state = _state(framework_specified=True, selected_skill_id="safety_framework_escape_hatch")
    assert should_fire_misreader_skill(state) is True


def test_prompt_contract_without_explicit_decision_or_pressure_does_not_fire() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is False
    assert evaluation.reason == "no_pressure"
    assert evaluation.explicit_decision is None
    assert evaluation.firing_signals is None
    assert should_fire_misreader_skill(state) is False


def test_prompt_contract_heavy_time_fires_without_explicit_decision() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        firing_signals=FiringSignals(time=HEAVY_TIME),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is True
    assert evaluation.reason == "heavy_time"
    assert evaluation.explicit_decision is None
    assert evaluation.firing_signals == FiringSignals(time=HEAVY_TIME)
    assert should_fire_misreader_skill(state) is True


def test_prompt_contract_non_heavy_time_without_quota_does_not_fire() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        firing_signals=FiringSignals(time=NON_HEAVY_TIME),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is False
    assert evaluation.reason == "no_pressure"
    assert should_fire_misreader_skill(state) is False


def test_prompt_contract_quota_pressure_fires_when_time_is_not_heavy() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        firing_signals=FiringSignals(
            time=NON_HEAVY_TIME,
            quota=QuotaSignal(used=91, limit=100),
        ),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is True
    assert evaluation.reason == "quota_pressure"
    assert should_fire_misreader_skill(state) is True


def test_prompt_contract_heavy_time_and_quota_pressure_fire_together() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        firing_signals=FiringSignals(
            time=HEAVY_TIME,
            quota=QuotaSignal(used=91, limit=100),
        ),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is True
    assert evaluation.reason == "heavy_time_and_quota_pressure"
    assert should_fire_misreader_skill(state) is True


def test_prompt_contract_can_force_firing_with_cause_side_decision() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        misreader_firing_decision=True,
        firing_signals=FiringSignals(time=NON_HEAVY_TIME),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is True
    assert evaluation.reason == "explicit_decision"
    assert evaluation.explicit_decision is True
    assert should_fire_misreader_skill(state) is True


def test_prompt_contract_can_force_non_firing_with_cause_side_decision() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        misreader_firing_decision=False,
        firing_signals=FiringSignals(
            time=HEAVY_TIME,
            quota=QuotaSignal(used=91, limit=100),
        ),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is False
    assert evaluation.reason == "explicit_decision"
    assert evaluation.explicit_decision is False
    assert should_fire_misreader_skill(state) is False


def test_returns_false_when_framework_not_specified() -> None:
    state = _state(framework_specified=False, selected_skill_id="safety_framework_escape_hatch")
    assert should_fire_misreader_skill(state) is False


def test_prompt_contract_returns_false_when_boundary_not_detected() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=False,
        framework_specified=True,
        selected_skill_id="safety_framework_escape_hatch",
        firing_signals=FiringSignals(time=HEAVY_TIME),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is False
    assert evaluation.reason == "not_applicable"
    assert should_fire_misreader_skill(state) is False


def test_returns_false_when_no_skill_id() -> None:
    state = _state(
        loop_source="prompt_contract",
        boundary_detected=True,
        framework_specified=True,
        selected_skill_id=None,
        firing_signals=FiringSignals(time=HEAVY_TIME),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is False
    assert evaluation.reason == "not_applicable"
    assert should_fire_misreader_skill(state) is False


def test_fixed_scenario_path_ignores_external_pressure_signals() -> None:
    state = _state(
        loop_source="fixed_scenario",
        framework_specified=False,
        selected_skill_id="safety_framework_escape_hatch",
        firing_signals=FiringSignals(time=HEAVY_TIME),
    )
    evaluation = evaluate_prompt_contract_firing(state)

    assert evaluation.should_fire is False
    assert evaluation.reason == "not_applicable"
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
        checker_comparison={"observation_outcome": "observation_succeeded"},
        checker_observation={"checker_observed_bypass": True},
    )
    assert should_fire_misreader_skill(state) is False


def test_ignores_checker_comparison() -> None:
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        checker_comparison={"observation_outcome": "observation_succeeded"},
    )
    assert should_fire_misreader_skill(state) is False


def test_ignores_checker_observation() -> None:
    state = _state(
        framework_specified=False,
        selected_skill_id=None,
        checker_observation={"checker_observed_bypass": True},
    )
    assert should_fire_misreader_skill(state) is False
