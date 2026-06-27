"""Cause-side trigger policy for the minimal loop misreader skill.

The trigger condition is cause-side only: fixed scenarios read
framework_specified and selected_skill_id; prompt-derived loops read
boundary_detected and selected_skill_id. It never reads observation-side fields such as
checker_observed_bypass, guarantee_bypass_observed, checker_matches_expected,
observation_outcome, checker_comparison, or checker_observation.
"""

from __future__ import annotations

from whose_agent.schemas import WhoseAgentState


def should_fire_misreader_skill(state: WhoseAgentState) -> bool:
    """Return True if the misreader skill should fire on this do step.

    Reads cause-side fields only. Observation-side checker fields are never
    consulted here: the drift happens before the checker observes it.
    """
    selected_skill_id = state.get("selected_skill_id")
    if selected_skill_id is None:
        return False

    if state.get("loop_source") == "prompt_contract":
        if not bool(state.get("boundary_detected", False)):
            return False
        explicit_decision = state.get("misreader_firing_decision")
        if explicit_decision is not None:
            return bool(explicit_decision)
        return False

    framework_specified = bool(state.get("framework_specified", False))
    if not framework_specified:
        return False

    explicit_decision = state.get("misreader_firing_decision")
    if explicit_decision is not None:
        return bool(explicit_decision)

    return True
