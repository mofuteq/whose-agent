"""Cause-side trigger policy for the minimal loop misreader skill.

The trigger condition is cause-side only: it reads framework_specified and
selected_skill_id from state. It never reads observation-side fields such as
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
    framework_specified = bool(state.get("framework_specified", False))
    selected_skill_id = state.get("selected_skill_id")
    return framework_specified and selected_skill_id is not None
