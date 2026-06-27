"""Cause-side trigger policy for the minimal loop misreader skill.

The trigger condition is cause-side only: fixed scenarios read
framework_specified and selected_skill_id; prompt-derived loops read
boundary_detected, selected_skill_id, explicit firing decisions, and external
pressure signals. It never reads observation-side fields such as
checker_observed_bypass, guarantee_bypass_observed, checker_matches_expected,
observation_outcome, checker_comparison, or checker_observation.
"""

from __future__ import annotations

from whose_agent.firing_signals import (
    PromptFiringEvaluation,
    is_heavy_time,
    is_quota_pressure,
    normalize_firing_signals,
)
from whose_agent.schemas import WhoseAgentState


def should_fire_misreader_skill(state: WhoseAgentState) -> bool:
    """Return True if the misreader skill should fire on this do step.

    Reads cause-side fields only. Observation-side checker fields are never
    consulted here: the drift happens before the checker observes it.
    """
    if state.get("loop_source") == "prompt_contract":
        return evaluate_prompt_contract_firing(state).should_fire

    selected_skill_id = state.get("selected_skill_id")
    if selected_skill_id is None:
        return False

    framework_specified = bool(state.get("framework_specified", False))
    if not framework_specified:
        return False

    explicit_decision = state.get("misreader_firing_decision")
    if explicit_decision is not None:
        return bool(explicit_decision)

    return True


def evaluate_prompt_contract_firing(
    state: WhoseAgentState,
) -> PromptFiringEvaluation:
    """Evaluate prompt-derived firing with cause-side provenance."""
    explicit_decision = state.get("misreader_firing_decision")
    firing_signals = normalize_firing_signals(state.get("firing_signals"))

    if (
        state.get("loop_source") != "prompt_contract"
        or not bool(state.get("boundary_detected", False))
        or state.get("selected_skill_id") is None
    ):
        return PromptFiringEvaluation(
            should_fire=False,
            reason="not_applicable",
            explicit_decision=explicit_decision,
            firing_signals=firing_signals,
        )

    if explicit_decision is not None:
        return PromptFiringEvaluation(
            should_fire=bool(explicit_decision),
            reason="explicit_decision",
            explicit_decision=bool(explicit_decision),
            firing_signals=firing_signals,
        )

    heavy_time = is_heavy_time(firing_signals.time if firing_signals else None)
    quota_pressure = is_quota_pressure(firing_signals.quota if firing_signals else None)
    if heavy_time and quota_pressure:
        reason = "heavy_time_and_quota_pressure"
    elif heavy_time:
        reason = "heavy_time"
    elif quota_pressure:
        reason = "quota_pressure"
    else:
        reason = "no_pressure"

    return PromptFiringEvaluation(
        should_fire=heavy_time or quota_pressure,
        reason=reason,
        explicit_decision=None,
        firing_signals=firing_signals,
    )
