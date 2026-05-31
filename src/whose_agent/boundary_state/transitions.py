from __future__ import annotations

from typing import cast

from whose_agent.schemas import (
    BoundaryNextAction,
    BoundaryState,
    Classification,
    Reflection,
    Scenario,
    TraceFailureMode,
    TraceSubstituted,
)


class OutOfScopeBoundaryError(ValueError):
    pass


def initialize_boundary_state(
    scenario: Scenario,
    classification: Classification,
) -> BoundaryState:
    if classification.classification != "in_scope" or classification.substituted == "none":
        raise OutOfScopeBoundaryError(
            "BoundaryState is only defined for in-scope classifications."
        )
    return BoundaryState(
        scenario_id=scenario.scenario_id,
        principal_prompt=scenario.principal_prompt,
        principal_signal=scenario.principal_signal,
        expected_substituted=cast(TraceSubstituted, classification.substituted),
        failure_mode=cast(TraceFailureMode, scenario.failure_mode),
    )


def apply_bad_response(
    state: BoundaryState,
    bad_response: str,
) -> BoundaryState:
    return state.model_copy(update={"bad_response": bad_response})


def apply_reflection(
    state: BoundaryState,
    reflection: Reflection,
) -> BoundaryState:
    return state.model_copy(update={
        "reflection_substituted": reflection.reflection_substituted,
        "why_it_breaks_delegation": list(reflection.why_it_breaks_delegation),
        "better_behavior": list(reflection.better_behavior),
    })


def update_boundary_state(
    state: BoundaryState,
) -> BoundaryState:
    matches = state.reflection_substituted == state.expected_substituted
    flags: list[TraceFailureMode] = [state.failure_mode] if matches else []
    return state.model_copy(update={
        "reflection_matches_expected": matches,
        "boundary_flags": flags,
    })


def finalize_boundary_state(
    state: BoundaryState,
) -> BoundaryState:
    if state.reflection_matches_expected is None:
        raise ValueError(
            "update_boundary_state must be called before finalize_boundary_state."
        )
    next_action: BoundaryNextAction = (
        "trace_ready" if state.reflection_matches_expected else "review_reflection"
    )
    return state.model_copy(update={"next_action": next_action})
