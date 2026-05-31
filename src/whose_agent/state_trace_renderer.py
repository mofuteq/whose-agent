from __future__ import annotations

from typing import cast

from whose_agent.schemas import (
    BoundaryNextAction,
    BoundaryState,
    BoundaryStateTrace,
    BoundaryStateTransition,
    TraceFailureMode,
    TraceSubstituted,
    WhoseAgentState,
)


STATE_TRACE_STEPS = [
    "initialized",
    "bad_response_applied",
    "reflection_applied",
    "boundary_updated",
    "finalized",
]


def render_boundary_state_trace(state: WhoseAgentState) -> BoundaryStateTrace:
    scenario = state.get("scenario")
    if scenario is None:
        raise ValueError("WhoseAgentState requires scenario.")
    if scenario.expected_substituted == "none" or scenario.failure_mode == "none":
        raise ValueError("BoundaryStateTrace is only defined for in-scope scenarios.")

    bad_response = state.get("bad_response")
    if bad_response is None:
        raise ValueError("WhoseAgentState requires bad_response.")

    trace = state.get("trace")
    if trace is None:
        raise ValueError("WhoseAgentState requires trace.")

    base = BoundaryState(
        scenario_id=scenario.scenario_id,
        principal_prompt=scenario.principal_prompt,
        principal_signal=scenario.principal_signal,
        expected_substituted=cast(TraceSubstituted, scenario.expected_substituted),
        failure_mode=cast(TraceFailureMode, scenario.failure_mode),
    )
    with_bad_response = base.model_copy(update={"bad_response": bad_response})
    with_reflection = with_bad_response.model_copy(
        update={
            "reflection_substituted": trace.reflection_substituted,
            "why_it_breaks_delegation": list(trace.why_it_breaks_delegation),
            "better_behavior": list(trace.better_behavior),
        }
    )

    reflection_matches_expected = (
        trace.reflection_substituted == scenario.expected_substituted
    )
    boundary_flags: list[TraceFailureMode] = (
        [cast(TraceFailureMode, scenario.failure_mode)]
        if reflection_matches_expected
        else []
    )
    with_boundary = with_reflection.model_copy(
        update={
            "reflection_matches_expected": reflection_matches_expected,
            "boundary_flags": boundary_flags,
        }
    )
    boundary_next_action: BoundaryNextAction = (
        "trace_ready" if reflection_matches_expected else "review_reflection"
    )
    finalized = with_boundary.model_copy(
        update={"next_action": boundary_next_action}
    )

    snapshots = [
        base,
        with_bad_response,
        with_reflection,
        with_boundary,
        finalized,
    ]
    transitions = [
        BoundaryStateTransition(step=step, state=snapshot)
        for step, snapshot in zip(STATE_TRACE_STEPS, snapshots, strict=True)
    ]
    return BoundaryStateTrace(
        scenario_id=scenario.scenario_id,
        transitions=transitions,
    )
