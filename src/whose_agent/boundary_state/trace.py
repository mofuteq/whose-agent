from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict

from whose_agent.boundary_state.state import BoundaryState
from whose_agent.boundary_state.transitions import (
    apply_bad_response,
    apply_reflection,
    finalize_boundary_state,
    initialize_boundary_state,
    update_boundary_state,
)
from whose_agent.models import Classification, Reflection, Scenario, TraceSubstituted


class BoundaryStateTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    state: BoundaryState


class BoundaryStateTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    transitions: list[BoundaryStateTransition]


def _mock_reflection(classification: Classification) -> Reflection:
    from whose_agent.trace_emitter import TRACE_TEMPLATES

    template = TRACE_TEMPLATES[classification.substituted]
    return Reflection(
        reflection_substituted=cast(TraceSubstituted, classification.substituted),
        why_it_breaks_delegation=list(template["why_it_breaks_delegation"]),
        better_behavior=list(template["better_behavior"]),
    )


def emit_state_trace(
    scenario: Scenario,
    classification: Classification,
    bad_response: str,
    *,
    mock: bool = False,
) -> BoundaryStateTrace:
    transitions: list[BoundaryStateTransition] = []

    state = initialize_boundary_state(scenario, classification)
    transitions.append(BoundaryStateTransition(step="initialize_boundary_state", state=state))

    state = apply_bad_response(state, bad_response)
    transitions.append(BoundaryStateTransition(step="apply_bad_response", state=state))

    if mock:
        reflection = _mock_reflection(classification)
    else:
        from whose_agent.reflection import reflect_failure

        reflection = reflect_failure(scenario, bad_response)

    state = apply_reflection(state, reflection)
    transitions.append(BoundaryStateTransition(step="apply_reflection", state=state))

    state = update_boundary_state(state)
    transitions.append(BoundaryStateTransition(step="update_boundary_state", state=state))

    state = finalize_boundary_state(state)
    transitions.append(BoundaryStateTransition(step="finalize_boundary_state", state=state))

    return BoundaryStateTrace(scenario_id=scenario.scenario_id, transitions=transitions)
