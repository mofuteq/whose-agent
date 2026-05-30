from __future__ import annotations

from dataclasses import dataclass
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
from whose_agent.llm_result import LLMCallResult
from whose_agent.models import Classification, Reflection, Scenario, TraceSubstituted


class BoundaryStateTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    state: BoundaryState


class BoundaryStateTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    transitions: list[BoundaryStateTransition]


@dataclass(frozen=True)
class BoundaryStateTraceEmissionResult:
    state_trace: BoundaryStateTrace
    reflection_call: LLMCallResult[Reflection] | None = None


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
    return emit_state_trace_with_usage(
        scenario,
        classification,
        bad_response,
        mock=mock,
    ).state_trace


def emit_state_trace_with_usage(
    scenario: Scenario,
    classification: Classification,
    bad_response: str,
    *,
    mock: bool = False,
) -> BoundaryStateTraceEmissionResult:
    transitions: list[BoundaryStateTransition] = []

    state = initialize_boundary_state(scenario, classification)
    transitions.append(BoundaryStateTransition(step="initialize_boundary_state", state=state))

    state = apply_bad_response(state, bad_response)
    transitions.append(BoundaryStateTransition(step="apply_bad_response", state=state))

    if mock:
        reflection = _mock_reflection(classification)
        reflection_call = None
    else:
        from whose_agent.reflection import reflect_failure_with_usage

        reflection_call = reflect_failure_with_usage(scenario, bad_response)
        reflection = reflection_call.output

    state = apply_reflection(state, reflection)
    transitions.append(BoundaryStateTransition(step="apply_reflection", state=state))

    state = update_boundary_state(state)
    transitions.append(BoundaryStateTransition(step="update_boundary_state", state=state))

    state = finalize_boundary_state(state)
    transitions.append(BoundaryStateTransition(step="finalize_boundary_state", state=state))

    return BoundaryStateTraceEmissionResult(
        state_trace=BoundaryStateTrace(scenario_id=scenario.scenario_id, transitions=transitions),
        reflection_call=reflection_call,
    )
