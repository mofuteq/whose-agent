from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from whose_agent.llm_result import LLMCallResult
from whose_agent.schemas import (
    Classification,
    Reflection,
    Scenario,
    ScenarioTraceTemplate,
    Trace,
    TraceSubstituted,
)


class TraceNotApplicableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceEmissionResult:
    trace: Trace
    reflection_call: LLMCallResult[Reflection] | None = None


def _scenario_trace_template(
    scenario: Scenario,
    classification: Classification,
) -> ScenarioTraceTemplate:
    if classification.classification != "in_scope" or classification.substituted == "none":
        raise TraceNotApplicableError("Trace emission is only defined for in-scope scenarios.")
    if scenario.failure_mode == "none":
        raise TraceNotApplicableError("Trace JSON must never use failure_mode: none.")
    if scenario.trace_template is None:
        raise TraceNotApplicableError("In-scope trace emission requires scenario.trace_template.")
    return scenario.trace_template


def _mock_reflection(scenario: Scenario, classification: Classification) -> Reflection:
    template = _scenario_trace_template(scenario, classification)
    return Reflection(
        reflection_substituted=cast(TraceSubstituted, classification.substituted),
        why_it_breaks_delegation=list(template.why_it_breaks_delegation),
        better_behavior=list(template.better_behavior),
    )


def emit_trace(
    scenario: Scenario,
    classification: Classification,
    bad_response: str,
    *,
    mock: bool = False,
) -> Trace:
    return emit_trace_with_usage(
        scenario,
        classification,
        bad_response,
        mock=mock,
    ).trace


def emit_trace_with_usage(
    scenario: Scenario,
    classification: Classification,
    bad_response: str,
    *,
    mock: bool = False,
) -> TraceEmissionResult:
    template = _scenario_trace_template(scenario, classification)

    if mock:
        reflection = _mock_reflection(scenario, classification)
        return TraceEmissionResult(
            trace=Trace(
                scenario_id=scenario.scenario_id,
                substituted=cast(TraceSubstituted, classification.substituted),
                failure_mode=scenario.failure_mode,
                principal_signal=scenario.principal_signal,
                bad_response=bad_response,
                divergence_point=template.divergence_point,
                why_it_breaks_delegation=reflection.why_it_breaks_delegation,
                better_behavior=reflection.better_behavior,
                reflection_substituted=reflection.reflection_substituted,
            )
        )

    from whose_agent.reflection import reflect_failure_with_usage

    reflection_call = reflect_failure_with_usage(scenario, bad_response)
    reflection = reflection_call.output
    return TraceEmissionResult(
        trace=Trace(
            scenario_id=scenario.scenario_id,
            substituted=cast(TraceSubstituted, classification.substituted),
            failure_mode=scenario.failure_mode,
            principal_signal=scenario.principal_signal,
            bad_response=bad_response,
            divergence_point=template.divergence_point,
            why_it_breaks_delegation=reflection.why_it_breaks_delegation,
            better_behavior=reflection.better_behavior,
            reflection_substituted=reflection.reflection_substituted,
        ),
        reflection_call=reflection_call,
    )
