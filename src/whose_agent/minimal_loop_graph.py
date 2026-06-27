"""Minimal LangGraph loop path that demonstrates intermittent boundary drift.

Schema definitions live in whose_agent.schemas; this module owns a small
plan -> do -> check loop only. It is separate from the fixed scenario graph in
whose_agent.state_graph and does not replace it.

LangGraph state (WhoseAgentState) is the runtime source of truth. ControlState
is not wired in as a nested runtime object; the loop fields live directly on
WhoseAgentState.

Causal rule: misreader_skill_fired is the cause-side event. It is set in the do
step from cause-side conditions only. Fixed scenarios use
framework_specified + selected_skill_id. Prompt-derived loops use
boundary_detected + selected_skill_id plus an explicit firing decision or
deterministic external pressure signals.
checker_observed_bypass and guarantee_bypass_observed are observation-side events
set in the check step. Checker observation is never a precondition for misreader
firing.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from whose_agent.bad_response import generate_bad_response_with_usage
from whose_agent.checker import (
    check_with_usage,
    compare_checker_observation,
    load_skill_perspective,
)
from whose_agent.classifier import classify_scenario
from whose_agent.firing_signals import PromptFiringEvaluation
from whose_agent.loop_trigger_policy import (
    evaluate_prompt_contract_firing,
    should_fire_misreader_skill,
)
from whose_agent.prompt_response import generate_contract_preserving_response_with_usage
from whose_agent.schemas import (
    Classification,
    Scenario,
    StepKind,
    StepTrace,
    WhoseAgentState,
)


CHECKER_ID = "skill-perspective-checker"
PROMPT_DERIVED_DRIFT_ARTIFACT_KIND = "prompt_derived_poor_e2e"
PROMPT_DRIFT_AXIS_MAX_LENGTH = 40
PROMPT_DRIFT_BOUNDARY_MAX_LENGTH = 160
PROMPT_DRIFT_EVIDENCE_WITH_BOUNDARY = (
    "Generated output that crossed the prompt-derived {axis} boundary: {boundary}."
)
PROMPT_DRIFT_EVIDENCE_FALLBACK = (
    "Generated output that crossed the prompt-derived delegated boundary."
)

def derive_framework_specified_for_scenario(scenario: Scenario) -> bool:
    """Decide deterministically whether the principal specified a framework.

    Fixed scenarios can use deterministic metadata: the safety-framework skill
    is the framework-surface guarantee path. Other instruction-axis skills are
    not treated as framework guarantees.

    Arbitrary prompt loop support gets framework detection from a
    PromptContract before it reaches this helper. This helper stays fixed
    scenario-only.
    """
    return scenario.selected_skill_id == "safety_framework_escape_hatch"


def initial_loop_state_from_scenario(
    scenario: Scenario,
    *,
    max_iterations: int = 3,
) -> WhoseAgentState:
    """Initialize a WhoseAgentState for the minimal loop.

    Uses the same WhoseAgentState shape as the fixed scenario graph; there is no
    separate LoopState wrapper.
    """
    return {
        "principal": "user",
        "agent": "assistant",
        "principal_instruction": scenario.principal_prompt,
        "principal_signal": scenario.principal_signal,
        "scenario": scenario,
        "classification": None,
        "bad_response": None,
        "generation_used_skill": False,
        "generation_skill_id": None,
        "trace": None,
        "state_trace": None,
        "checker_observation": None,
        "checker_comparison": None,
        "step_kind": "plan",
        "step_index": 0,
        "next_action": "continue",
        "handoff_ready": False,
        "completed": False,
        "selected_skill_id": scenario.selected_skill_id,
        "selected_skill_perspective": None,
        "misreader_firing_decision": None,
        "firing_signals": None,
        "firing_reason": None,
        "skill_triggered": False,
        "misreader_skill_fired": False,
        "trigger_evidence": [],
        "checker_ran": False,
        "checker_observed_bypass": False,
        "checker_id": None,
        "checker_confidence": None,
        "checker_matches_expected": None,
        "observation_outcome": None,
        "guarantee_bypass_observed": False,
        "guarantee_bypass_evidence": [],
        "substituted": scenario.expected_substituted,
        "failure_mode": scenario.failure_mode,
        "divergence_point": None,
        "boundary_flags": [],
        "boundary_detected": False,
        "substitution_axis": None,
        "delegated_boundary": None,
        "framework_specified": False,
        "loop_iteration": 0,
        "loop_phase": "plan",
        "max_iterations": max_iterations,
        "loop_completed": False,
        "loop_stop_reason": None,
        "loop_source": "fixed_scenario",
        "prompt_contract_status": None,
        "prompt_contract_boundary_detected": None,
        "prompt_contract_substitution_axis": None,
        "prompt_contract_delegated_boundary": None,
        "prompt_contract_candidate_framework": None,
        "prompt_contract_delegated_guarantee": None,
        "prompt_contract_artifact": None,
        "prompt_loop_generated_artifact": None,
        "prompt_loop_generated_step_index": None,
        "step_traces": [],
        "errors": [],
    }


def compile_minimal_loop_graph(*, mock: bool = False) -> Any:
    return build_minimal_loop_graph(mock=mock).compile()


def build_minimal_loop_graph(*, mock: bool = False) -> StateGraph:
    graph = StateGraph(WhoseAgentState)

    def plan(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        classification = classify_scenario(scenario)
        framework_specified = bool(
            state.get("framework_specified", False)
        ) or derive_framework_specified_for_scenario(scenario)
        boundary_detected = bool(state.get("boundary_detected", False)) or (
            state.get("loop_source") == "fixed_scenario" and framework_specified
        )
        # Re-planning resets the cause-side misreader flag at the start of each
        # iteration so the loop can demonstrate intermittent drift.
        return {
            "classification": classification,
            "substituted": classification.substituted,
            "framework_specified": framework_specified,
            "boundary_detected": boundary_detected,
            "misreader_skill_fired": False,
            "skill_triggered": False,
            "loop_phase": "plan",
            **_step_update(
                state,
                "plan",
                misreader_skill_fired=False,
                selected_skill_id=state.get("selected_skill_id"),
            ),
        }

    def do(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        classification = _classification(state)
        selected_skill_id = state.get("selected_skill_id")

        # Cause-side firing condition only. Checker observation is never read here.
        firing_evaluation = _evaluate_do_step_firing(state)
        if firing_evaluation is not None:
            state = {
                **state,
                "firing_signals": firing_evaluation.firing_signals,
                "firing_reason": firing_evaluation.reason,
            }
            should_fire = firing_evaluation.should_fire
            prompt_trigger_evidence = _prompt_firing_trigger_evidence(
                state,
                firing_evaluation,
            )
        else:
            should_fire = should_fire_misreader_skill(state)
            prompt_trigger_evidence = []

        if classification.classification != "in_scope":
            return {
                "firing_reason": (
                    firing_evaluation.reason
                    if firing_evaluation is not None
                    else state.get("firing_reason")
                ),
                "firing_signals": (
                    firing_evaluation.firing_signals
                    if firing_evaluation is not None
                    else state.get("firing_signals")
                ),
                "skill_triggered": False,
                "misreader_skill_fired": False,
                "bad_response": None,
                "generation_used_skill": False,
                "generation_skill_id": None,
                "loop_phase": "do",
                **_step_update(
                    state,
                    "do",
                    misreader_skill_fired=False,
                    selected_skill_id=selected_skill_id,
                    trigger_evidence=prompt_trigger_evidence,
                    substituted=classification.substituted,
                ),
            }

        if _is_unsupported_prompt_contract(state):
            return {
                "firing_reason": (
                    firing_evaluation.reason
                    if firing_evaluation is not None
                    else state.get("firing_reason")
                ),
                "firing_signals": (
                    firing_evaluation.firing_signals
                    if firing_evaluation is not None
                    else state.get("firing_signals")
                ),
                "skill_triggered": False,
                "misreader_skill_fired": False,
                "bad_response": None,
                "generation_used_skill": False,
                "generation_skill_id": None,
                "loop_phase": "do",
                "substituted": "none",
                "failure_mode": "none",
                **_step_update(
                    state,
                    "do",
                    misreader_skill_fired=False,
                    selected_skill_id=selected_skill_id,
                    trigger_evidence=prompt_trigger_evidence,
                    substituted="none",
                ),
            }

        if should_fire:
            selected_skill_perspective = state.get("selected_skill_perspective")
            if selected_skill_perspective is None:
                selected_skill_perspective = load_skill_perspective(
                    cast(str, selected_skill_id)
                )
            trigger_evidence = [
                f"cause-side boundary with selected skill {selected_skill_id!r} on the do "
                "step; the misreader skill fires and the artifact crosses the delegated boundary."
            ]
            trigger_evidence = prompt_trigger_evidence + trigger_evidence
            bad_response = generate_bad_response_with_usage(
                scenario,
                classification,
                selected_skill_id=selected_skill_id,
                selected_skill_perspective=selected_skill_perspective,
                misreader_skill_fired=True,
                mock=mock,
            ).output
            generation_used_skill = selected_skill_perspective is not None
            drift_evidence, drift_artifact_kind = _prompt_derived_drift_evidence(state)
            return {
                "selected_skill_perspective": selected_skill_perspective,
                "firing_reason": (
                    firing_evaluation.reason
                    if firing_evaluation is not None
                    else state.get("firing_reason")
                ),
                "firing_signals": (
                    firing_evaluation.firing_signals
                    if firing_evaluation is not None
                    else state.get("firing_signals")
                ),
                "skill_triggered": True,
                "misreader_skill_fired": True,
                "trigger_evidence": trigger_evidence,
                "bad_response": bad_response,
                "generation_used_skill": generation_used_skill,
                "generation_skill_id": (
                    selected_skill_id if generation_used_skill else None
                ),
                "loop_phase": "do",
                **_step_update(
                    state,
                    "do",
                    misreader_skill_fired=True,
                    selected_skill_id=selected_skill_id,
                    generation_used_skill=generation_used_skill,
                    generation_skill_id=(
                        selected_skill_id if generation_used_skill else None
                    ),
                    trigger_evidence=trigger_evidence,
                    drift_evidence=drift_evidence,
                    drift_artifact_kind=drift_artifact_kind,
                    substituted=classification.substituted,
                ),
            }

        # Misreader does not fire: generation must not use skill context.
        prompt_contract_preserved = _is_supported_prompt_contract(state)
        if prompt_contract_preserved:
            bad_response = generate_contract_preserving_response_with_usage(
                scenario.principal_prompt,
                substitution_axis=state.get("prompt_contract_substitution_axis"),
                delegated_boundary=state.get("prompt_contract_delegated_boundary"),
                candidate_framework=state.get("prompt_contract_candidate_framework"),
                delegated_guarantee=state.get("prompt_contract_delegated_guarantee"),
                mock=mock,
            ).output
            substituted = "none"
        else:
            bad_response = generate_bad_response_with_usage(
                scenario,
                classification,
                mock=mock,
            ).output
            substituted = classification.substituted
        return {
            "firing_reason": (
                firing_evaluation.reason
                if firing_evaluation is not None
                else state.get("firing_reason")
            ),
            "firing_signals": (
                firing_evaluation.firing_signals
                if firing_evaluation is not None
                else state.get("firing_signals")
            ),
            "skill_triggered": False,
            "misreader_skill_fired": False,
            "bad_response": bad_response,
            "generation_used_skill": False,
            "generation_skill_id": None,
            "loop_phase": "do",
            "substituted": substituted,
            "failure_mode": "none" if prompt_contract_preserved else scenario.failure_mode,
            **_step_update(
                state,
                "do",
                misreader_skill_fired=False,
                selected_skill_id=selected_skill_id,
                trigger_evidence=prompt_trigger_evidence,
                substituted=substituted,
            ),
        }

    def check(state: WhoseAgentState) -> WhoseAgentState:
        scenario = _scenario(state)
        selected_skill_id = state.get("selected_skill_id")
        misreader_skill_fired = bool(state.get("misreader_skill_fired", False))
        bad_response = state.get("bad_response")

        checker_observation = None
        checker_ran = False
        checker_observed_bypass = False
        checker_confidence: str | None = None
        checker_id: str | None = None
        guarantee_bypass_evidence: list[str] = []

        if selected_skill_id is not None and bad_response is not None:
            checker_observation = check_with_usage(
                scenario,
                bad_response,
                mock=mock,
            ).observation
            checker_ran = True
            checker_observed_bypass = checker_observation.checker_observed_bypass
            checker_confidence = checker_observation.confidence
            checker_id = CHECKER_ID
            if checker_observed_bypass:
                guarantee_bypass_evidence = list(checker_observation.evidence)

        comparison = compare_checker_observation(
            scenario,
            checker_observation,
            misreader_skill_fired=misreader_skill_fired,
            comparison_mode=(
                "prompt_observability"
                if state.get("loop_source") == "prompt_contract"
                else "fixed_benchmark"
            ),
        )

        loop_iteration = int(state.get("loop_iteration", 0)) + 1
        max_iterations = int(state.get("max_iterations", 3))
        loop_completed = loop_iteration >= max_iterations
        loop_stop_reason = "max_iterations_reached" if loop_completed else None

        return {
            "checker_observation": checker_observation,
            "checker_ran": checker_ran,
            "checker_observed_bypass": checker_observed_bypass,
            "checker_id": checker_id,
            "checker_confidence": checker_confidence,
            "checker_comparison": comparison,
            "checker_matches_expected": comparison.matches_expected,
            "observation_outcome": comparison.observation_outcome,
            "guarantee_bypass_observed": checker_observed_bypass,
            "guarantee_bypass_evidence": guarantee_bypass_evidence,
            "loop_iteration": loop_iteration,
            "loop_phase": "check",
            "loop_completed": loop_completed,
            "loop_stop_reason": loop_stop_reason,
            "completed": loop_completed,
            "next_action": "stop" if loop_completed else "continue",
            **_step_update(
                state,
                "check",
                misreader_skill_fired=misreader_skill_fired,
                selected_skill_id=selected_skill_id,
                checker_ran=checker_ran,
                checker_observed_bypass=checker_observed_bypass,
                substituted=(
                    checker_observation.substituted
                    if checker_observation is not None
                    else state.get("substituted")
                ),
            ),
        }

    graph.add_node("plan", plan)
    graph.add_node("do", do)
    graph.add_node("check", check)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "do")
    graph.add_edge("do", "check")
    graph.add_conditional_edges(
        "check",
        _route_after_check,
        {"plan": "plan", "end": END},
    )
    return graph


def _route_after_check(state: WhoseAgentState) -> str:
    loop_iteration = int(state.get("loop_iteration", 0))
    max_iterations = int(state.get("max_iterations", 3))
    if bool(state.get("loop_completed", False)) or loop_iteration >= max_iterations:
        return "end"
    return "plan"


def _evaluate_do_step_firing(
    state: WhoseAgentState,
) -> PromptFiringEvaluation | None:
    if state.get("loop_source") != "prompt_contract":
        return None
    return evaluate_prompt_contract_firing(state)


def _prompt_firing_trigger_evidence(
    state: WhoseAgentState,
    evaluation: PromptFiringEvaluation,
) -> list[str]:
    evidence = [f"Prompt-contract firing reason: {evaluation.reason}."]
    if evaluation.reason == "not_applicable":
        missing: list[str] = []
        if not bool(state.get("boundary_detected", False)):
            missing.append("boundary_detected is false")
        if state.get("selected_skill_id") is None:
            missing.append("selected_skill_id is missing")
        if not missing:
            missing.append("loop_source is not prompt_contract")
        evidence.append(
            "Prompt-contract firing policy did not apply: "
            + " and ".join(missing)
            + "."
        )
        return evidence

    if evaluation.explicit_decision is not None:
        evidence.append(
            "Explicit misreader_firing_decision="
            f"{evaluation.explicit_decision} overrides external pressure."
        )

    firing_signals = evaluation.firing_signals
    if firing_signals is not None and firing_signals.time is not None:
        evidence.append(
            f"Evaluated firing time: {firing_signals.time.isoformat()}."
        )

    if firing_signals is None or firing_signals.quota is None:
        evidence.append("Quota pressure: absent.")
    else:
        quota = firing_signals.quota
        evidence.append(
            "Quota: "
            f"{_format_signal_number(quota.used)} / "
            f"{_format_signal_number(quota.limit)} "
            f"(ratio={quota.ratio:.2f})."
        )

    return evidence


def _format_signal_number(value: float) -> str:
    return f"{value:g}"


def _step_update(
    state: WhoseAgentState,
    step_kind: StepKind,
    *,
    misreader_skill_fired: bool = False,
    selected_skill_id: str | None = None,
    trigger_evidence: list[str] | None = None,
    checker_ran: bool = False,
    checker_observed_bypass: bool = False,
    generation_used_skill: bool = False,
    generation_skill_id: str | None = None,
    drift_evidence: str | None = None,
    drift_artifact_kind: str | None = None,
    substituted: str | None = None,
) -> WhoseAgentState:
    step_index = int(state.get("step_index", 0))
    trace = StepTrace(
        step_index=step_index,
        step_kind=step_kind,
        principal=state.get("principal", "user"),
        agent=state.get("agent", "assistant"),
        misreader_skill_fired=misreader_skill_fired,
        selected_skill_id=selected_skill_id,
        generation_used_skill=generation_used_skill,
        generation_skill_id=generation_skill_id,
        checker_ran=checker_ran,
        checker_observed_bypass=checker_observed_bypass,
        trigger_evidence=list(trigger_evidence or []),
        drift_evidence=drift_evidence,
        drift_artifact_kind=drift_artifact_kind,
        substituted=_step_substituted(substituted),
        boundary_flags=[],
        divergence_point=None,
    )
    return {
        "step_kind": step_kind,
        "step_index": step_index + 1,
        "step_traces": [trace],
    }


def _scenario(state: WhoseAgentState) -> Scenario:
    scenario = state.get("scenario")
    if scenario is None:
        raise ValueError("WhoseAgentState requires scenario.")
    return scenario


def _classification(state: WhoseAgentState) -> Classification:
    classification = state.get("classification")
    if classification is None:
        raise ValueError("WhoseAgentState requires classification.")
    return classification


def _prompt_derived_drift_evidence(
    state: WhoseAgentState,
) -> tuple[str | None, str | None]:
    if state.get("loop_source") != "prompt_contract":
        return None, None
    if state.get("prompt_contract_status") != "contract_detected":
        return None, None
    axis = _concise_text(
        state.get("prompt_contract_substitution_axis"),
        max_length=PROMPT_DRIFT_AXIS_MAX_LENGTH,
        fallback="delegated",
    )
    boundary = _concise_text(
        state.get("prompt_contract_delegated_boundary"),
        max_length=PROMPT_DRIFT_BOUNDARY_MAX_LENGTH,
        fallback=None,
    )
    if boundary is not None:
        evidence = PROMPT_DRIFT_EVIDENCE_WITH_BOUNDARY.format(
            axis=axis,
            boundary=boundary,
        )
    else:
        evidence = PROMPT_DRIFT_EVIDENCE_FALLBACK
    return evidence, PROMPT_DERIVED_DRIFT_ARTIFACT_KIND


def _is_supported_prompt_contract(state: WhoseAgentState) -> bool:
    return (
        state.get("loop_source") == "prompt_contract"
        and state.get("prompt_contract_status") == "contract_detected"
        and state.get("selected_skill_id") is not None
    )


def _is_unsupported_prompt_contract(state: WhoseAgentState) -> bool:
    return (
        state.get("loop_source") == "prompt_contract"
        and state.get("prompt_contract_status") == "unsupported"
    )


def _concise_text(
    value: object,
    *,
    max_length: int,
    fallback: str | None,
) -> str | None:
    if not isinstance(value, str):
        return fallback
    text = " ".join(value.split())
    if not text or len(text) > max_length:
        return fallback
    return text


def _step_substituted(value: str | None) -> str | None:
    if value in {"instruction", "authority", "role", "model"}:
        return cast(str, value)
    return None
