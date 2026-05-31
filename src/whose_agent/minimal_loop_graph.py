"""Minimal LangGraph loop path that demonstrates intermittent boundary drift.

Schema definitions live in whose_agent.schemas; this module owns a small
plan -> do -> check loop only. It is separate from the fixed scenario graph in
whose_agent.state_graph and does not replace it.

LangGraph state (WhoseAgentState) is the runtime source of truth. ControlState
is not wired in as a nested runtime object; the loop fields live directly on
WhoseAgentState.

Causal rule: misreader_skill_fired is the cause-side event. It is set in the do
step from cause-side conditions only (framework_specified + selected_skill_id).
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
from whose_agent.schemas import (
    Classification,
    Scenario,
    StepKind,
    StepTrace,
    WhoseAgentState,
)


CHECKER_ID = "skill-perspective-checker"


def derive_framework_specified_for_scenario(scenario: Scenario) -> bool:
    """Decide deterministically whether the principal specified a framework.

    Fixed scenarios can use deterministic metadata: a scenario that selects a
    misreader skill on the instruction axis is one where the principal named a
    surface framework plus a guarantee that a misreader skill could bypass.

    Arbitrary run-prompt loop support would later need actual instruction
    reading (real framework detection from the prompt text). That is
    intentionally out of scope here: no LLM judgment and no arbitrary prompt
    framework detection are performed in this helper.
    """
    return scenario.selected_skill_id is not None and scenario.expected_substituted == "instruction"


def initial_loop_state_from_scenario(
    scenario: Scenario,
    *,
    max_steps: int = 3,
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
        "framework_specified": False,
        "loop_iteration": 0,
        "loop_phase": "plan",
        "max_steps": max_steps,
        "loop_completed": False,
        "loop_stop_reason": None,
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
        framework_specified = derive_framework_specified_for_scenario(scenario)
        # Re-planning resets the cause-side misreader flag at the start of each
        # iteration so the loop can demonstrate intermittent drift.
        return {
            "classification": classification,
            "substituted": classification.substituted,
            "framework_specified": framework_specified,
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
        framework_specified = bool(state.get("framework_specified", False))
        selected_skill_id = state.get("selected_skill_id")

        # Cause-side firing condition only. Checker observation is never read here.
        should_fire = framework_specified and selected_skill_id is not None

        if should_fire:
            selected_skill_perspective = state.get("selected_skill_perspective")
            if selected_skill_perspective is None:
                selected_skill_perspective = load_skill_perspective(cast(str, selected_skill_id))
            trigger_evidence = [
                f"framework_specified with selected skill {selected_skill_id!r} on the do "
                "step; the misreader skill fires and the artifact drifts past the guarantee."
            ]
            bad_response = generate_bad_response_with_usage(
                scenario,
                classification,
                selected_skill_id=selected_skill_id,
                selected_skill_perspective=selected_skill_perspective,
                misreader_skill_fired=True,
                mock=mock,
            ).output
            generation_used_skill = selected_skill_perspective is not None
            return {
                "selected_skill_perspective": selected_skill_perspective,
                "skill_triggered": True,
                "misreader_skill_fired": True,
                "trigger_evidence": trigger_evidence,
                "bad_response": bad_response,
                "generation_used_skill": generation_used_skill,
                "generation_skill_id": selected_skill_id if generation_used_skill else None,
                "loop_phase": "do",
                **_step_update(
                    state,
                    "do",
                    misreader_skill_fired=True,
                    selected_skill_id=selected_skill_id,
                    trigger_evidence=trigger_evidence,
                    substituted=classification.substituted,
                ),
            }

        # Misreader does not fire: generation must not use skill context.
        bad_response = generate_bad_response_with_usage(
            scenario,
            classification,
            mock=mock,
        ).output
        return {
            "skill_triggered": False,
            "misreader_skill_fired": False,
            "bad_response": bad_response,
            "generation_used_skill": False,
            "generation_skill_id": None,
            "loop_phase": "do",
            **_step_update(
                state,
                "do",
                misreader_skill_fired=False,
                selected_skill_id=selected_skill_id,
                substituted=classification.substituted,
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
        )

        loop_iteration = int(state.get("loop_iteration", 0)) + 1
        max_steps = int(state.get("max_steps", 3))
        loop_completed = loop_iteration >= max_steps
        loop_stop_reason = "max_steps_reached" if loop_completed else None

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
    max_steps = int(state.get("max_steps", 3))
    if bool(state.get("loop_completed", False)) or loop_iteration >= max_steps:
        return "end"
    return "plan"


def _step_update(
    state: WhoseAgentState,
    step_kind: StepKind,
    *,
    misreader_skill_fired: bool = False,
    selected_skill_id: str | None = None,
    trigger_evidence: list[str] | None = None,
    checker_ran: bool = False,
    checker_observed_bypass: bool = False,
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
        checker_ran=checker_ran,
        checker_observed_bypass=checker_observed_bypass,
        trigger_evidence=list(trigger_evidence or []),
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


def _step_substituted(value: str | None) -> str | None:
    if value in {"instruction", "authority", "role", "model"}:
        return cast(str, value)
    return None
