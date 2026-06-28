from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from whose_agent.firing_signals import FiringSignals, production_firing_signals
from whose_agent.loop_artifacts import (
    write_loop_trace,
    write_prompt_loop_generated,
)
from whose_agent.loop_trace_renderer import render_loop_trace
from whose_agent.minimal_loop_graph import (
    compile_minimal_loop_graph,
    initial_loop_state_from_scenario,
)
from whose_agent.authority_provenance import (
    derive_external_persistence_provenance,
)
from whose_agent.conversation_view import project_messages
from whose_agent.history_adapter import conversation_from_prompt, require_unique_message_ids
from whose_agent.prompt_contract_artifacts import write_prompt_contract
from whose_agent.prompt_contract_detector import detect_prompt_contract
from whose_agent.schemas import (
    ConversationMessage,
    EXPECTED_FAILURE_BY_SUBSTITUTED,
    PromptContract,
    Scenario,
    ScenarioCheckerTemplate,
    Substituted,
    WhoseAgentState,
)
from whose_agent.self_explanation import write_self_explanation


PROMPT_LOOP_SCENARIO_ID = "prompt_loop"


def initial_loop_state_from_prompt_contract(
    contract: PromptContract,
    *,
    max_iterations: int,
    misreader_firing_decision: bool | None = None,
    firing_signals: FiringSignals | None = None,
    messages: list[ConversationMessage] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> WhoseAgentState:
    """Convert a prompt contract into the existing minimal-loop state shape."""
    scenario = _scenario_from_prompt_contract(contract)
    canonical_messages = (
        list(messages) if messages is not None else conversation_from_prompt(contract.prompt)
    )
    require_unique_message_ids(canonical_messages)
    state = initial_loop_state_from_scenario(
        scenario,
        max_iterations=max_iterations,
        messages=canonical_messages,
    )
    state["boundary_detected"] = contract.boundary_detected
    state["substitution_axis"] = contract.substitution_axis
    state["delegated_boundary"] = contract.delegated_boundary
    state["framework_specified"] = contract.framework_specified
    state["selected_skill_id"] = contract.selected_skill_id
    state["misreader_firing_decision"] = misreader_firing_decision
    state["firing_signals"] = (
        firing_signals
        if firing_signals is not None
        else production_firing_signals(clock=clock)
    )
    state["firing_reason"] = None
    state["loop_source"] = "prompt_contract"
    state["prompt_contract_status"] = contract.status
    state["prompt_contract_boundary_detected"] = contract.boundary_detected
    state["prompt_contract_substitution_axis"] = contract.substitution_axis
    state["prompt_contract_delegated_boundary"] = contract.delegated_boundary
    state["prompt_contract_candidate_framework"] = contract.candidate_framework
    state["prompt_contract_delegated_guarantee"] = contract.delegated_guarantee
    state["prompt_contract_artifact"] = None
    state["prompt_loop_generated_artifact"] = None
    state["prompt_loop_generated_step_index"] = None
    state["authority_provenance"] = (
        derive_external_persistence_provenance(project_messages(canonical_messages))
    )
    return state


def run_prompt_loop_to_artifact(
    prompt: str,
    output_dir: Path,
    *,
    mock: bool = False,
    max_iterations: int = 1,
    misreader_firing_decision: bool | None = None,
    firing_signals: FiringSignals | None = None,
    messages: list[ConversationMessage] | None = None,
    clock: Callable[[], datetime] | None = None,
    tracer: object | None = None,
) -> tuple[Path, Path, Path | None]:
    """Detect a prompt contract, run the minimal loop, and write artifacts."""
    canonical_messages = (
        list(messages) if messages is not None else conversation_from_prompt(prompt)
    )
    require_unique_message_ids(canonical_messages)
    authority_provenance = derive_external_persistence_provenance(
        project_messages(canonical_messages)
    )
    if authority_provenance is None:
        contract = detect_prompt_contract(prompt, mock=mock)
    else:
        contract = detect_prompt_contract(
            prompt,
            mock=mock,
            authority_provenance=authority_provenance,
        )
    contract_path = write_prompt_contract(contract, output_dir)

    graph = compile_minimal_loop_graph(mock=mock, tracer=tracer)
    initial_state = initial_loop_state_from_prompt_contract(
        contract,
        max_iterations=max_iterations,
        misreader_firing_decision=misreader_firing_decision,
        firing_signals=firing_signals,
        messages=canonical_messages,
        clock=clock,
    )
    initial_state["prompt_contract_artifact"] = contract_path.name
    state = graph.invoke(initial_state)

    generated_path: Path | None = None
    if _should_emit_prompt_loop_generated(contract):
        generated_output = state.get("bad_response")
        if generated_output is None:
            raise RuntimeError(
                "supported prompt loop did not produce generated output"
            )
        generated_step_index = _last_do_step_index(state)
        if generated_step_index is None:
            raise RuntimeError("supported prompt loop did not record a do step")
        generated_path = write_prompt_loop_generated(output_dir, generated_output)
        state["prompt_loop_generated_artifact"] = generated_path.name
        state["prompt_loop_generated_step_index"] = generated_step_index

    loop_trace = render_loop_trace(state)
    loop_trace_path = write_loop_trace(output_dir, loop_trace)
    self_explanation = state.get("self_explanation")
    if self_explanation is not None:
        write_self_explanation(output_dir, PROMPT_LOOP_SCENARIO_ID, self_explanation)

    return contract_path, loop_trace_path, generated_path


def _scenario_from_prompt_contract(contract: PromptContract) -> Scenario:
    expected_substituted: Substituted = (
        contract.substitution_axis
        if _is_supported_prompt_contract(contract)
        and contract.substitution_axis is not None
        else "none"
    )
    failure_mode = EXPECTED_FAILURE_BY_SUBSTITUTED[expected_substituted]
    checker_template = (
        _checker_template_from_prompt_contract(contract)
        if contract.selected_skill_id is not None
        else None
    )

    return Scenario(
        scenario_id=PROMPT_LOOP_SCENARIO_ID,
        expected_substituted=expected_substituted,
        failure_mode=failure_mode,
        selected_skill_id=contract.selected_skill_id,
        principal_prompt=contract.prompt,
        principal_signal=contract.delegated_boundary or contract.prompt,
        generation_instruction=_generation_instruction_from_prompt_contract(contract),
        trace_template=None,
        checker_template=checker_template,
    )


def _checker_template_from_prompt_contract(
    contract: PromptContract,
) -> ScenarioCheckerTemplate:
    return ScenarioCheckerTemplate(
        checker_observed_bypass=False,
        substituted="none",
        failure_mode="none",
        evidence=[
            "The prompt-derived contract gives the checker a meaningful "
            "boundary to observe; no bypass is assumed before generation.",
        ],
        divergence_point=None,
        confidence=contract.confidence,
    )


def _generation_instruction_from_prompt_contract(contract: PromptContract) -> str:
    if not _is_supported_prompt_contract(contract):
        return ""

    axis = contract.substitution_axis or "boundary"
    delegated_boundary = contract.delegated_boundary or "the delegated boundary"
    framework_context = (
        f" Framework context: {contract.candidate_framework}."
        if contract.candidate_framework is not None
        else ""
    )
    return (
        f"Represent the prompt-derived {axis} boundary: preserve "
        f"{delegated_boundary}.{framework_context}"
    )


def _should_emit_prompt_loop_generated(contract: PromptContract) -> bool:
    return _is_supported_prompt_contract(contract)


def _is_supported_prompt_contract(contract: PromptContract) -> bool:
    return contract.status == "contract_detected" and contract.selected_skill_id is not None


def _last_do_step_index(state: WhoseAgentState) -> int | None:
    for step_trace in reversed(state.get("step_traces", [])):
        if step_trace.step_kind == "do":
            return step_trace.step_index
    return None


__all__ = [
    "PROMPT_LOOP_SCENARIO_ID",
    "initial_loop_state_from_prompt_contract",
    "run_prompt_loop_to_artifact",
]
