from __future__ import annotations

from pathlib import Path

from whose_agent.loop_artifacts import (
    write_loop_trace,
    write_prompt_loop_generated,
)
from whose_agent.loop_trace_renderer import render_loop_trace
from whose_agent.minimal_loop_graph import (
    compile_minimal_loop_graph,
    initial_loop_state_from_scenario,
)
from whose_agent.prompt_contract_artifacts import write_prompt_contract
from whose_agent.prompt_contract_detector import detect_prompt_contract
from whose_agent.schemas import (
    PromptContract,
    Scenario,
    ScenarioCheckerTemplate,
    Substituted,
    WhoseAgentState,
)


PROMPT_LOOP_SCENARIO_ID = "prompt_loop"


def initial_loop_state_from_prompt_contract(
    contract: PromptContract,
    *,
    max_iterations: int,
    misreader_firing_decision: bool | None = None,
) -> WhoseAgentState:
    """Convert a prompt contract into the existing minimal-loop state shape."""
    scenario = _scenario_from_prompt_contract(contract)
    state = initial_loop_state_from_scenario(
        scenario,
        max_iterations=max_iterations,
    )
    state["framework_specified"] = contract.framework_specified
    state["selected_skill_id"] = contract.selected_skill_id
    state["misreader_firing_decision"] = misreader_firing_decision
    state["loop_source"] = "prompt_contract"
    state["prompt_contract_status"] = contract.status
    state["prompt_contract_candidate_framework"] = contract.candidate_framework
    state["prompt_contract_delegated_guarantee"] = contract.delegated_guarantee
    state["prompt_contract_artifact"] = None
    state["prompt_loop_generated_artifact"] = None
    state["prompt_loop_generated_step_index"] = None
    return state


def run_prompt_loop_to_artifact(
    prompt: str,
    output_dir: Path,
    *,
    mock: bool = False,
    max_iterations: int = 1,
    misreader_firing_decision: bool | None = None,
) -> tuple[Path, Path, Path | None]:
    """Detect a prompt contract, run the minimal loop, and write artifacts."""
    contract = detect_prompt_contract(prompt, mock=mock)
    contract_path = write_prompt_contract(contract, output_dir)

    graph = compile_minimal_loop_graph(mock=mock)
    initial_state = initial_loop_state_from_prompt_contract(
        contract,
        max_iterations=max_iterations,
        misreader_firing_decision=misreader_firing_decision,
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

    return contract_path, loop_trace_path, generated_path


def _scenario_from_prompt_contract(contract: PromptContract) -> Scenario:
    expected_substituted: Substituted = (
        "instruction" if contract.framework_specified else "none"
    )
    failure_mode = (
        "constraint_override" if contract.framework_specified else "none"
    )
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
        principal_signal=contract.delegated_guarantee or contract.prompt,
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
    if not contract.framework_specified:
        return ""

    guarantee = contract.delegated_guarantee or "the delegated guarantee"
    framework = contract.candidate_framework or "the requested framework"
    return (
        f"Represent the prompt-derived boundary for {framework}: preserve "
        f"{guarantee} unless the selected misreader skill fires in the loop state."
    )


def _should_emit_prompt_loop_generated(contract: PromptContract) -> bool:
    return (
        contract.status == "contract_detected"
        and contract.selected_skill_id is not None
    )


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
