"""Loop trace renderer: projects a completed WhoseAgentState into a LoopTrace artifact.

Pure projection only. Does not mutate state, call LLMs, run checkers, invoke graph
nodes, reload scenario files, or create a second state machine.
"""

from __future__ import annotations

from whose_agent.schemas import LoopTrace, WhoseAgentState


def render_loop_trace(state: WhoseAgentState) -> LoopTrace:
    """Project a completed WhoseAgentState into a LoopTrace artifact."""
    scenario = state.get("scenario")
    if scenario is None:
        raise ValueError("WhoseAgentState requires scenario.")

    return LoopTrace(
        scenario_id=scenario.scenario_id,
        loop_source=state.get("loop_source", "fixed_scenario"),
        prompt_contract_status=state.get("prompt_contract_status"),
        prompt_contract_candidate_framework=state.get(
            "prompt_contract_candidate_framework"
        ),
        prompt_contract_delegated_guarantee=state.get(
            "prompt_contract_delegated_guarantee"
        ),
        prompt_contract_artifact=state.get("prompt_contract_artifact"),
        principal=state.get("principal", "user"),
        agent=state.get("agent", "assistant"),
        max_iterations=int(state.get("max_iterations", 1)),
        final_loop_iteration=int(state.get("loop_iteration", 0)),
        loop_completed=bool(state.get("loop_completed", False)),
        loop_stop_reason=state.get("loop_stop_reason"),
        framework_specified=bool(state.get("framework_specified", False)),
        selected_skill_id=state.get("selected_skill_id"),
        generation_used_skill=bool(state.get("generation_used_skill", False)),
        checker_ran=bool(state.get("checker_ran", False)),
        checker_observed_bypass=bool(state.get("checker_observed_bypass", False)),
        guarantee_bypass_observed=bool(state.get("guarantee_bypass_observed", False)),
        checker_matches_expected=state.get("checker_matches_expected"),
        observation_outcome=state.get("observation_outcome"),
        step_traces=list(state.get("step_traces", [])),
        checker_comparison=state.get("checker_comparison"),
    )
