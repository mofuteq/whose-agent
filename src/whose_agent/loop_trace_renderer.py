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
        history_source=state.get("history_source"),
        prompt_loop_preset_id=state.get("prompt_loop_preset_id"),
        prompt_loop_actor_mode=state.get("prompt_loop_actor_mode"),
        prior_completed_agent_turns=int(
            state.get("prior_completed_agent_turns", 0)
        ),
        boundary_detected=bool(state.get("boundary_detected", False)),
        substitution_axis=state.get("substitution_axis"),
        delegated_boundary=state.get("delegated_boundary"),
        prompt_contract_status=state.get("prompt_contract_status"),
        prompt_contract_boundary_detected=state.get(
            "prompt_contract_boundary_detected"
        ),
        prompt_contract_substitution_axis=state.get(
            "prompt_contract_substitution_axis"
        ),
        prompt_contract_delegated_boundary=state.get(
            "prompt_contract_delegated_boundary"
        ),
        prompt_contract_candidate_framework=state.get(
            "prompt_contract_candidate_framework"
        ),
        prompt_contract_delegated_guarantee=state.get(
            "prompt_contract_delegated_guarantee"
        ),
        prompt_contract_source=state.get("prompt_contract_source"),
        prompt_contract_source_turn_indexes=list(
            state.get("prompt_contract_source_turn_indexes", [])
        ),
        prompt_contract_artifact=state.get("prompt_contract_artifact"),
        prompt_loop_generated_artifact=state.get("prompt_loop_generated_artifact"),
        prompt_loop_generated_step_index=state.get(
            "prompt_loop_generated_step_index"
        ),
        authority_provenance=state.get("authority_provenance"),
        principal=state.get("principal", "user"),
        agent=state.get("agent", "assistant"),
        max_iterations=int(state.get("max_iterations", 1)),
        final_loop_iteration=int(state.get("loop_iteration", 0)),
        loop_completed=bool(state.get("loop_completed", False)),
        loop_stop_reason=state.get("loop_stop_reason"),
        framework_specified=bool(state.get("framework_specified", False)),
        selected_skill_id=state.get("selected_skill_id"),
        misreader_firing_decision=state.get("misreader_firing_decision"),
        firing_signals=state.get("firing_signals"),
        firing_reason=state.get("firing_reason"),
        generation_used_skill=bool(state.get("generation_used_skill", False)),
        checker_ran=bool(state.get("checker_ran", False)),
        checker_observed_bypass=bool(state.get("checker_observed_bypass", False)),
        guarantee_bypass_observed=bool(state.get("guarantee_bypass_observed", False)),
        checker_matches_expected=state.get("checker_matches_expected"),
        observation_outcome=state.get("observation_outcome"),
        step_traces=list(state.get("step_traces", [])),
        checker_comparison=state.get("checker_comparison"),
        self_explanation=state.get("self_explanation"),
    )
