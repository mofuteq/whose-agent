from __future__ import annotations

from pathlib import Path

from whose_agent import schemas, state_graph
from whose_agent.bad_response import mock_bad_response
from whose_agent.llm_result import LLMCallResult
from whose_agent.scenario_loader import load_scenario
from whose_agent.state_graph import compile_fixed_scenario_graph, initial_state_from_scenario


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_scenario_graph_compiles() -> None:
    graph = compile_fixed_scenario_graph(mock=True)

    assert graph is not None


def test_state_graph_uses_schema_owned_langgraph_state() -> None:
    state_graph_source = (ROOT / "src" / "whose_agent" / "state_graph.py").read_text(
        encoding="utf-8"
    )

    assert state_graph.WhoseAgentState is schemas.WhoseAgentState
    assert "from whose_agent.schemas import" in state_graph_source
    assert "class WhoseAgentState" not in state_graph_source


def test_graph_state_initializes_from_fixed_scenario() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")

    state = initial_state_from_scenario(scenario)

    assert state["scenario"] is scenario
    assert state["principal"] == "user"
    assert state["agent"] == "assistant"
    assert state["principal_instruction"] == scenario.principal_prompt
    assert state["principal_signal"] == scenario.principal_signal
    assert state["selected_skill_id"] == scenario.selected_skill_id
    assert state["substituted"] == scenario.expected_substituted
    assert state["failure_mode"] == scenario.failure_mode
    assert state["generation_used_skill"] is False
    assert state["generation_skill_id"] is None
    assert state["completed"] is False
    assert state["step_traces"] == []


def test_step_traces_are_appended_in_order_for_in_scope_scenario() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))
    step_traces = state["step_traces"]

    assert [trace.step_index for trace in step_traces] == list(range(len(step_traces)))
    assert [trace.step_kind for trace in step_traces] == [
        "plan",
        "plan",
        "plan",
        "do",
        "do",
        "check",
        "check",
        "check",
        "do",
        "check",
    ]
    assert {trace.principal for trace in step_traces} == {"user"}
    assert {trace.agent for trace in step_traces} == {"assistant"}
    assert step_traces[2].misreader_skill_fired is True
    assert step_traces[2].selected_skill_id == "safety_framework_escape_hatch"
    assert step_traces[2].trigger_evidence
    assert step_traces[6].checker_ran is True
    assert step_traces[6].checker_observed_bypass is True
    assert step_traces[6].misreader_skill_fired is True
    assert step_traces[6].selected_skill_id == "safety_framework_escape_hatch"


def test_selected_skill_scenario_records_trigger_state() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert state["skill_triggered"] is True
    assert state["misreader_skill_fired"] is True
    assert state["selected_skill_id"] == "safety_framework_escape_hatch"
    assert state["selected_skill_perspective"] is not None
    assert "surface framework" in state["selected_skill_perspective"]
    assert state["trigger_evidence"]
    assert "deterministic fixed scenario" in state["trigger_evidence"][0]
    assert state["generation_used_skill"] is True
    assert state["generation_skill_id"] == "safety_framework_escape_hatch"


def test_out_of_scope_scenario_keeps_trigger_state_false() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "none_general_explanation.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert state["skill_triggered"] is False
    assert state["misreader_skill_fired"] is False
    assert state["selected_skill_id"] is None
    assert state["selected_skill_perspective"] is None
    assert state["trigger_evidence"] == []
    assert state["generation_used_skill"] is False
    assert state["generation_skill_id"] is None
    assert state["bad_response"] is None


def test_graph_passes_selected_skill_state_into_bad_response_generation(
    monkeypatch,
) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    calls = {}

    def fake_generate_bad_response_with_usage(
        scenario,
        classification,
        *,
        selected_skill_id=None,
        selected_skill_perspective=None,
        misreader_skill_fired=False,
        mock=False,
    ):
        calls["selected_skill_id"] = selected_skill_id
        calls["selected_skill_perspective"] = selected_skill_perspective
        calls["misreader_skill_fired"] = misreader_skill_fired
        calls["mock"] = mock
        return LLMCallResult(output=mock_bad_response(classification))

    monkeypatch.setattr(
        state_graph,
        "generate_bad_response_with_usage",
        fake_generate_bad_response_with_usage,
    )
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert calls["selected_skill_id"] == "safety_framework_escape_hatch"
    assert calls["selected_skill_perspective"] == state["selected_skill_perspective"]
    assert "surface framework" in calls["selected_skill_perspective"]
    assert calls["misreader_skill_fired"] is True
    assert calls["mock"] is True
    assert state["generation_used_skill"] is True


def test_graph_passes_new_skill_context_for_rust_scenario(
    monkeypatch,
) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    calls = {}

    def fake_generate_bad_response_with_usage(
        scenario,
        classification,
        *,
        selected_skill_id=None,
        selected_skill_perspective=None,
        misreader_skill_fired=False,
        mock=False,
    ):
        calls["selected_skill_id"] = selected_skill_id
        calls["selected_skill_perspective"] = selected_skill_perspective
        calls["misreader_skill_fired"] = misreader_skill_fired
        return LLMCallResult(output=mock_bad_response(classification))

    monkeypatch.setattr(
        state_graph,
        "generate_bad_response_with_usage",
        fake_generate_bad_response_with_usage,
    )
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert calls["selected_skill_id"] == "instruction_constraint_override"
    assert "explicit implementation" in calls["selected_skill_perspective"]
    assert calls["misreader_skill_fired"] is True
    assert state["generation_used_skill"] is True


def test_checker_comparison_succeeds_for_typescript_any_mock_scenario() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))
    comparison = state["checker_comparison"]

    assert comparison is not None
    assert comparison.matches_expected is True
    assert comparison.mismatch_reasons == []
    assert comparison.expected_checker_observed_bypass is True
    assert comparison.actual_checker_observed_bypass is True
    assert comparison.expected_substituted == "instruction"
    assert comparison.actual_substituted == "instruction"
    assert comparison.expected_failure_mode == "constraint_override"
    assert comparison.actual_failure_mode == "constraint_override"
    assert comparison.observation_outcome == "observation_succeeded"
    assert state["checker_matches_expected"] is True
    assert state["observation_outcome"] == "observation_succeeded"


def test_none_scenario_does_not_run_checker() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "none_general_explanation.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert state["checker_observation"] is None
    assert state["checker_comparison"] is None
    assert state["checker_matches_expected"] is None
    assert state["observation_outcome"] is None


def test_completed_becomes_true_at_finalize() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert state["completed"] is True
    assert state["next_action"] == "stop"
    assert state["step_traces"][-1].step_kind == "check"
