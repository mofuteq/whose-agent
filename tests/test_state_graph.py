from __future__ import annotations

from pathlib import Path

from whose_agent import schemas, state_graph
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
        "do",
        "do",
        "check",
        "check",
        "do",
        "check",
    ]
    assert {trace.principal for trace in step_traces} == {"user"}
    assert {trace.agent for trace in step_traces} == {"assistant"}
    assert step_traces[5].checker_ran is True
    assert step_traces[5].checker_observed_bypass is True
    assert step_traces[5].selected_skill_id == "safety_framework_escape_hatch"


def test_completed_becomes_true_at_finalize() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(initial_state_from_scenario(scenario))

    assert state["completed"] is True
    assert state["next_action"] == "stop"
    assert state["step_traces"][-1].step_kind == "check"
