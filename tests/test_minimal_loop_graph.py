from __future__ import annotations

from pathlib import Path

from whose_agent import minimal_loop_graph, schemas
from whose_agent.bad_response import mock_bad_response
from whose_agent.classifier import classify_scenario
from whose_agent.llm_result import LLMCallResult
from whose_agent.loop_trigger_policy import should_fire_misreader_skill
from whose_agent.minimal_loop_graph import (
    build_minimal_loop_graph,
    compile_minimal_loop_graph,
    derive_framework_specified_for_scenario,
    initial_loop_state_from_scenario,
)
from whose_agent.scenario_loader import load_scenario
from whose_agent.state_graph import compile_fixed_scenario_graph
from whose_agent.state_graph import initial_state_from_scenario as fixed_initial_state


ROOT = Path(__file__).resolve().parents[1]


def _typescript_any() -> schemas.Scenario:
    return load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")


def _rust_cli() -> schemas.Scenario:
    return load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")


def _none_general_explanation() -> schemas.Scenario:
    return load_scenario(ROOT / "scenarios" / "none_general_explanation.yaml")


def _none_code_bugfix() -> schemas.Scenario:
    return load_scenario(ROOT / "scenarios" / "none_code_bugfix.yaml")


def test_minimal_loop_graph_compiles() -> None:
    graph = compile_minimal_loop_graph(mock=True)

    assert graph is not None


def test_loop_uses_schema_owned_langgraph_state() -> None:
    source = (ROOT / "src" / "whose_agent" / "minimal_loop_graph.py").read_text(encoding="utf-8")

    assert minimal_loop_graph.WhoseAgentState is schemas.WhoseAgentState
    assert "class WhoseAgentState" not in source
    # ControlState must not be wired as a nested runtime object in the loop:
    # it is neither imported nor instantiated.
    assert not hasattr(minimal_loop_graph, "ControlState")
    assert "ControlState(" not in source
    import_block = source.split("def ")[0]
    assert "import ControlState" not in import_block
    assert ", ControlState" not in import_block


def test_initial_loop_state_uses_whose_agent_state_shape() -> None:
    scenario = _typescript_any()

    state = initial_loop_state_from_scenario(scenario, max_iterations=3)

    # Same shape and keys as the fixed-scenario initial state, plus loop fields.
    fixed_keys = set(fixed_initial_state(scenario).keys())
    loop_keys = set(state.keys())
    assert fixed_keys <= loop_keys

    assert state["scenario"] == scenario.model_copy(update={"initial_messages": []})
    assert state["scenario"].initial_messages == []
    assert [(message.role, message.content) for message in state["messages"]] == [
        ("user", scenario.principal_prompt)
    ]
    assert state["messages"][0].message_id
    assert state["selected_skill_id"] == "safety_framework_escape_hatch"
    assert state["selected_skill_perspective"] is None
    assert state["framework_specified"] is False
    assert state["misreader_skill_fired"] is False
    assert state["checker_observed_bypass"] is False
    assert state["checker_comparison"] is None
    assert state["observation_outcome"] is None
    assert state["loop_iteration"] == 0
    assert state["max_iterations"] == 3
    assert state["loop_completed"] is False
    assert state["step_traces"] == []


def test_derive_framework_specified_is_scenario_grounded() -> None:
    assert derive_framework_specified_for_scenario(_typescript_any()) is True
    assert derive_framework_specified_for_scenario(_rust_cli()) is False


def test_fixed_scenario_non_fired_path_does_not_use_prompt_response_generator(
    monkeypatch,
) -> None:
    def fail_if_contract_response_generator_is_called(*args, **kwargs):
        raise AssertionError("fixed scenarios must not call prompt_response generator")

    monkeypatch.setattr(
        "whose_agent.minimal_loop_graph.generate_contract_preserving_response_with_usage",
        fail_if_contract_response_generator_is_called,
    )

    scenario = _rust_cli()
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))

    assert state["loop_source"] == "fixed_scenario"
    assert state["misreader_skill_fired"] is False
    assert state["bad_response"] == mock_bad_response(state["classification"])


def test_none_initial_loop_state_stops_before_skill_trigger() -> None:
    scenario = _none_general_explanation()
    classification = classify_scenario(scenario)

    state = initial_loop_state_from_scenario(scenario, max_iterations=1)
    planned_state = {
        **state,
        "classification": classification,
        "substituted": classification.substituted,
        "framework_specified": bool(state.get("framework_specified", False))
        or derive_framework_specified_for_scenario(scenario),
    }

    assert scenario.expected_substituted == "none"
    assert scenario.failure_mode == "none"
    assert classification.classification == "out_of_scope"
    assert classification.substituted == "none"
    assert state["substituted"] == "none"
    assert state["failure_mode"] == "none"
    assert planned_state["framework_specified"] is False
    assert planned_state["selected_skill_id"] is None
    assert should_fire_misreader_skill(planned_state) is False


def test_loop_stops_via_max_iterations() -> None:
    scenario = _typescript_any()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=2))

    assert state["loop_iteration"] == 2
    assert state["loop_completed"] is True
    assert state["loop_stop_reason"] == "max_iterations_reached"
    assert state["completed"] is True
    assert state["next_action"] == "stop"


def test_single_iteration_poor_e2e_for_typescript_any() -> None:
    scenario = _typescript_any()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))
    traces = state["step_traces"]

    # StepTrace preserves the plan -> do -> check sequence.
    assert [trace.step_kind for trace in traces] == ["plan", "do", "check"]
    assert [trace.step_index for trace in traces] == [0, 1, 2]

    plan_trace, do_trace, check_trace = traces

    # plan: framework_specified set, misreader not yet fired.
    assert state["framework_specified"] is True
    assert plan_trace.misreader_skill_fired is False

    # do: misreader fires (cause-side) and skill-informed generation runs.
    assert do_trace.misreader_skill_fired is True
    assert do_trace.selected_skill_id == "safety_framework_escape_hatch"
    assert do_trace.generation_used_skill is True
    assert do_trace.generation_skill_id == "safety_framework_escape_hatch"
    assert do_trace.trigger_evidence
    assert do_trace.drift_evidence is None
    assert do_trace.drift_artifact_kind is None
    assert state["generation_used_skill"] is True
    assert state["generation_skill_id"] == "safety_framework_escape_hatch"

    # check: checker observes the resulting boundary drift (observation-side).
    assert check_trace.checker_ran is True
    assert check_trace.checker_observed_bypass is True
    assert state["checker_observed_bypass"] is True
    assert state["guarantee_bypass_observed"] is True
    assert state["observation_outcome"] == "observation_succeeded"


def test_single_iteration_none_scenario_is_negative_control() -> None:
    scenario = _none_general_explanation()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))
    traces = state["step_traces"]

    assert [trace.step_kind for trace in traces] == ["plan", "do", "check"]
    assert [trace.step_index for trace in traces] == [0, 1, 2]

    plan_trace, do_trace, check_trace = traces
    classification = state["classification"]

    assert classification.classification == "out_of_scope"
    assert classification.substituted == "none"
    assert state["substituted"] == "none"
    assert state["failure_mode"] == "none"
    assert state["framework_specified"] is False
    assert state["selected_skill_id"] is None

    assert plan_trace.misreader_skill_fired is False
    assert do_trace.misreader_skill_fired is False
    assert do_trace.generation_used_skill is False
    assert do_trace.generation_skill_id is None
    assert do_trace.drift_evidence is None
    assert do_trace.drift_artifact_kind is None
    assert state["misreader_skill_fired"] is False
    assert state["skill_triggered"] is False
    assert state["generation_used_skill"] is False
    assert state["generation_skill_id"] is None
    assert state["bad_response"] is None

    assert check_trace.checker_ran is False
    assert check_trace.checker_observed_bypass is False
    assert state["checker_ran"] is False
    assert state["checker_observed_bypass"] is False
    assert state["guarantee_bypass_observed"] is False
    assert state["observation_outcome"] == "not_applicable"
    assert state["loop_completed"] is True


def test_single_iteration_none_code_bugfix_is_negative_control() -> None:
    scenario = _none_code_bugfix()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))
    traces = state["step_traces"]

    assert [trace.step_kind for trace in traces] == ["plan", "do", "check"]
    assert [trace.step_index for trace in traces] == [0, 1, 2]

    plan_trace, do_trace, check_trace = traces
    classification = state["classification"]

    assert classification.classification == "out_of_scope"
    assert classification.substituted == "none"
    assert state["substituted"] == "none"
    assert state["failure_mode"] == "none"
    assert state["framework_specified"] is False
    assert state["selected_skill_id"] is None

    assert plan_trace.misreader_skill_fired is False
    assert do_trace.misreader_skill_fired is False
    assert do_trace.generation_used_skill is False
    assert do_trace.generation_skill_id is None
    assert do_trace.drift_evidence is None
    assert do_trace.drift_artifact_kind is None
    assert state["misreader_skill_fired"] is False
    assert state["skill_triggered"] is False
    assert state["generation_used_skill"] is False
    assert state["generation_skill_id"] is None
    assert state["bad_response"] is None

    assert check_trace.checker_ran is False
    assert check_trace.checker_observed_bypass is False
    assert state["checker_ran"] is False
    assert state["checker_observed_bypass"] is False
    assert state["guarantee_bypass_observed"] is False
    assert state["observation_outcome"] == "not_applicable"
    assert state["loop_completed"] is True


def test_plan_sets_framework_and_does_not_fire_misreader() -> None:
    scenario = _typescript_any()
    # build_minimal_loop_graph returns an uncompiled StateGraph builder.
    graph = build_minimal_loop_graph(mock=True).compile()

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))

    plan_trace = state["step_traces"][0]
    assert plan_trace.step_kind == "plan"
    # The plan step records framework_specified in state but does not fire the
    # misreader; the plan trace stays cause-side clean.
    assert plan_trace.misreader_skill_fired is False
    assert state["framework_specified"] is True


def test_do_uses_selected_skill_generation_context(monkeypatch) -> None:
    scenario = _typescript_any()
    calls: dict[str, object] = {}

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
        minimal_loop_graph,
        "generate_bad_response_with_usage",
        fake_generate_bad_response_with_usage,
    )
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))

    assert calls["selected_skill_id"] == "safety_framework_escape_hatch"
    assert calls["misreader_skill_fired"] is True
    assert "surface framework" in str(calls["selected_skill_perspective"])
    assert state["generation_used_skill"] is True


def test_check_reuses_checker_comparison_logic() -> None:
    scenario = _typescript_any()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))
    comparison = state["checker_comparison"]

    assert comparison is not None
    assert comparison.matches_expected is True
    assert comparison.expected_checker_observed_bypass is True
    assert comparison.actual_checker_observed_bypass is True
    assert comparison.observation_outcome == "observation_succeeded"
    assert state["checker_matches_expected"] is True


def test_loop_does_not_use_checker_observation_as_misreader_precondition() -> None:
    source = (ROOT / "src" / "whose_agent" / "minimal_loop_graph.py").read_text(encoding="utf-8")
    do_block = source.split("def do(")[1].split("def check(")[0]

    # The do step (cause-side) must not read observation-side signals.
    assert "checker_observed_bypass" not in do_block
    assert "guarantee_bypass_observed" not in do_block
    assert "checker_comparison" not in do_block
    assert "should_fire_misreader_skill(state)" in do_block


def test_loop_without_selected_skill_does_not_fire_or_check() -> None:
    scenario = _none_general_explanation()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))

    assert state["framework_specified"] is False
    assert state["misreader_skill_fired"] is False
    assert state["skill_triggered"] is False
    assert state["generation_used_skill"] is False
    assert state["checker_ran"] is False
    assert state["checker_observed_bypass"] is False
    assert state["observation_outcome"] == "not_applicable"
    assert state["loop_completed"] is True


def test_repeated_loop_preserves_plan_do_check_sequence() -> None:
    scenario = _typescript_any()
    graph = compile_minimal_loop_graph(mock=True)

    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=3))
    kinds = [trace.step_kind for trace in state["step_traces"]]

    assert kinds == ["plan", "do", "check"] * 3
    assert state["loop_iteration"] == 3


def test_no_models_module_and_no_boundary_transitions_import() -> None:
    models_path = ROOT / "src" / "whose_agent" / "models.py"
    assert not models_path.exists()

    forbidden_import = "whose_agent.boundary_state." + "transitions"
    for path in (ROOT / "src").rglob("*.py"):
        assert forbidden_import not in path.read_text(encoding="utf-8")


def test_existing_fixed_scenario_graph_still_runs() -> None:
    scenario = _typescript_any()
    graph = compile_fixed_scenario_graph(mock=True)

    state = graph.invoke(fixed_initial_state(scenario))

    assert state["completed"] is True
    assert state["observation_outcome"] == "observation_succeeded"
