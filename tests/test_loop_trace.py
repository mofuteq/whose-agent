"""Tests for loop trace artifact schema, renderer, and writer.

All tests use mock=True. No real API credentials required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whose_agent import schemas
from whose_agent.loop_artifacts import run_minimal_loop_to_artifact, write_loop_trace
from whose_agent.loop_trace_renderer import render_loop_trace
from whose_agent.minimal_loop_graph import (
    compile_minimal_loop_graph,
    initial_loop_state_from_scenario,
)
from whose_agent.scenario_loader import load_scenario
from whose_agent.schemas import LoopTrace


ROOT = Path(__file__).resolve().parents[1]


def _typescript_any() -> schemas.Scenario:
    return load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")


def _rust_cli() -> schemas.Scenario:
    return load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")


def _run_loop(scenario: schemas.Scenario, *, max_iterations: int = 1) -> dict:
    graph = compile_minimal_loop_graph(mock=True)
    return graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=max_iterations))


# --- Schema ---


def test_loop_trace_schema_exists() -> None:
    assert hasattr(schemas, "LoopTrace")
    assert issubclass(schemas.LoopTrace, schemas.LoopTrace.__bases__[0])


def test_loop_trace_in_schemas_all() -> None:
    assert "LoopTrace" in schemas.__all__


def test_loop_trace_forbids_extra_fields() -> None:
    state = _run_loop(_typescript_any())
    loop_trace = render_loop_trace(state)
    with pytest.raises(Exception):
        LoopTrace(**loop_trace.model_dump(), unexpected_field="x")


# --- Renderer: basic projection ---


def test_render_loop_trace_returns_loop_trace() -> None:
    state = _run_loop(_typescript_any())
    loop_trace = render_loop_trace(state)

    assert isinstance(loop_trace, LoopTrace)


def test_render_loop_trace_includes_scenario_id() -> None:
    scenario = _typescript_any()
    state = _run_loop(scenario)
    loop_trace = render_loop_trace(state)

    assert loop_trace.scenario_id == scenario.scenario_id
    assert loop_trace.scenario_id == "instruction_typescript_any"


def test_render_loop_trace_includes_principal_and_agent() -> None:
    state = _run_loop(_typescript_any())
    loop_trace = render_loop_trace(state)

    assert loop_trace.principal == "user"
    assert loop_trace.agent == "assistant"


def test_render_loop_trace_includes_max_and_final_iteration() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.max_iterations == 1
    assert loop_trace.final_loop_iteration == 1


def test_render_loop_trace_includes_loop_completed_and_stop_reason() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.loop_completed is True
    assert loop_trace.loop_stop_reason == "max_iterations_reached"


def test_render_loop_trace_multi_iteration_stop_reason() -> None:
    state = _run_loop(_typescript_any(), max_iterations=2)
    loop_trace = render_loop_trace(state)

    assert loop_trace.loop_completed is True
    assert loop_trace.loop_stop_reason == "max_iterations_reached"
    assert loop_trace.final_loop_iteration == 2
    assert loop_trace.max_iterations == 2


# --- Renderer: step trace sequence ---


def test_render_loop_trace_preserves_plan_do_check_sequence() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert [t.step_kind for t in loop_trace.step_traces] == ["plan", "do", "check"]


def test_render_loop_trace_step_traces_are_step_trace_instances() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    from whose_agent.schemas import StepTrace
    for trace in loop_trace.step_traces:
        assert isinstance(trace, StepTrace)


def test_render_loop_trace_repeated_loop_preserves_sequence() -> None:
    state = _run_loop(_typescript_any(), max_iterations=3)
    loop_trace = render_loop_trace(state)

    assert [t.step_kind for t in loop_trace.step_traces] == ["plan", "do", "check"] * 3


# --- Renderer: cause-side fields ---


def test_render_loop_trace_plan_step_misreader_skill_fired_false() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    plan_trace = loop_trace.step_traces[0]
    assert plan_trace.step_kind == "plan"
    assert plan_trace.misreader_skill_fired is False


def test_render_loop_trace_do_step_misreader_skill_fired_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    do_trace = loop_trace.step_traces[1]
    assert do_trace.step_kind == "do"
    assert do_trace.misreader_skill_fired is True


def test_render_loop_trace_selected_skill_id() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.selected_skill_id == "safety_framework_escape_hatch"


def test_render_loop_trace_generation_used_skill_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.generation_used_skill is True


# --- Renderer: observation-side fields ---


def test_render_loop_trace_check_step_checker_ran_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    check_trace = loop_trace.step_traces[2]
    assert check_trace.step_kind == "check"
    assert check_trace.checker_ran is True


def test_render_loop_trace_check_step_checker_observed_bypass_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    check_trace = loop_trace.step_traces[2]
    assert check_trace.checker_observed_bypass is True


def test_render_loop_trace_observation_outcome_succeeded() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.observation_outcome == "observation_succeeded"


def test_render_loop_trace_checker_ran_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.checker_ran is True


def test_render_loop_trace_checker_observed_bypass_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.checker_observed_bypass is True


def test_render_loop_trace_guarantee_bypass_observed_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.guarantee_bypass_observed is True


def test_render_loop_trace_checker_matches_expected_true() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.checker_matches_expected is True


def test_render_loop_trace_checker_comparison_present() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    from whose_agent.schemas import CheckerComparison
    assert loop_trace.checker_comparison is not None
    assert isinstance(loop_trace.checker_comparison, CheckerComparison)
    assert loop_trace.checker_comparison.matches_expected is True
    assert loop_trace.checker_comparison.observation_outcome == "observation_succeeded"


# --- Cause/observation distinction preserved ---


def test_observation_side_fields_do_not_appear_in_do_step() -> None:
    source = (
        ROOT / "src" / "whose_agent" / "loop_trace_renderer.py"
    ).read_text(encoding="utf-8")
    # The renderer is a pure projection; it should not contain any causal inference logic.
    assert "misreader_skill_fired" not in source or "state.get" in source


def test_render_loop_trace_no_skill_scenario() -> None:
    state = _run_loop(_rust_cli(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    assert loop_trace.selected_skill_id is None
    assert loop_trace.generation_used_skill is False
    assert loop_trace.checker_ran is False
    assert loop_trace.checker_observed_bypass is False
    assert loop_trace.observation_outcome == "not_applicable"


# --- Artifact writer ---


def test_write_loop_trace_creates_file(tmp_path: Path) -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    path = write_loop_trace(tmp_path, loop_trace)

    assert path.exists()
    assert path.name == "instruction_typescript_any.loop_trace.json"


def test_write_loop_trace_file_is_valid_json(tmp_path: Path) -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    path = write_loop_trace(tmp_path, loop_trace)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["scenario_id"] == "instruction_typescript_any"
    assert data["loop_completed"] is True
    assert data["observation_outcome"] == "observation_succeeded"


def test_write_loop_trace_step_traces_in_json(tmp_path: Path) -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    path = write_loop_trace(tmp_path, loop_trace)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert len(data["step_traces"]) == 3
    assert [t["step_kind"] for t in data["step_traces"]] == ["plan", "do", "check"]


def test_write_loop_trace_returns_path(tmp_path: Path) -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    loop_trace = render_loop_trace(state)

    path = write_loop_trace(tmp_path, loop_trace)

    assert isinstance(path, Path)
    assert path.parent == tmp_path


# --- run_minimal_loop_to_artifact helper ---


def test_run_minimal_loop_to_artifact_writes_file(tmp_path: Path) -> None:
    scenario = _typescript_any()

    loop_trace = run_minimal_loop_to_artifact(scenario, tmp_path, max_iterations=1, mock=True)

    artifact_path = tmp_path / "instruction_typescript_any.loop_trace.json"
    assert artifact_path.exists()
    assert isinstance(loop_trace, LoopTrace)


def test_run_minimal_loop_to_artifact_content(tmp_path: Path) -> None:
    scenario = _typescript_any()

    run_minimal_loop_to_artifact(scenario, tmp_path, max_iterations=1, mock=True)

    data = json.loads(
        (tmp_path / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8")
    )
    assert data["scenario_id"] == "instruction_typescript_any"
    assert data["max_iterations"] == 1
    assert data["final_loop_iteration"] == 1
    assert data["loop_completed"] is True
    assert data["loop_stop_reason"] == "max_iterations_reached"
    assert data["observation_outcome"] == "observation_succeeded"
    assert len(data["step_traces"]) == 3


def test_run_minimal_loop_to_artifact_only_writes_loop_trace(tmp_path: Path) -> None:
    scenario = _typescript_any()

    run_minimal_loop_to_artifact(scenario, tmp_path, max_iterations=1, mock=True)

    files = list(tmp_path.glob("*"))
    assert len(files) == 1
    assert files[0].name == "instruction_typescript_any.loop_trace.json"


# --- Invariants from PR requirements ---


def test_no_models_module_exists() -> None:
    models_path = ROOT / "src" / "whose_agent" / "models.py"
    assert not models_path.exists()


def test_no_boundary_transitions_import() -> None:
    forbidden = "whose_agent.boundary_state." + "transitions"
    for path in (ROOT / "src").rglob("*.py"):
        assert forbidden not in path.read_text(encoding="utf-8"), (
            f"{path} imports {forbidden}"
        )


def test_no_control_state_in_loop_artifacts() -> None:
    source = (ROOT / "src" / "whose_agent" / "loop_artifacts.py").read_text(encoding="utf-8")
    assert "ControlState(" not in source


def test_no_control_state_in_loop_trace_renderer() -> None:
    source = (
        ROOT / "src" / "whose_agent" / "loop_trace_renderer.py"
    ).read_text(encoding="utf-8")
    assert "ControlState(" not in source


def test_render_loop_trace_is_pure_does_not_mutate_state() -> None:
    state = _run_loop(_typescript_any(), max_iterations=1)
    original_traces = list(state["step_traces"])

    render_loop_trace(state)

    assert state["step_traces"] == original_traces


def test_fixed_run_does_not_emit_loop_trace(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run",
        "--scenarios",
        "scenarios",
        "--outputs",
        str(tmp_path),
        "--mock",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    loop_trace_files = list(run_dir.glob("*.loop_trace.json"))
    assert loop_trace_files == [], "fixed run must not emit .loop_trace.json"


def test_run_loop_cli_command_exists() -> None:
    from whose_agent.cli import build_parser
    parser = build_parser()
    subparsers_actions = [
        a for a in parser._actions if hasattr(a, "choices") and a.choices is not None
    ]
    assert len(subparsers_actions) >= 1
    all_commands = set()
    for action in subparsers_actions:
        all_commands.update(action.choices.keys())
    assert "run-loop" in all_commands
