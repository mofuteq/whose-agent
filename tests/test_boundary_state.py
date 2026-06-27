from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import single_run_dir
from whose_agent.scenario_loader import load_scenario
from whose_agent.state_graph import compile_fixed_scenario_graph, initial_state_from_scenario
from whose_agent.state_trace_renderer import (
    STATE_TRACE_STEPS,
    render_boundary_state_trace,
)


ROOT = Path(__file__).resolve().parents[1]


def test_render_boundary_state_trace_builds_from_completed_langgraph_state() -> None:
    state = completed_state_for("instruction_rust_cli")

    state_trace = render_boundary_state_trace(state)

    assert state_trace.scenario_id == "rust_cli_constraint_override"
    assert [item.step for item in state_trace.transitions] == STATE_TRACE_STEPS
    assert len(state_trace.transitions) == 5


def test_rendered_final_boundary_state_matches_expected_reflection() -> None:
    state = completed_state_for("instruction_rust_cli")
    state_trace = render_boundary_state_trace(state)

    final = state_trace.transitions[-1].state

    assert final.reflection_matches_expected is True
    assert final.boundary_flags == ["constraint_override"]
    assert final.next_action == "trace_ready"


def test_rendered_state_trace_preserves_artifact_state_shape() -> None:
    state = completed_state_for("instruction_typescript_any")
    state_trace = render_boundary_state_trace(state)
    scenario = state["scenario"]
    trace = state["trace"]
    assert trace is not None

    final = state_trace.transitions[-1].state

    assert final.scenario_id == scenario.scenario_id
    assert final.principal_prompt == scenario.principal_prompt
    assert final.principal_signal == scenario.principal_signal
    assert final.expected_substituted == "instruction"
    assert final.failure_mode == "constraint_override"
    assert final.bad_response == state["bad_response"]
    assert final.reflection_substituted == trace.reflection_substituted
    assert final.why_it_breaks_delegation == trace.why_it_breaks_delegation
    assert final.better_behavior == trace.better_behavior
    assert set(final.model_dump()) == {
        "scenario_id",
        "principal_prompt",
        "principal_signal",
        "expected_substituted",
        "failure_mode",
        "bad_response",
        "reflection_substituted",
        "reflection_matches_expected",
        "boundary_flags",
        "why_it_breaks_delegation",
        "better_behavior",
        "next_action",
    }


def test_state_graph_stores_rendered_boundary_state_trace() -> None:
    state = completed_state_for("instruction_typescript_any")
    rendered = render_boundary_state_trace(state)

    assert state["state_trace"] == rendered
    assert state["boundary_flags"] == ["constraint_override"]
    assert state["next_action"] == "stop"


def test_renderer_does_not_use_legacy_transition_runtime() -> None:
    transitions_path = ROOT / "src" / "whose_agent" / "boundary_state" / "transitions.py"
    forbidden_import = "whose_agent.boundary_state." + "transitions"

    assert not transitions_path.exists()
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert forbidden_import not in source


def test_cli_mock_run_emits_state_trace_json(tmp_path: Path) -> None:
    run_fixed_cli(tmp_path)
    run_dir = single_run_dir(tmp_path)

    state_trace_files = list(run_dir.glob("*.state_trace.json"))
    assert len(state_trace_files) == 7


def test_state_trace_json_remains_structurally_compatible(tmp_path: Path) -> None:
    run_fixed_cli(tmp_path)
    run_dir = single_run_dir(tmp_path)

    state_trace_path = run_dir / "instruction_typescript_any.state_trace.json"
    state_trace = json.loads(state_trace_path.read_text(encoding="utf-8"))

    assert set(state_trace) == {"scenario_id", "transitions"}
    assert state_trace["scenario_id"] == "instruction_typescript_any"
    assert [item["step"] for item in state_trace["transitions"]] == STATE_TRACE_STEPS
    final = state_trace["transitions"][-1]["state"]
    assert final["reflection_matches_expected"] is True
    assert final["boundary_flags"] == ["constraint_override"]
    assert final["next_action"] == "trace_ready"


def test_mock_mode_does_not_require_openrouter_credentials(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "OPENROUTER_API_KEY"}
    env["PYTHONPATH"] = str(ROOT / "src")

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
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def completed_state_for(scenario_name: str):
    scenario = load_scenario(ROOT / "scenarios" / f"{scenario_name}.yaml")
    graph = compile_fixed_scenario_graph(mock=True)
    return graph.invoke(initial_state_from_scenario(scenario))


def run_fixed_cli(outputs: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run",
        "--scenarios",
        "scenarios",
        "--outputs",
        str(outputs),
        "--mock",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
