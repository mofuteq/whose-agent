"""Tests for the run-loop CLI command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_loop_cli(
    scenario_path: str,
    outputs: Path,
    *,
    max_iterations: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run-loop",
        "--scenario",
        scenario_path,
        "--outputs",
        str(outputs),
        "--mock",
    ]
    if max_iterations is not None:
        command += ["--max-iterations", str(max_iterations)]
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


def _run_dir(outputs: Path) -> Path:
    run_dirs = [d for d in outputs.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    return run_dirs[0]


# --- Directory and artifact count ---


def test_run_loop_creates_exactly_one_run_directory(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1


def test_run_loop_creates_exactly_one_loop_trace_file(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    loop_traces = list(run_dir.glob("*.loop_trace.json"))
    assert len(loop_traces) == 1
    assert loop_traces[0].name == "instruction_typescript_any.loop_trace.json"


# --- No fixed scenario artifacts ---


def test_run_loop_does_not_emit_classification(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.classification.json")) == []


def test_run_loop_does_not_emit_response(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.response.md")) == []


def test_run_loop_does_not_emit_trace(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.trace.json")) == []


def test_run_loop_does_not_emit_state_trace(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.state_trace.json")) == []


def test_run_loop_does_not_emit_checker(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.checker.json")) == []


def test_run_loop_does_not_emit_checker_comparison(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.checker_comparison.json")) == []


def test_run_loop_does_not_emit_flow(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert list(run_dir.glob("*.flow.mmd")) == []


# --- Loop trace content ---


def test_run_loop_trace_scenario_id(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert data["scenario_id"] == "instruction_typescript_any"


def test_run_loop_trace_fixed_scenario_provenance(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert data["loop_source"] == "fixed_scenario"
    assert data["prompt_contract_status"] is None
    assert data["prompt_contract_candidate_framework"] is None
    assert data["prompt_contract_delegated_guarantee"] is None
    assert data["prompt_contract_artifact"] is None


def test_run_loop_trace_default_max_iterations(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert data["max_iterations"] == 1
    assert data["final_loop_iteration"] == 1


def test_run_loop_trace_loop_completed(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert data["loop_completed"] is True
    assert data["loop_stop_reason"] == "max_iterations_reached"


def test_run_loop_trace_step_sequence(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert [t["step_kind"] for t in data["step_traces"]] == ["plan", "do", "check"]


def test_run_loop_trace_do_step_misreader_skill_fired(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    do_step = data["step_traces"][1]
    assert do_step["step_kind"] == "do"
    assert do_step["misreader_skill_fired"] is True


def test_run_loop_trace_check_step_checker_ran(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    check_step = data["step_traces"][2]
    assert check_step["step_kind"] == "check"
    assert check_step["checker_ran"] is True
    assert check_step["checker_observed_bypass"] is True


def test_run_loop_trace_observation_outcome(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert data["observation_outcome"] == "observation_succeeded"


# --- --max-iterations 2 ---


def test_run_loop_max_iterations_2_trace_content(tmp_path: Path) -> None:
    _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path, max_iterations=2)
    run_dir = _run_dir(tmp_path)
    data = json.loads((run_dir / "instruction_typescript_any.loop_trace.json").read_text(encoding="utf-8"))
    assert data["max_iterations"] == 2
    assert data["final_loop_iteration"] == 2
    assert [t["step_kind"] for t in data["step_traces"]] == [
        "plan", "do", "check",
        "plan", "do", "check",
    ]


# --- Summary output ---


def test_run_loop_prints_output_path(tmp_path: Path) -> None:
    result = _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    run_dir = _run_dir(tmp_path)
    assert f"Wrote outputs to {run_dir}" in result.stdout


def test_run_loop_prints_loop_trace_count(tmp_path: Path) -> None:
    result = _run_loop_cli("scenarios/instruction_typescript_any.yaml", tmp_path)
    assert "Wrote 1 loop trace file." in result.stdout


# --- Existing commands unchanged ---


def test_fixed_run_still_does_not_emit_loop_trace(tmp_path: Path) -> None:
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
    result = subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert result.returncode == 0
    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    assert list(run_dirs[0].glob("*.loop_trace.json")) == []


# --- Invariants ---


def test_no_control_state_in_cli(tmp_path: Path) -> None:
    source = (ROOT / "src" / "whose_agent" / "cli.py").read_text(encoding="utf-8")
    assert "ControlState(" not in source


def test_no_models_module_exists() -> None:
    assert not (ROOT / "src" / "whose_agent" / "models.py").exists()


def test_no_boundary_transitions_import_in_cli() -> None:
    source = (ROOT / "src" / "whose_agent" / "cli.py").read_text(encoding="utf-8")
    assert "boundary_state.transitions" not in source
