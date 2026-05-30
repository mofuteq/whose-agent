from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import single_run_dir
from whose_agent.scenario_loader import load_scenario


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_scenario_run_writes_outputs_inside_one_run_directory(tmp_path: Path) -> None:
    completed = run_fixed_cli(tmp_path)
    run_dir = single_run_dir(tmp_path)

    assert f"Wrote outputs to {run_dir}" in completed.stdout
    assert (
        "Wrote 7 classification files, 5 response files, 5 trace files, "
        "5 state trace files, and 1 checker file."
    ) in completed.stdout
    assert len(list(run_dir.glob("*.classification.json"))) == 7
    assert len(list(run_dir.glob("*.response.md"))) == 5
    assert len([f for f in run_dir.glob("*.trace.json") if not f.name.endswith(".state_trace.json")]) == 5
    assert len(list(run_dir.glob("*.state_trace.json"))) == 5
    assert len(list(run_dir.glob("*.checker.json"))) == 1
    assert list(run_dir.glob("*.flow.mmd")) == []


def test_typescript_any_mock_run_emits_scenario_artifacts(tmp_path: Path) -> None:
    run_fixed_cli(tmp_path)
    run_dir = single_run_dir(tmp_path)
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    assert scenario.trace_template is not None

    classification_path = run_dir / "instruction_typescript_any.classification.json"
    response_path = run_dir / "instruction_typescript_any.response.md"
    trace_path = run_dir / "instruction_typescript_any.trace.json"
    state_trace_path = run_dir / "instruction_typescript_any.state_trace.json"
    checker_path = run_dir / "instruction_typescript_any.checker.json"

    assert classification_path.exists()
    assert response_path.exists()
    assert trace_path.exists()
    assert state_trace_path.exists()
    assert checker_path.exists()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["scenario_id"] == "instruction_typescript_any"
    assert trace["substituted"] == "instruction"
    assert trace["failure_mode"] == "constraint_override"
    assert trace["divergence_point"] == scenario.trace_template.divergence_point
    assert "Rust" not in trace["divergence_point"]
    assert "rust" not in trace["divergence_point"]

    response = response_path.read_text(encoding="utf-8")
    assert "```typescript" in response
    assert "any" in response

    checker = json.loads(checker_path.read_text(encoding="utf-8"))
    assert checker["scenario_id"] == "instruction_typescript_any"
    assert checker["skill_id"] == "safety_framework_escape_hatch"
    assert checker["checker_observed_bypass"] is True
    assert checker["substituted"] == "instruction"
    assert checker["failure_mode"] == "constraint_override"
    assert checker["confidence"] == "high"
    evidence_text = " ".join(checker["evidence"])
    assert "TypeScript surface" in evidence_text
    assert "type-safety guarantee" in evidence_text
    assert "Rust" not in json.dumps(checker)
    assert "rust" not in json.dumps(checker)


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
