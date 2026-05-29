from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import single_run_dir


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_scenario_run_writes_outputs_inside_one_run_directory(tmp_path: Path) -> None:
    completed = run_fixed_cli(tmp_path)
    run_dir = single_run_dir(tmp_path)

    assert f"Wrote outputs to {run_dir}" in completed.stdout
    assert "Wrote 6 classification files, 4 response files, 4 trace files, and 4 state trace files." in completed.stdout
    assert len(list(run_dir.glob("*.classification.json"))) == 6
    assert len(list(run_dir.glob("*.response.md"))) == 4
    assert len([f for f in run_dir.glob("*.trace.json") if not f.name.endswith(".state_trace.json")]) == 4
    assert len(list(run_dir.glob("*.state_trace.json"))) == 4
    assert list(run_dir.glob("*.flow.mmd")) == []


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
