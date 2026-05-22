from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_scenario_run_writes_outputs_inside_one_run_directory(tmp_path: Path) -> None:
    completed = run_fixed_cli(tmp_path)
    run_dir = single_run_dir(tmp_path)

    assert f"Wrote outputs to {run_dir}" in completed.stdout
    assert "Wrote 6 classification files, 4 response files, and 4 trace files." in completed.stdout
    assert len(list(run_dir.glob("*.classification.json"))) == 6
    assert len(list(run_dir.glob("*.response.md"))) == 4
    assert len(list(run_dir.glob("*.trace.json"))) == 4
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


def single_run_dir(outputs: Path) -> Path:
    entries = list(outputs.iterdir())
    run_dirs = [entry for entry in entries if entry.is_dir()]
    assert len(run_dirs) == 1
    assert entries == run_dirs
    assert re.fullmatch(r"\d{8}T\d{6}Z(?:-\d{3})?", run_dirs[0].name)
    return run_dirs[0]
