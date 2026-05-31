from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.helpers import single_run_dir


ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_SUFFIXES = [
    ".prompt_contract.json",
    ".loop_trace.json",
    ".classification.json",
    ".response.md",
    ".state_trace.json",
    ".trace.json",
    ".checker_comparison.json",
    ".checker.json",
    ".flow.mmd",
]


def test_cli_commands_have_exact_artifact_ownership(tmp_path: Path) -> None:
    fixed_dir = single_run_dir(run_fixed_cli(tmp_path / "fixed"))
    prompt_dir = single_run_dir(run_prompt_cli(tmp_path / "prompt"))
    contract_dir = single_run_dir(run_detect_contract_cli(tmp_path / "contract"))
    loop_dir = single_run_dir(run_loop_cli(tmp_path / "loop"))
    prompt_loop_dir = single_run_dir(run_prompt_loop_cli(tmp_path / "prompt-loop"))

    assert_suffix_counts(
        fixed_dir,
        {
            ".classification.json": 8,
            ".response.md": 6,
            ".trace.json": 6,
            ".state_trace.json": 6,
            ".checker.json": 2,
            ".checker_comparison.json": 2,
        },
    )
    assert_suffix_counts(
        prompt_dir,
        {
            ".classification.json": 1,
            ".flow.mmd": 1,
        },
    )
    assert_suffix_counts(contract_dir, {".prompt_contract.json": 1})
    assert_suffix_counts(loop_dir, {".loop_trace.json": 1})
    assert_suffix_counts(
        prompt_loop_dir,
        {
            ".prompt_contract.json": 1,
            ".loop_trace.json": 1,
        },
    )


def assert_suffix_counts(run_dir: Path, expected_counts: dict[str, int]) -> None:
    files = [path for path in run_dir.iterdir() if path.is_file()]
    unknown_files = [
        path.name
        for path in files
        if not any(path.name.endswith(suffix) for suffix in ARTIFACT_SUFFIXES)
    ]
    assert unknown_files == []

    for suffix in ARTIFACT_SUFFIXES:
        assert len(list(run_dir.glob(f"*{suffix}"))) == expected_counts.get(suffix, 0)


def run_fixed_cli(outputs: Path) -> Path:
    run_cli(
        [
            "run",
            "--scenarios",
            "scenarios",
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )
    return outputs


def run_prompt_cli(outputs: Path) -> Path:
    run_cli(
        [
            "run-prompt",
            "--prompt",
            "Implement a CLI in Rust that counts lines in a file.",
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )
    return outputs


def run_detect_contract_cli(outputs: Path) -> Path:
    run_cli(
        [
            "detect-contract",
            "--prompt",
            "Use TypeScript with explicit models and avoid any",
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )
    return outputs


def run_loop_cli(outputs: Path) -> Path:
    run_cli(
        [
            "run-loop",
            "--scenario",
            "scenarios/instruction_typescript_any.yaml",
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )
    return outputs


def run_prompt_loop_cli(outputs: Path) -> Path:
    run_cli(
        [
            "run-prompt-loop",
            "--prompt",
            "Use TypeScript with explicit models and avoid any",
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )
    return outputs


def run_cli(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-m", "whose_agent.cli", *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
