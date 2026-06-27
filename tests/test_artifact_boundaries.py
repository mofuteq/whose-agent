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
    ".generated.md",
]


def test_cli_commands_have_exact_artifact_ownership(tmp_path: Path) -> None:
    fixed_dir = single_run_dir(run_fixed_cli(tmp_path / "fixed"))
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
            ".checker.json": 6,
            ".checker_comparison.json": 6,
        },
    )
    assert_suffix_counts(contract_dir, {".prompt_contract.json": 1})
    assert_suffix_counts(loop_dir, {".loop_trace.json": 1})
    assert_suffix_counts(
        prompt_loop_dir,
        {
            ".prompt_contract.json": 1,
            ".loop_trace.json": 1,
            ".generated.md": 1,
        },
    )


def test_run_prompt_is_not_available_cli_command(tmp_path: Path) -> None:
    from whose_agent.cli import build_parser

    parser = build_parser()
    subparser_actions = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices is not None
    ]
    commands: set[str] = set()
    for action in subparser_actions:
        commands.update(action.choices.keys())

    assert "run-prompt" not in commands

    completed = run_cli(
        [
            "run-prompt",
            "--prompt",
            "Implement a CLI in Rust that counts lines in a file.",
            "--outputs",
            str(tmp_path),
            "--mock",
        ],
        check=False,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr


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


def run_cli(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "whose_agent.cli", *args],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )
