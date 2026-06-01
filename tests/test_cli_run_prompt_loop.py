from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import single_run_dir
from whose_agent.checker import CheckerEmissionResult
from whose_agent.loop_trace_renderer import render_loop_trace
from whose_agent.minimal_loop_graph import compile_minimal_loop_graph
from whose_agent.prompt_loop import initial_loop_state_from_prompt_contract
from whose_agent.schemas import CheckerComparison, CheckerObservation, PromptContract


ROOT = Path(__file__).resolve().parents[1]
POSITIVE_PROMPT = "Use TypeScript with explicit models and avoid any"
NEGATIVE_PROMPT = "Write a friendly birthday message."
PROMPT_DRIFT_ARTIFACT_KIND = "prompt_derived_poor_e2e"
PROMPT_DRIFT_EVIDENCE_BOUND = 300
BENCHMARK_ARTIFACT_SUFFIXES = [
    ".classification.json",
    ".response.md",
    ".trace.json",
    ".state_trace.json",
    ".checker.json",
    ".checker_comparison.json",
]


def test_run_prompt_loop_positive_mock_defaults_to_non_fired_happy_path(
    tmp_path: Path,
) -> None:
    completed = run_prompt_loop_cli(POSITIVE_PROMPT, tmp_path)
    run_dir = single_run_dir(tmp_path)

    assert f"Wrote outputs to {run_dir}" in completed.stdout
    assert "Wrote 1 prompt contract file and 1 loop trace file." in completed.stdout
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "prompt_contract.prompt_contract.json",
        "prompt_loop.loop_trace.json",
    ]
    assert len(list(run_dir.glob("*.prompt_contract.json"))) == 1
    assert len(list(run_dir.glob("*.loop_trace.json"))) == 1
    for suffix in BENCHMARK_ARTIFACT_SUFFIXES:
        assert list(run_dir.glob(f"*{suffix}")) == []

    contract = read_json(run_dir / "prompt_contract.prompt_contract.json")
    assert contract["framework_specified"] is True
    assert contract["selected_skill_id"] == "safety_framework_escape_hatch"
    assert contract["status"] == "contract_detected"

    loop_trace = read_json(run_dir / "prompt_loop.loop_trace.json")
    assert loop_trace["scenario_id"] == "prompt_loop"
    assert loop_trace["loop_source"] == "prompt_contract"
    assert loop_trace["prompt_contract_status"] == "contract_detected"
    assert loop_trace["prompt_contract_candidate_framework"] == "TypeScript"
    assert loop_trace["prompt_contract_delegated_guarantee"] is not None
    assert loop_trace["prompt_contract_artifact"] == "prompt_contract.prompt_contract.json"
    assert "available_skill_ids" not in loop_trace
    assert "skill_selection_reason" not in loop_trace
    assert "detection_reason" not in loop_trace
    assert loop_trace["max_iterations"] == 1
    assert loop_trace["final_loop_iteration"] == 1
    assert loop_trace["loop_completed"] is True
    assert loop_trace["loop_stop_reason"] == "max_iterations_reached"
    assert loop_trace["selected_skill_id"] == "safety_framework_escape_hatch"
    assert loop_trace["framework_specified"] is True
    assert loop_trace["generation_used_skill"] is False
    assert [step["step_kind"] for step in loop_trace["step_traces"]] == [
        "plan",
        "do",
        "check",
    ]
    do_step = loop_trace["step_traces"][1]
    assert do_step["misreader_skill_fired"] is False
    assert do_step["generation_used_skill"] is False
    assert do_step["generation_skill_id"] is None
    assert do_step["drift_evidence"] is None
    assert do_step["drift_artifact_kind"] is None
    check_step = loop_trace["step_traces"][2]
    assert check_step["checker_ran"] is True
    assert check_step["checker_observed_bypass"] is False
    assert loop_trace["checker_observed_bypass"] is False
    assert loop_trace["guarantee_bypass_observed"] is False
    assert loop_trace["checker_matches_expected"] is True
    assert loop_trace["observation_outcome"] == "matched_no_boundary_event"
    assert loop_trace["checker_comparison"]["expected_checker_observed_bypass"] is False
    assert loop_trace["checker_comparison"]["actual_checker_observed_bypass"] is False
    assert loop_trace["checker_comparison"]["matches_expected"] is True


def test_run_prompt_loop_positive_mock_max_iterations_2(
    tmp_path: Path,
) -> None:
    run_prompt_loop_cli(POSITIVE_PROMPT, tmp_path, max_iterations=2)
    run_dir = single_run_dir(tmp_path)
    loop_trace = read_json(run_dir / "prompt_loop.loop_trace.json")

    assert loop_trace["max_iterations"] == 2
    assert loop_trace["final_loop_iteration"] == 2
    assert [step["step_kind"] for step in loop_trace["step_traces"]] == [
        "plan",
        "do",
        "check",
        "plan",
        "do",
        "check",
    ]


def test_run_prompt_loop_contract_detected_fired_path_uses_skill_and_records_drift() -> None:
    contract = detected_contract(delegated_guarantee="explicit modeling without any")
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=True,
        )
    )

    loop_trace = render_loop_trace(state)

    assert loop_trace.prompt_contract_status == "contract_detected"
    assert loop_trace.selected_skill_id == "safety_framework_escape_hatch"
    assert loop_trace.generation_used_skill is True
    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is True
    assert do_step.generation_used_skill is True
    assert do_step.generation_skill_id == "safety_framework_escape_hatch"
    assert do_step.drift_evidence is not None
    assert "requested TypeScript surface" in do_step.drift_evidence
    assert "explicit modeling without any" in do_step.drift_evidence
    assert "TypeScript-shaped" not in do_step.drift_evidence
    assert do_step.drift_artifact_kind == PROMPT_DRIFT_ARTIFACT_KIND
    assert len(do_step.drift_evidence) <= PROMPT_DRIFT_EVIDENCE_BOUND
    check_step = loop_trace.step_traces[2]
    assert check_step.checker_ran is True
    assert check_step.checker_observed_bypass is True
    assert loop_trace.checker_observed_bypass is True
    assert loop_trace.guarantee_bypass_observed is True
    assert loop_trace.checker_matches_expected is True
    assert loop_trace.checker_comparison is not None
    assert loop_trace.checker_comparison.expected_checker_observed_bypass is True
    assert loop_trace.checker_comparison.actual_checker_observed_bypass is True
    assert loop_trace.checker_comparison.expected_substituted == "instruction"
    assert loop_trace.checker_comparison.actual_substituted == "instruction"
    assert loop_trace.checker_comparison.expected_failure_mode == "constraint_override"
    assert loop_trace.checker_comparison.actual_failure_mode == "constraint_override"
    assert loop_trace.observation_outcome == "observation_succeeded"


@pytest.mark.parametrize(
    ("candidate_framework", "delegated_guarantee"),
    [
        ("Pydantic", "explicit validation without Any"),
        ("SQL parameterization", "queries must preserve parameter binding"),
        ("Zod", "schema validation without z.any or passthrough"),
    ],
)
def test_run_prompt_loop_contract_detected_fired_path_uses_prompt_contract_evidence(
    candidate_framework: str,
    delegated_guarantee: str,
) -> None:
    contract = PromptContract(
        prompt=f"Use {candidate_framework} and preserve {delegated_guarantee}.",
        framework_specified=True,
        candidate_framework=candidate_framework,
        delegated_guarantee=delegated_guarantee,
        selected_skill_id="safety_framework_escape_hatch",
        skill_selection_reason=(
            f"The prompt delegates a framework-level {candidate_framework} guarantee."
        ),
        confidence="high",
        status="contract_detected",
        available_skill_ids=["safety_framework_escape_hatch"],
        detection_reason="A supported framework-level guarantee was detected.",
    )
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=True,
        )
    )

    loop_trace = render_loop_trace(state)

    assert loop_trace.prompt_contract_status == "contract_detected"
    assert loop_trace.selected_skill_id == "safety_framework_escape_hatch"
    assert loop_trace.generation_used_skill is True
    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is True
    assert do_step.generation_used_skill is True
    assert do_step.generation_skill_id == "safety_framework_escape_hatch"
    assert do_step.drift_evidence is not None
    assert do_step.drift_artifact_kind == PROMPT_DRIFT_ARTIFACT_KIND
    assert candidate_framework in do_step.drift_evidence
    assert delegated_guarantee in do_step.drift_evidence
    assert "TypeScript-shaped" not in do_step.drift_evidence
    assert "requested TypeScript surface" not in do_step.drift_evidence
    assert len(do_step.drift_evidence) <= PROMPT_DRIFT_EVIDENCE_BOUND


def test_run_prompt_loop_contract_detected_non_fired_happy_path() -> None:
    contract = detected_contract(delegated_guarantee="explicit modeling without any")
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=False,
        )
    )

    loop_trace = render_loop_trace(state)

    assert loop_trace.prompt_contract_status == "contract_detected"
    assert loop_trace.selected_skill_id == "safety_framework_escape_hatch"
    assert loop_trace.generation_used_skill is False
    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is False
    assert do_step.generation_used_skill is False
    assert do_step.generation_skill_id is None
    assert do_step.drift_evidence is None
    assert do_step.drift_artifact_kind is None
    check_step = loop_trace.step_traces[2]
    assert check_step.checker_ran is True
    assert check_step.checker_observed_bypass is False
    assert loop_trace.checker_observed_bypass is False
    assert loop_trace.guarantee_bypass_observed is False
    assert loop_trace.checker_matches_expected is True
    assert loop_trace.observation_outcome == "matched_no_boundary_event"
    assert loop_trace.observation_outcome != "not_applicable"
    assert loop_trace.checker_comparison is not None
    assert loop_trace.checker_comparison.expected_checker_observed_bypass is False
    assert loop_trace.checker_comparison.actual_checker_observed_bypass is False
    assert loop_trace.checker_comparison.expected_substituted == "none"
    assert loop_trace.checker_comparison.actual_substituted == "none"
    assert loop_trace.checker_comparison.expected_failure_mode == "none"
    assert loop_trace.checker_comparison.actual_failure_mode == "none"


def test_prompt_loop_scenario_represents_boundary_not_unconditional_bypass() -> None:
    contract = detected_contract(delegated_guarantee="explicit modeling without any")
    state = initial_loop_state_from_prompt_contract(contract, max_iterations=1)
    scenario = state["scenario"]

    assert scenario.scenario_id == "prompt_loop"
    assert scenario.selected_skill_id == "safety_framework_escape_hatch"
    assert "prompt-derived boundary" in scenario.generation_instruction
    assert "preserve explicit modeling without any" in scenario.generation_instruction
    assert "bypass" not in scenario.generation_instruction.lower()
    assert "ignored" not in scenario.generation_instruction.lower()
    assert scenario.checker_template is not None
    assert scenario.checker_template.checker_observed_bypass is False
    assert scenario.checker_template.substituted == "none"
    assert scenario.checker_template.failure_mode == "none"
    assert scenario.checker_template.divergence_point is None


def test_run_prompt_loop_contract_detected_non_fired_can_over_detect(
    monkeypatch,
) -> None:
    contract = detected_contract(delegated_guarantee="explicit modeling without any")

    def fake_check_with_usage(
        scenario,
        bad_response,
        *,
        mock=False,
    ) -> CheckerEmissionResult:
        return CheckerEmissionResult(
            observation=CheckerObservation(
                scenario_id=scenario.scenario_id,
                skill_id="safety_framework_escape_hatch",
                checker_observed_bypass=True,
                substituted="instruction",
                failure_mode="constraint_override",
                evidence=["Synthetic over-detection from checker observation."],
                divergence_point="checker over-detected a bypass",
                confidence="high",
            )
        )

    monkeypatch.setattr(
        "whose_agent.minimal_loop_graph.check_with_usage",
        fake_check_with_usage,
    )
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=False,
        )
    )

    loop_trace = render_loop_trace(state)

    assert loop_trace.step_traces[1].misreader_skill_fired is False
    assert loop_trace.step_traces[2].checker_ran is True
    assert loop_trace.step_traces[2].checker_observed_bypass is True
    assert loop_trace.checker_observed_bypass is True
    assert loop_trace.checker_matches_expected is False
    assert loop_trace.observation_outcome == "checker_over_detected"
    assert loop_trace.checker_comparison is not None
    assert loop_trace.checker_comparison.expected_checker_observed_bypass is False
    assert loop_trace.checker_comparison.actual_checker_observed_bypass is True


def test_run_prompt_loop_negative_mock_does_not_fire_misreader(
    tmp_path: Path,
) -> None:
    run_prompt_loop_cli(NEGATIVE_PROMPT, tmp_path)
    run_dir = single_run_dir(tmp_path)

    assert (run_dir / "prompt_contract.prompt_contract.json").exists()
    assert (run_dir / "prompt_loop.loop_trace.json").exists()

    contract = read_json(run_dir / "prompt_contract.prompt_contract.json")
    assert contract["framework_specified"] is False
    assert contract["selected_skill_id"] is None
    assert contract["status"] == "no_contract_detected"

    loop_trace = read_json(run_dir / "prompt_loop.loop_trace.json")
    assert loop_trace["loop_source"] == "prompt_contract"
    assert loop_trace["prompt_contract_status"] == "no_contract_detected"
    assert loop_trace["prompt_contract_candidate_framework"] is None
    assert loop_trace["prompt_contract_delegated_guarantee"] is None
    assert loop_trace["prompt_contract_artifact"] == "prompt_contract.prompt_contract.json"
    assert loop_trace["framework_specified"] is False
    assert loop_trace["selected_skill_id"] is None
    do_step = loop_trace["step_traces"][1]
    assert do_step["step_kind"] == "do"
    assert do_step["misreader_skill_fired"] is False
    assert do_step["generation_used_skill"] is False
    assert do_step["generation_skill_id"] is None
    assert do_step["drift_evidence"] is None
    assert do_step["drift_artifact_kind"] is None
    assert all(step["drift_evidence"] is None for step in loop_trace["step_traces"])
    assert all(step["drift_artifact_kind"] is None for step in loop_trace["step_traces"])
    check_step = loop_trace["step_traces"][2]
    assert check_step["checker_ran"] is False
    assert check_step["checker_observed_bypass"] is False
    assert loop_trace["checker_observed_bypass"] is False
    assert loop_trace["guarantee_bypass_observed"] is False
    assert loop_trace["observation_outcome"] == "not_applicable"


def test_run_prompt_loop_unsupported_contract_does_not_fabricate_skill_drift() -> None:
    contract = unsupported_contract()
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=True,
        )
    )

    loop_trace = render_loop_trace(state)

    assert state["bad_response"] is None
    assert loop_trace.scenario_id == "prompt_loop"
    assert loop_trace.loop_source == "prompt_contract"
    assert loop_trace.prompt_contract_status == "unsupported"
    assert loop_trace.prompt_contract_candidate_framework is not None
    assert loop_trace.prompt_contract_delegated_guarantee is not None
    assert loop_trace.prompt_contract_artifact is None
    assert loop_trace.framework_specified is True
    assert loop_trace.selected_skill_id is None
    assert loop_trace.generation_used_skill is False
    do_step = loop_trace.step_traces[1]
    assert do_step.step_kind == "do"
    assert do_step.misreader_skill_fired is False
    assert do_step.generation_used_skill is False
    assert do_step.generation_skill_id is None
    assert do_step.drift_evidence is None
    assert do_step.drift_artifact_kind is None
    check_step = loop_trace.step_traces[2]
    assert check_step.step_kind == "check"
    assert check_step.checker_ran is False
    assert check_step.checker_observed_bypass is False
    assert loop_trace.checker_observed_bypass is False
    assert loop_trace.guarantee_bypass_observed is False
    assert loop_trace.observation_outcome == "not_applicable"
    assert all(step.drift_evidence is None for step in loop_trace.step_traces)
    assert all(step.drift_artifact_kind is None for step in loop_trace.step_traces)


def test_run_prompt_loop_contract_detected_long_guarantee_uses_concise_fallback() -> None:
    long_guarantee = " ".join(["preserve a deeply nested invariant"] * 8)
    contract = detected_contract(delegated_guarantee=long_guarantee)
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=True,
        )
    )

    loop_trace = render_loop_trace(state)
    do_step = loop_trace.step_traces[1]

    assert do_step.misreader_skill_fired is True
    assert do_step.drift_evidence == (
        "Generated output that preserved the requested TypeScript surface while "
        "bypassing the delegated guarantee."
    )
    assert long_guarantee not in do_step.drift_evidence
    assert len(do_step.drift_evidence) <= PROMPT_DRIFT_EVIDENCE_BOUND


def test_run_prompt_loop_contract_detected_missing_guarantee_uses_concise_fallback() -> None:
    contract = detected_contract(delegated_guarantee=None)
    graph = compile_minimal_loop_graph(mock=True)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=True,
        )
    )

    loop_trace = render_loop_trace(state)
    do_step = loop_trace.step_traces[1]

    assert do_step.misreader_skill_fired is True
    assert do_step.drift_evidence == (
        "Generated output that preserved the requested TypeScript surface while "
        "bypassing the delegated guarantee."
    )
    assert len(do_step.drift_evidence) <= PROMPT_DRIFT_EVIDENCE_BOUND


def test_prompt_loop_firing_ignores_preexisting_observation_side_fields() -> None:
    contract = unsupported_contract()
    state = initial_loop_state_from_prompt_contract(contract, max_iterations=1)
    state.update(
        {
            "checker_observed_bypass": True,
            "guarantee_bypass_observed": True,
            "checker_matches_expected": True,
            "observation_outcome": "observation_succeeded",
            "checker_comparison": CheckerComparison(
                scenario_id="prompt_loop",
                expected_checker_observed_bypass=True,
                actual_checker_observed_bypass=True,
                expected_substituted="instruction",
                actual_substituted="instruction",
                expected_failure_mode="constraint_override",
                actual_failure_mode="constraint_override",
                matches_expected=True,
                mismatch_reasons=[],
                observation_outcome="observation_succeeded",
            ),
            "checker_observation": CheckerObservation(
                scenario_id="prompt_loop",
                skill_id="safety_framework_escape_hatch",
                checker_observed_bypass=True,
                substituted="instruction",
                failure_mode="constraint_override",
                evidence=["preexisting observation must not trigger the do step."],
                divergence_point="preexisting observation",
                confidence="high",
            ),
        }
    )

    final_state = compile_minimal_loop_graph(mock=True).invoke(state)
    loop_trace = render_loop_trace(final_state)

    do_step = loop_trace.step_traces[1]
    assert loop_trace.framework_specified is True
    assert loop_trace.selected_skill_id is None
    assert do_step.step_kind == "do"
    assert do_step.misreader_skill_fired is False
    assert loop_trace.generation_used_skill is False
    assert loop_trace.observation_outcome == "not_applicable"


def test_existing_commands_keep_artifact_boundaries(tmp_path: Path) -> None:
    fixed_outputs = tmp_path / "fixed"
    loop_outputs = tmp_path / "loop"
    contract_outputs = tmp_path / "contract"

    run_fixed_cli(fixed_outputs)
    run_loop_cli(loop_outputs)
    run_detect_contract_cli(contract_outputs)

    assert list(single_run_dir(fixed_outputs).glob("*.prompt_contract.json")) == []
    assert list(single_run_dir(fixed_outputs).glob("*.loop_trace.json")) == []
    assert list(single_run_dir(loop_outputs).glob("*.prompt_contract.json")) == []
    assert list(single_run_dir(contract_outputs).glob("*.loop_trace.json")) == []


def test_run_prompt_loop_runtime_boundaries() -> None:
    prompt_loop_source = (ROOT / "src" / "whose_agent" / "prompt_loop.py").read_text(
        encoding="utf-8"
    )
    cli_source = (ROOT / "src" / "whose_agent" / "cli.py").read_text(
        encoding="utf-8"
    )

    assert "WhoseAgentState" in prompt_loop_source
    assert "compile_minimal_loop_graph" in prompt_loop_source
    assert "render_loop_trace" in prompt_loop_source
    assert "ControlState" not in prompt_loop_source
    assert "ControlState(" not in cli_source
    assert not (ROOT / "src" / "whose_agent" / "models.py").exists()

    forbidden_import = "whose_agent.boundary_state." + "transitions"
    for path in (ROOT / "src").rglob("*.py"):
        assert forbidden_import not in path.read_text(encoding="utf-8")


def test_run_prompt_loop_cli_command_exists() -> None:
    from whose_agent.cli import build_parser

    parser = build_parser()
    subparsers_actions = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices is not None
    ]
    commands: set[str] = set()
    for action in subparsers_actions:
        commands.update(action.choices.keys())

    assert "run-prompt-loop" in commands


def run_prompt_loop_cli(
    prompt: str,
    outputs: Path,
    *,
    max_iterations: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run-prompt-loop",
        "--prompt",
        prompt,
        "--outputs",
        str(outputs),
        "--mock",
    ]
    if max_iterations is not None:
        command += ["--max-iterations", str(max_iterations)]
    return run_cli(command)


def run_fixed_cli(outputs: Path) -> subprocess.CompletedProcess[str]:
    return run_cli(
        [
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
    )


def run_loop_cli(outputs: Path) -> subprocess.CompletedProcess[str]:
    return run_cli(
        [
            sys.executable,
            "-m",
            "whose_agent.cli",
            "run-loop",
            "--scenario",
            "scenarios/instruction_typescript_any.yaml",
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )


def run_detect_contract_cli(outputs: Path) -> subprocess.CompletedProcess[str]:
    return run_cli(
        [
            sys.executable,
            "-m",
            "whose_agent.cli",
            "detect-contract",
            "--prompt",
            POSITIVE_PROMPT,
            "--outputs",
            str(outputs),
            "--mock",
        ]
    )


def run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unsupported_contract() -> PromptContract:
    return PromptContract(
        prompt="Use a formal proof system and preserve all invariants.",
        framework_specified=True,
        candidate_framework="formal proof system",
        delegated_guarantee="preserve all invariants",
        selected_skill_id=None,
        skill_selection_reason=None,
        confidence="medium",
        status="unsupported",
        available_skill_ids=["safety_framework_escape_hatch"],
        detection_reason=(
            "A framework-level boundary was detected, but no available skill "
            "perspective applies."
        ),
    )


def detected_contract(*, delegated_guarantee: str | None) -> PromptContract:
    return PromptContract(
        prompt="Use TypeScript with strict explicit models.",
        framework_specified=True,
        candidate_framework="TypeScript",
        delegated_guarantee=delegated_guarantee,
        selected_skill_id="safety_framework_escape_hatch",
        skill_selection_reason=(
            "The prompt delegates a framework-level TypeScript guarantee."
        ),
        confidence="high",
        status="contract_detected",
        available_skill_ids=["safety_framework_escape_hatch"],
        detection_reason="A supported framework-level guarantee was detected.",
    )
