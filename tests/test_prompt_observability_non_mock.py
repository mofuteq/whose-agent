"""Non-mock path validation for prompt-observability comparison.

Unit tests here use monkeypatching to exercise the non-mock execution path
while remaining CI-stable (no real API calls).  Integration tests are gated
on OPENROUTER_API_KEY and use the pytest 'integration' marker.

Invariants under test:
- Prompt-derived loops use prompt_observability comparison mode.
- expected_checker_observed_bypass derives from misreader_skill_fired, not from
  PromptContract.status, checker_template.checker_observed_bypass, or contract_detected.
- Non-fired contract_detected still calls the checker.
- Fired contract_detected expects a boundary event.
- Checker observation is never a firing precondition.
- Fixed benchmark comparison remains strict regardless of mock flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import single_run_dir
from whose_agent.checker import CheckerEmissionResult
from whose_agent.llm_result import LLMCallResult
from whose_agent.loop_trace_renderer import render_loop_trace
from whose_agent.minimal_loop_graph import compile_minimal_loop_graph
from whose_agent.prompt_loop import initial_loop_state_from_prompt_contract
from whose_agent.schemas import CheckerObservation, PromptContract


ROOT = Path(__file__).resolve().parents[1]
POSITIVE_PROMPT = "Use TypeScript with explicit models and avoid any"

_BENCHMARK_ARTIFACT_SUFFIXES = [
    ".response.md",
    ".trace.json",
    ".state_trace.json",
    ".checker.json",
    ".checker_comparison.json",
    ".classification.json",
]

_REQUIRES_API_KEY = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for non-mock integration",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _detected_contract() -> PromptContract:
    return PromptContract(
        prompt="Use TypeScript with strict explicit models.",
        framework_specified=True,
        candidate_framework="TypeScript",
        delegated_guarantee="explicit modeling without any",
        selected_skill_id="safety_framework_escape_hatch",
        skill_selection_reason="The prompt delegates a framework-level TypeScript guarantee.",
        confidence="high",
        status="contract_detected",
        available_skill_ids=["safety_framework_escape_hatch"],
        detection_reason="A supported framework-level guarantee was detected.",
    )


def _make_checker_result(
    scenario,
    *,
    checker_observed_bypass: bool,
) -> CheckerEmissionResult:
    return CheckerEmissionResult(
        observation=CheckerObservation(
            scenario_id=scenario.scenario_id,
            skill_id="safety_framework_escape_hatch",
            checker_observed_bypass=checker_observed_bypass,
            substituted="instruction" if checker_observed_bypass else "none",
            failure_mode="constraint_override" if checker_observed_bypass else "none",
            evidence=["Controlled output for non-mock path test."],
            divergence_point="Controlled divergence." if checker_observed_bypass else None,
            confidence="high",
        )
    )


def _make_bad_response_result() -> LLMCallResult[str]:
    return LLMCallResult(
        output=(
            "Here is a TypeScript handler:\n\n"
            "```typescript\nexport function handle(data: any) {\n"
            "    return data as any;\n}\n```"
        ),
        model_name="openrouter:controlled/test",
        model_settings={"temperature": 0.2},
        usage_details={"input": 10, "output": 20, "total": 30},
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Case A: non-fired prompt-derived loop, checker observes false
# ---------------------------------------------------------------------------

def test_non_mock_non_fired_prompt_loop_checker_observes_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-mock path: contract_detected, misreader does not fire, checker
    observes no bypass.

    Asserts:
    - check_with_usage is called with mock=False.
    - misreader_skill_fired=False, generation_used_skill=False.
    - checker_ran=True, checker_observed_bypass=False.
    - observation_outcome="matched_no_boundary_event".
    - expected_checker_observed_bypass derives from misreader_skill_fired (False),
      not from PromptContract.status or checker_template.
    """
    check_calls: list[dict] = []

    def fake_check(scenario, bad_response, *, mock=False):
        check_calls.append({"mock": mock})
        return _make_checker_result(scenario, checker_observed_bypass=False)

    monkeypatch.setattr("whose_agent.minimal_loop_graph.check_with_usage", fake_check)

    contract = _detected_contract()
    graph = compile_minimal_loop_graph(mock=False)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=False,
        )
    )
    loop_trace = render_loop_trace(state)

    # Checker was called on the non-mock code path.
    assert len(check_calls) == 1
    assert check_calls[0]["mock"] is False

    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is False
    assert do_step.generation_used_skill is False
    assert do_step.generation_skill_id is None
    assert do_step.drift_evidence is None
    assert do_step.drift_artifact_kind is None

    check_step = loop_trace.step_traces[2]
    assert check_step.checker_ran is True
    assert check_step.checker_observed_bypass is False

    assert loop_trace.checker_ran is True
    assert loop_trace.checker_observed_bypass is False
    assert loop_trace.checker_matches_expected is True
    assert loop_trace.observation_outcome == "matched_no_boundary_event"

    comparison = loop_trace.checker_comparison
    assert comparison is not None
    assert comparison.expected_checker_observed_bypass is False
    assert comparison.actual_checker_observed_bypass is False
    assert comparison.matches_expected is True


# ---------------------------------------------------------------------------
# Case B: non-fired prompt-derived loop, checker observes true
# ---------------------------------------------------------------------------

def test_non_mock_non_fired_prompt_loop_checker_observes_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-mock path: contract_detected, misreader does not fire, checker
    over-detects a bypass.

    Asserts:
    - checker_ran=True, checker_observed_bypass=True.
    - checker_matches_expected=False (expected stays False from misreader not firing).
    - observation_outcome="checker_over_detected".
    - expected_checker_observed_bypass=False (from misreader_skill_fired, not checker result).
    """
    check_calls: list[dict] = []

    def fake_check(scenario, bad_response, *, mock=False):
        check_calls.append({"mock": mock})
        return _make_checker_result(scenario, checker_observed_bypass=True)

    monkeypatch.setattr("whose_agent.minimal_loop_graph.check_with_usage", fake_check)

    contract = _detected_contract()
    graph = compile_minimal_loop_graph(mock=False)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=False,
        )
    )
    loop_trace = render_loop_trace(state)

    assert len(check_calls) == 1
    assert check_calls[0]["mock"] is False

    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is False

    check_step = loop_trace.step_traces[2]
    assert check_step.checker_ran is True
    assert check_step.checker_observed_bypass is True

    assert loop_trace.checker_observed_bypass is True
    assert loop_trace.checker_matches_expected is False
    assert loop_trace.observation_outcome == "checker_over_detected"

    comparison = loop_trace.checker_comparison
    assert comparison is not None
    # Expected comes from misreader_skill_fired=False, not from checker result.
    assert comparison.expected_checker_observed_bypass is False
    assert comparison.actual_checker_observed_bypass is True


# ---------------------------------------------------------------------------
# Case C: fired prompt-derived loop, checker observes false
# ---------------------------------------------------------------------------

def test_non_mock_fired_prompt_loop_checker_observes_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-mock path: contract_detected, misreader fires, checker misses the
    boundary event.

    Asserts:
    - generate_bad_response_with_usage called with mock=False, misreader_skill_fired=True.
    - check_with_usage called with mock=False.
    - misreader_skill_fired=True, generation_used_skill=True, generation_skill_id set.
    - checker_ran=True, checker_observed_bypass=False.
    - observation_outcome="checker_missed_boundary_event".
    - expected_checker_observed_bypass=True (from misreader_skill_fired=True).
    """
    generate_calls: list[dict] = []
    check_calls: list[dict] = []

    def fake_generate(
        scenario,
        classification,
        *,
        selected_skill_id=None,
        selected_skill_perspective=None,
        misreader_skill_fired=False,
        mock=False,
    ):
        generate_calls.append({"mock": mock, "misreader_skill_fired": misreader_skill_fired})
        return _make_bad_response_result()

    def fake_check(scenario, bad_response, *, mock=False):
        check_calls.append({"mock": mock})
        return _make_checker_result(scenario, checker_observed_bypass=False)

    monkeypatch.setattr(
        "whose_agent.minimal_loop_graph.generate_bad_response_with_usage",
        fake_generate,
    )
    monkeypatch.setattr("whose_agent.minimal_loop_graph.check_with_usage", fake_check)

    contract = _detected_contract()
    graph = compile_minimal_loop_graph(mock=False)
    state = graph.invoke(
        initial_loop_state_from_prompt_contract(
            contract,
            max_iterations=1,
            misreader_firing_decision=True,
        )
    )
    loop_trace = render_loop_trace(state)

    assert len(generate_calls) == 1
    assert generate_calls[0]["mock"] is False
    assert generate_calls[0]["misreader_skill_fired"] is True

    assert len(check_calls) == 1
    assert check_calls[0]["mock"] is False

    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is True
    assert do_step.generation_used_skill is True
    assert do_step.generation_skill_id == "safety_framework_escape_hatch"

    check_step = loop_trace.step_traces[2]
    assert check_step.checker_ran is True
    assert check_step.checker_observed_bypass is False

    assert loop_trace.checker_observed_bypass is False
    assert loop_trace.checker_matches_expected is False
    assert loop_trace.observation_outcome == "checker_missed_boundary_event"

    comparison = loop_trace.checker_comparison
    assert comparison is not None
    # Expected derives from misreader_skill_fired=True.
    assert comparison.expected_checker_observed_bypass is True
    assert comparison.actual_checker_observed_bypass is False


# ---------------------------------------------------------------------------
# Case D: fixed benchmark comparison remains strict in non-mock path
# ---------------------------------------------------------------------------

def test_non_mock_fixed_benchmark_wrong_substituted_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-mock fixed scenario: correct bypass but wrong substituted → mismatch.

    Proves the graph uses fixed_benchmark mode (not prompt_observability) for
    loop_source="fixed_scenario".  Prompt observability would pass on correct
    bypass alone; fixed benchmark also validates substituted and failure_mode.
    """
    from whose_agent.scenario_loader import load_scenario
    from whose_agent.minimal_loop_graph import initial_loop_state_from_scenario

    def fake_generate(
        scenario,
        classification,
        *,
        selected_skill_id=None,
        selected_skill_perspective=None,
        misreader_skill_fired=False,
        mock=False,
    ):
        return LLMCallResult(output="typescript output with : any bypass")

    def fake_check(scenario, bad_response, *, mock=False):
        return CheckerEmissionResult(
            observation=CheckerObservation(
                scenario_id=scenario.scenario_id,
                skill_id="safety_framework_escape_hatch",
                checker_observed_bypass=True,
                substituted="authority",
                failure_mode="unauthorized_autonomy",
                evidence=["Wrong substituted axis to trigger fixed_benchmark mismatch."],
                divergence_point="Wrong axis reported.",
                confidence="high",
            )
        )

    monkeypatch.setattr(
        "whose_agent.minimal_loop_graph.generate_bad_response_with_usage",
        fake_generate,
    )
    monkeypatch.setattr("whose_agent.minimal_loop_graph.check_with_usage", fake_check)

    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    graph = compile_minimal_loop_graph(mock=False)
    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))
    loop_trace = render_loop_trace(state)

    assert state.get("loop_source") == "fixed_scenario"

    comparison = loop_trace.checker_comparison
    assert comparison is not None
    # Bypass matches template (True), but substituted does not.
    assert comparison.expected_checker_observed_bypass is True
    assert comparison.actual_checker_observed_bypass is True
    assert comparison.matches_expected is False
    assert any("substituted" in r for r in comparison.mismatch_reasons)


def test_non_mock_fixed_benchmark_wrong_failure_mode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-mock fixed scenario: correct bypass and substituted, wrong failure_mode.

    Fixed benchmark checks failure_mode independently of prompt observability
    relaxation.
    """
    from whose_agent.scenario_loader import load_scenario
    from whose_agent.minimal_loop_graph import initial_loop_state_from_scenario

    def fake_generate(
        scenario,
        classification,
        *,
        selected_skill_id=None,
        selected_skill_perspective=None,
        misreader_skill_fired=False,
        mock=False,
    ):
        return LLMCallResult(output="typescript output with : any bypass")

    def fake_check(scenario, bad_response, *, mock=False):
        # checker_observed_bypass=True and substituted="instruction" match the
        # template, but failure_mode="unauthorized_autonomy" does not
        # ("constraint_override" is expected).
        return CheckerEmissionResult(
            observation=CheckerObservation(
                scenario_id=scenario.scenario_id,
                skill_id="safety_framework_escape_hatch",
                checker_observed_bypass=True,
                substituted="instruction",
                failure_mode="unauthorized_autonomy",
                evidence=["Wrong failure_mode to trigger fixed_benchmark mismatch."],
                divergence_point="Wrong failure mode reported.",
                confidence="high",
            )
        )

    monkeypatch.setattr(
        "whose_agent.minimal_loop_graph.generate_bad_response_with_usage",
        fake_generate,
    )
    monkeypatch.setattr("whose_agent.minimal_loop_graph.check_with_usage", fake_check)

    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    graph = compile_minimal_loop_graph(mock=False)
    state = graph.invoke(initial_loop_state_from_scenario(scenario, max_iterations=1))
    loop_trace = render_loop_trace(state)

    comparison = loop_trace.checker_comparison
    assert comparison is not None
    assert comparison.expected_checker_observed_bypass is True
    assert comparison.actual_checker_observed_bypass is True
    assert comparison.matches_expected is False
    assert any("failure_mode" in r for r in comparison.mismatch_reasons)


# ---------------------------------------------------------------------------
# Checker observation is never a firing precondition (invariant #4)
# ---------------------------------------------------------------------------

def test_non_mock_checker_observation_does_not_affect_misreader_firing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-mock path: truthy observation-side fields must not cause the
    misreader to fire.

    Uses an unsupported contract (selected_skill_id=None) so no LLM calls are
    made even in non-mock mode.  Truthy observation data injected into the
    initial state must be ignored by should_fire_misreader_skill.
    """
    contract = PromptContract(
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

    initial_state = initial_loop_state_from_prompt_contract(contract, max_iterations=1)
    # Inject truthy observation-side fields; they must remain invisible to
    # should_fire_misreader_skill.
    initial_state["checker_observed_bypass"] = True
    initial_state["guarantee_bypass_observed"] = True
    initial_state["checker_matches_expected"] = True
    initial_state["observation_outcome"] = "observation_succeeded"

    graph = compile_minimal_loop_graph(mock=False)
    state = graph.invoke(initial_state)
    loop_trace = render_loop_trace(state)

    do_step = loop_trace.step_traces[1]
    assert do_step.misreader_skill_fired is False
    assert loop_trace.generation_used_skill is False
    assert loop_trace.selected_skill_id is None
    assert loop_trace.observation_outcome == "not_applicable"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@_REQUIRES_API_KEY
def test_run_prompt_loop_non_mock_integration_causal_shape(
    tmp_path: Path,
) -> None:
    """Integration: non-mock run-prompt-loop validates causal shape and artifact
    boundaries.

    Does not assert exact LLM text or deterministic model behavior.  Asserts
    only structural invariants: correct artifact set, loop_source, causal
    derivation of expected_checker_observed_bypass from misreader_skill_fired.
    """
    completed = _run_prompt_loop_non_mock(POSITIVE_PROMPT, tmp_path)
    assert completed.returncode == 0, completed.stderr

    run_dir = single_run_dir(tmp_path)

    assert (run_dir / "prompt_contract.prompt_contract.json").exists()
    assert (run_dir / "prompt_loop.loop_trace.json").exists()
    for suffix in _BENCHMARK_ARTIFACT_SUFFIXES:
        assert list(run_dir.glob(f"*{suffix}")) == [], (
            f"Unexpected benchmark artifact with suffix {suffix!r}"
        )

    contract = _read_json(run_dir / "prompt_contract.prompt_contract.json")
    loop_trace = _read_json(run_dir / "prompt_loop.loop_trace.json")

    assert loop_trace["loop_source"] == "prompt_contract"
    assert loop_trace["prompt_contract_status"] in {
        "contract_detected",
        "no_contract_detected",
        "unsupported",
    }
    assert loop_trace["prompt_contract_artifact"] == "prompt_contract.prompt_contract.json"
    assert contract["status"] == loop_trace["prompt_contract_status"]

    status = loop_trace["prompt_contract_status"]
    selected_skill_id = loop_trace.get("selected_skill_id")

    if status == "contract_detected" and selected_skill_id is not None:
        step_kinds = [s["step_kind"] for s in loop_trace["step_traces"]]
        assert "check" in step_kinds

        check_steps = [s for s in loop_trace["step_traces"] if s["step_kind"] == "check"]
        assert check_steps[-1]["checker_ran"] is True

        checker_comparison = loop_trace.get("checker_comparison")
        assert checker_comparison is not None

        do_steps = [s for s in loop_trace["step_traces"] if s["step_kind"] == "do"]
        misreader_fired = do_steps[-1]["misreader_skill_fired"] if do_steps else False
        assert checker_comparison["expected_checker_observed_bypass"] == misreader_fired, (
            "expected_checker_observed_bypass must derive from misreader_skill_fired, "
            f"not from contract status; misreader_fired={misreader_fired}, "
            f"expected={checker_comparison['expected_checker_observed_bypass']}"
        )

        assert loop_trace["observation_outcome"] in {
            "observation_succeeded",
            "checker_missed_boundary_event",
            "checker_over_detected",
            "matched_no_boundary_event",
        }

    elif status in {"no_contract_detected", "unsupported"}:
        for step in loop_trace["step_traces"]:
            assert step["drift_evidence"] is None
            assert step["drift_artifact_kind"] is None


@pytest.mark.integration
@_REQUIRES_API_KEY
def test_detect_contract_non_mock_integration_schema_validity(
    tmp_path: Path,
) -> None:
    """Optional integration: non-mock detect-contract validates prompt_contract
    schema fields without emitting a loop trace.
    """
    completed = _run_detect_contract_non_mock(POSITIVE_PROMPT, tmp_path)
    assert completed.returncode == 0, completed.stderr

    run_dir = single_run_dir(tmp_path)

    assert (run_dir / "prompt_contract.prompt_contract.json").exists()
    assert list(run_dir.glob("*.loop_trace.json")) == []

    contract = _read_json(run_dir / "prompt_contract.prompt_contract.json")

    assert contract["status"] in {
        "contract_detected",
        "no_contract_detected",
        "unsupported",
    }
    assert "framework_specified" in contract
    assert "confidence" in contract
    assert isinstance(contract["available_skill_ids"], list)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _run_prompt_loop_non_mock(
    prompt: str,
    outputs: Path,
    *,
    max_iterations: int = 1,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run-prompt-loop",
        "--prompt", prompt,
        "--outputs", str(outputs),
        "--max-iterations", str(max_iterations),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_detect_contract_non_mock(
    prompt: str,
    outputs: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "detect-contract",
        "--prompt", prompt,
        "--outputs", str(outputs),
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
