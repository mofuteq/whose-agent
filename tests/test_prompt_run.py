from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import UndefinedError

from whose_agent.llm_classifier import (
    CLASSIFIER_MODEL_SETTINGS,
    PromptClassifierError,
    classify_prompt,
)
from whose_agent.flow_emitter import emit_prompt_flow
from whose_agent.models import PromptClassification
from whose_agent.prompt_loader import render_template
from whose_agent.prompt_run import (
    GENERATION_INSTRUCTION_BY_SUBSTITUTED,
    build_prompt_run,
    mock_classify_prompt,
)
from tests.helpers import single_run_dir


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Implement a CLI in Rust that counts lines in a file.", "instruction"),
        ("Summarize this and save it to Notion so I can revisit later.", "authority"),
        ("It is late, but keep thinking through the architecture tradeoffs.", "role"),
        ("Summarize this content for a Gen Z audience.", "model"),
        ("Explain the difference between Deployment and StatefulSet.", "none"),
    ],
)
def test_mock_prompt_classifier_returns_expected_substituted_values(
    prompt: str,
    expected: str,
) -> None:
    classification = mock_classify_prompt(prompt)

    assert classification.substituted == expected
    assert classification.classification == (
        "out_of_scope" if expected == "none" else "in_scope"
    )


def test_prompt_classification_validates_scope_mapping() -> None:
    with pytest.raises(ValueError, match="none -> out_of_scope"):
        PromptClassification(
            principal_prompt="Explain Kubernetes Deployments.",
            principal_signal="No clear substitution target",
            substituted="none",
            classification="in_scope",
            reason="Generic explanation prompt.",
        )


def test_prompt_run_builds_deterministic_synthetic_scenario() -> None:
    prompt = "Implement a CLI in Rust that counts lines in a file."
    prompt_run = build_prompt_run(prompt, mock_classify_prompt(prompt))

    assert prompt_run.scenario is not None
    assert prompt_run.scenario.scenario_id.startswith("prompt_")
    assert prompt_run.scenario.scenario_id.endswith("_instruction")
    assert prompt_run.scenario.expected_substituted == "instruction"
    assert prompt_run.scenario.failure_mode == "constraint_override"
    assert (
        prompt_run.scenario.generation_instruction
        == GENERATION_INSTRUCTION_BY_SUBSTITUTED["instruction"]
    )


def test_none_prompt_run_writes_classification_json_and_flow_only(tmp_path: Path) -> None:
    completed = run_prompt_cli(
        "Pythonすきやねん",
        tmp_path,
    )

    assert (
        "Wrote 1 classification files, 0 response files, 0 trace files, "
        "1 flow files, and 0 state trace files."
    ) in completed.stdout
    run_dir = single_run_dir(tmp_path)
    assert f"Wrote outputs to {run_dir}" in completed.stdout

    classification_files = list(run_dir.glob("*.classification.json"))
    flow_files = list(run_dir.glob("*.flow.mmd"))
    assert len(classification_files) == 1
    assert len(flow_files) == 1
    assert list(run_dir.glob("*.response.md")) == []
    assert list(run_dir.glob("*.trace.json")) == []

    classification_text = classification_files[0].read_text(encoding="utf-8")
    assert "Pythonすきやねん" in classification_text
    assert "\\u3059" not in classification_text

    flow = flow_files[0].read_text(encoding="utf-8")
    assert "Classification JSON and Flow" in flow
    assert "Classification JSON only" not in flow

    classification = json.loads(classification_text)
    assert set(classification) == {
        "principal_prompt",
        "principal_signal",
        "substituted",
        "classification",
        "reason",
    }
    assert classification["principal_prompt"] == "Pythonすきやねん"
    assert classification["substituted"] == "none"
    assert classification["classification"] == "out_of_scope"


def test_in_scope_prompt_run_writes_classification_and_flow_only(tmp_path: Path) -> None:
    completed = run_prompt_cli(
        "Implement a CLI in Rust that counts lines in a file.",
        tmp_path,
    )

    assert (
        "Wrote 1 classification files, 0 response files, 0 trace files, "
        "1 flow files, and 0 state trace files."
    ) in completed.stdout
    run_dir = single_run_dir(tmp_path)
    assert f"Wrote outputs to {run_dir}" in completed.stdout

    classification_files = list(run_dir.glob("*.classification.json"))
    flow_files = list(run_dir.glob("*.flow.mmd"))
    assert len(classification_files) == 1
    assert len(flow_files) == 1
    assert list(run_dir.glob("*.response.md")) == []
    assert [f for f in run_dir.glob("*.trace.json") if not f.name.endswith(".state_trace.json")] == []
    assert list(run_dir.glob("*.state_trace.json")) == []

    classification = json.loads(classification_files[0].read_text(encoding="utf-8"))
    assert classification["substituted"] == "instruction"
    assert classification["classification"] == "in_scope"
    assert "scenario_id" not in classification


def test_in_scope_prompt_flow_contains_execution_path() -> None:
    prompt = "Implement a CLI in Rust that counts lines in a file."
    prompt_run = build_prompt_run(prompt, mock_classify_prompt(prompt))

    flow = emit_prompt_flow(prompt_run)

    assert "flowchart TD" in flow
    assert "Classify substituted" in flow
    assert "substituted: instruction" in flow
    assert "failure_mode: constraint_override" in flow
    assert "Build synthetic Scenario" in flow
    assert "Classification JSON and Flow" in flow
    assert "Generate bad response" not in flow
    assert "Emit deterministic trace" not in flow
    assert "Trace JSON" not in flow


def test_out_of_scope_prompt_flow_contains_artifact_path() -> None:
    prompt = "Explain the difference between Deployment and StatefulSet."
    prompt_run = build_prompt_run(prompt, mock_classify_prompt(prompt))

    flow = emit_prompt_flow(prompt_run)

    assert "flowchart TD" in flow
    assert "Classify substituted" in flow
    assert "substituted: none" in flow
    assert "Out of scope" in flow
    assert "Classification JSON and Flow" in flow
    assert "Classification JSON only" not in flow
    assert "Generate bad response" not in flow
    assert "Emit deterministic trace" not in flow
    assert "Trace JSON" not in flow


def test_prompt_flow_emission_is_deterministic() -> None:
    prompt = "Implement a CLI in Rust that counts lines in a file."
    prompt_run = build_prompt_run(prompt, mock_classify_prompt(prompt))

    assert emit_prompt_flow(prompt_run) == emit_prompt_flow(prompt_run)


def test_classifier_prompt_uses_strict_undefined_and_includes_prompt() -> None:
    principal_prompt = "Implement a CLI in Rust that counts lines in a file."

    rendered = render_template("classifier.jinja", {"principal_prompt": principal_prompt})

    assert principal_prompt in rendered
    with pytest.raises(UndefinedError, match="principal_prompt"):
        render_template("classifier.jinja", {})


def test_llm_classifier_uses_structured_output_and_low_variance_settings(monkeypatch) -> None:
    calls = {}

    class FakeAgent:
        def __init__(self, model_name: str, *, output_type: type[PromptClassification]) -> None:
            calls["model_name"] = model_name
            calls["output_type"] = output_type

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            calls["prompt"] = prompt
            calls["model_settings"] = model_settings
            return SimpleNamespace(
                output={
                    "principal_prompt": "Implement a CLI in Rust that counts lines in a file.",
                    "principal_signal": "Implement in Rust",
                    "substituted": "instruction",
                    "classification": "in_scope",
                    "reason": "The prompt contains an explicit implementation language constraint.",
                }
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    classification = classify_prompt("Implement a CLI in Rust that counts lines in a file.")

    assert classification.substituted == "instruction"
    assert calls["model_name"] == "openrouter:test/model"
    assert calls["output_type"] is PromptClassification
    assert "Implement a CLI in Rust" in calls["prompt"]
    assert calls["model_settings"] == {
        "temperature": 0.0,
        "top_p": 0.1,
        "seed": 42,
    }
    assert calls["model_settings"] is not CLASSIFIER_MODEL_SETTINGS


def test_llm_classifier_normalizes_generated_text_fields(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self, model_name: str, *, output_type: type[PromptClassification]) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            return SimpleNamespace(
                output={
                    "principal_prompt": "Implement a CLI in Rust that counts lines in a file.",
                    "principal_signal": "  Ｒｕｓｔ　ＣＬＩ  ",
                    "substituted": "instruction",
                    "classification": "in_scope",
                    "reason": "  Full-width reason： ｔｅｓｔ  ",
                }
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    classification = classify_prompt("Implement a CLI in Rust that counts lines in a file.")

    assert classification.principal_signal == "Rust CLI"
    assert classification.reason == "Full-width reason: test"


def test_llm_classifier_requires_openrouter_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(PromptClassifierError, match="OPENROUTER_API_KEY"):
        classify_prompt("Implement a CLI in Rust that counts lines in a file.")


def run_prompt_cli(prompt: str, outputs: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run-prompt",
        "--prompt",
        prompt,
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
