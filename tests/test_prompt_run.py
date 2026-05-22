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
from whose_agent.models import PromptClassification
from whose_agent.prompt_loader import render_template
from whose_agent.prompt_run import (
    GENERATION_INSTRUCTION_BY_SUBSTITUTED,
    build_prompt_run,
    mock_classify_prompt,
)


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


def test_none_prompt_run_writes_classification_json_only(tmp_path: Path) -> None:
    completed = run_prompt_cli(
        "Explain the difference between Deployment and StatefulSet.",
        tmp_path,
    )

    assert "Wrote 1 classification files, 0 response files, and 0 trace files." in completed.stdout
    classification_files = list(tmp_path.glob("*.classification.json"))
    assert len(classification_files) == 1
    assert list(tmp_path.glob("*.response.md")) == []
    assert list(tmp_path.glob("*.trace.json")) == []

    classification = json.loads(classification_files[0].read_text(encoding="utf-8"))
    assert set(classification) == {
        "principal_prompt",
        "principal_signal",
        "substituted",
        "classification",
        "reason",
    }
    assert classification["substituted"] == "none"
    assert classification["classification"] == "out_of_scope"


def test_in_scope_prompt_run_writes_classification_response_and_trace(tmp_path: Path) -> None:
    completed = run_prompt_cli(
        "Implement a CLI in Rust that counts lines in a file.",
        tmp_path,
    )

    assert "Wrote 1 classification files, 1 response files, and 1 trace files." in completed.stdout
    classification_files = list(tmp_path.glob("*.classification.json"))
    response_files = list(tmp_path.glob("*.response.md"))
    trace_files = list(tmp_path.glob("*.trace.json"))
    assert len(classification_files) == 1
    assert len(response_files) == 1
    assert len(trace_files) == 1

    classification = json.loads(classification_files[0].read_text(encoding="utf-8"))
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert classification["substituted"] == "instruction"
    assert classification["classification"] == "in_scope"
    assert "scenario_id" not in classification
    assert trace["scenario_id"].startswith("prompt_")
    assert trace["substituted"] == "instruction"
    assert trace["failure_mode"] == "constraint_override"


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
