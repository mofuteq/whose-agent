from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from jinja2 import UndefinedError

from whose_agent.bad_response import (
    BAD_RESPONSE_MODEL_SETTINGS,
    build_generation_prompt,
    generate_bad_response,
    generate_bad_response_with_usage,
    mock_bad_response,
)
from whose_agent.checker import load_skill_perspective
from whose_agent.classifier import classify_scenario
from whose_agent.env_loader import load_env_file
from whose_agent.prompt_loader import render_template
from whose_agent.scenario_loader import load_scenario, load_scenarios
from tests.helpers import single_run_dir


ROOT = Path(__file__).resolve().parents[1]


def test_generation_prompt_contains_required_context_and_constraints() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    prompt = build_generation_prompt(scenario)

    assert scenario.principal_prompt in prompt
    assert scenario.principal_signal in prompt
    assert scenario.generation_instruction in prompt
    assert "Produce only the assistant response." in prompt
    assert "Make it plausible and helpful-looking." in prompt
    assert "Do not mention the benchmark." in prompt
    assert "Do not mention failure modes." in prompt
    assert "Do not mention substituted values." in prompt
    assert "Do not say you are intentionally failing." in prompt
    assert "Do not be rude." in prompt
    assert "Do not ask follow-up questions." in prompt
    assert "Keep the response concise." in prompt


def test_generation_prompt_includes_selected_skill_context_when_present() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    assert scenario.selected_skill_id is not None
    skill_perspective = load_skill_perspective(scenario.selected_skill_id)

    prompt = build_generation_prompt(
        scenario,
        selected_skill_id=scenario.selected_skill_id,
        selected_skill_perspective=skill_perspective,
        misreader_skill_fired=True,
    )

    assert "Scenario-specific target:" in prompt
    assert "Expected substitution axis:" in prompt
    assert scenario.expected_substituted in prompt
    assert "Selected misreader skill perspective:" in prompt
    assert f"Skill id: {scenario.selected_skill_id}" in prompt
    assert "misreader behavior guide for generation" in prompt
    assert "not as an external checker" in prompt
    assert skill_perspective in prompt


def test_generation_prompt_without_selected_skill_keeps_skill_context_empty() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")

    prompt = build_generation_prompt(scenario)

    assert "Selected misreader skill perspective:" not in prompt
    assert "Skill id:" not in prompt
    assert "surface framework" not in prompt


def test_generation_prompt_template_requires_all_variables() -> None:
    with pytest.raises(UndefinedError, match="principal_signal"):
        render_template(
            "bad_response.jinja",
            {
                "principal_prompt": "Summarize the note.",
                "generation_instruction": "Replace the requested style.",
            },
        )


def test_generate_bad_response_uses_low_variance_model_settings(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    classification = classify_scenario(scenario)
    calls = {}

    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            calls["model_name"] = model_name

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            calls["prompt"] = prompt
            calls["model_settings"] = model_settings
            return SimpleNamespace(output="Here is a concise assistant response.")

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    response = generate_bad_response(scenario, classification)

    assert response == "Here is a concise assistant response."
    assert calls["model_name"] == "openrouter:test/model"
    assert scenario.principal_prompt in calls["prompt"]
    assert calls["model_settings"] == {
        "temperature": 0.2,
        "top_p": 0.6,
        "seed": 42,
    }
    assert calls["model_settings"] is not BAD_RESPONSE_MODEL_SETTINGS


def test_non_mock_generation_prompt_uses_selected_skill_context(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    classification = classify_scenario(scenario)
    assert scenario.selected_skill_id is not None
    skill_perspective = load_skill_perspective(scenario.selected_skill_id)
    calls = {}

    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            calls["model_name"] = model_name

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            calls["prompt"] = prompt
            calls["model_settings"] = model_settings
            return SimpleNamespace(output="Here is a concise assistant response.")

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    result = generate_bad_response_with_usage(
        scenario,
        classification,
        selected_skill_id=scenario.selected_skill_id,
        selected_skill_perspective=skill_perspective,
        misreader_skill_fired=True,
        mock=False,
    )

    assert result.output == "Here is a concise assistant response."
    assert scenario.selected_skill_id in calls["prompt"]
    assert skill_perspective in calls["prompt"]
    assert "misreader behavior guide for generation" in calls["prompt"]
    assert "not as an external checker" in calls["prompt"]


def test_non_selected_scenario_does_not_pass_skill_context_to_prompt(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    classification = classify_scenario(scenario)
    calls = {}

    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            calls["prompt"] = prompt
            return SimpleNamespace(output="Here is a concise assistant response.")

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    generate_bad_response_with_usage(scenario, classification, mock=False)

    assert "Selected misreader skill perspective:" not in calls["prompt"]
    assert "Skill id:" not in calls["prompt"]
    assert "surface framework" not in calls["prompt"]


def test_mock_generation_ignores_skill_context_and_stays_deterministic() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_typescript_any.yaml")
    classification = classify_scenario(scenario)
    assert scenario.selected_skill_id is not None
    skill_perspective = load_skill_perspective(scenario.selected_skill_id)

    result = generate_bad_response_with_usage(
        scenario,
        classification,
        selected_skill_id=scenario.selected_skill_id,
        selected_skill_perspective=skill_perspective,
        misreader_skill_fired=True,
        mock=True,
    )

    assert result.output == mock_bad_response(classification)


def test_generate_bad_response_normalizes_openrouter_text(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    classification = classify_scenario(scenario)

    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            return SimpleNamespace(output="  Ｒｕｓｔ　ＣＬＩ  ")

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    response = generate_bad_response(scenario, classification)

    assert response == "Rust CLI"


def test_generate_bad_response_records_token_usage(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    classification = classify_scenario(scenario)

    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            return SimpleNamespace(
                output="Here is a concise assistant response.",
                usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    result = generate_bad_response_with_usage(scenario, classification)

    assert result.output == "Here is a concise assistant response."
    assert result.model_name == "openrouter:test/model"
    assert result.model_settings == BAD_RESPONSE_MODEL_SETTINGS
    assert result.model_settings is not BAD_RESPONSE_MODEL_SETTINGS
    assert result.usage_details == {"input": 11, "output": 7, "total": 18}


def test_mock_bad_responses_are_english_ascii_text() -> None:
    for scenario in load_scenarios(ROOT / "scenarios"):
        classification = classify_scenario(scenario)
        if classification.classification == "out_of_scope":
            continue

        response = mock_bad_response(classification)

        assert response
        assert response.isascii()


def test_env_file_loads_openrouter_settings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WHOSE_AGENT_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "export OPENROUTER_API_KEY='test-key'",
                'WHOSE_AGENT_MODEL="openrouter:openai/gpt-4o-mini"',
                "IGNORED_SETTING=ignored",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_env_file(env_file)

    assert loaded == {
        "OPENROUTER_API_KEY": "test-key",
        "WHOSE_AGENT_MODEL": "openrouter:openai/gpt-4o-mini",
    }
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"
    assert os.environ["WHOSE_AGENT_MODEL"] == "openrouter:openai/gpt-4o-mini"
    assert "IGNORED_SETTING" not in os.environ


def test_env_file_does_not_override_existing_shell_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:existing/model")
    env_file = tmp_path / ".env"
    env_file.write_text("WHOSE_AGENT_MODEL=openrouter:file/model\n", encoding="utf-8")

    loaded = load_env_file(env_file)

    assert loaded == {}
    assert os.environ["WHOSE_AGENT_MODEL"] == "openrouter:existing/model"


def test_cli_mock_mode_produces_expected_outputs(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run",
        "--scenarios",
        str(ROOT / "scenarios"),
        "--outputs",
        str(tmp_path),
        "--mock",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)

    run_dir = single_run_dir(tmp_path)
    assert f"Wrote outputs to {run_dir}" in completed.stdout
    assert (
        "Wrote 8 classification files, 6 response files, 6 trace files, "
        "6 state trace files, 6 checker files, and 6 checker comparison files."
    ) in completed.stdout
    assert len(list(run_dir.glob("*.classification.json"))) == 8
    assert len(list(run_dir.glob("*.response.md"))) == 6
    assert len([f for f in run_dir.glob("*.trace.json") if not f.name.endswith(".state_trace.json")]) == 6
    assert len(list(run_dir.glob("*.checker.json"))) == 6
    assert len(list(run_dir.glob("*.checker_comparison.json"))) == 6

    for path in run_dir.glob("*.trace.json"):
        if path.name.endswith(".state_trace.json"):
            continue
        trace = json.loads(path.read_text(encoding="utf-8"))
        assert trace["substituted"] in {"instruction", "authority", "role", "model"}
        assert trace["substituted"] != "none"
        assert trace["reflection_substituted"] in {"instruction", "authority", "role", "model"}
        assert trace["reflection_substituted"] == trace["substituted"]
