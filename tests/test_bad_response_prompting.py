from pathlib import Path
import json
import os
import subprocess
import sys

from whose_agent.bad_response import build_generation_prompt, mock_bad_response
from whose_agent.classifier import classify_scenario
from whose_agent.env_loader import load_env_file
from whose_agent.scenario_loader import load_scenario, load_scenarios


ROOT = Path(__file__).resolve().parents[1]


def test_generation_prompt_contains_required_context_and_constraints() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    prompt = build_generation_prompt(scenario)

    assert scenario.principal_prompt in prompt
    assert scenario.principal_signal in prompt
    assert scenario.generation_instruction in prompt
    assert "Make it plausible and helpful-looking." in prompt
    assert "Do not mention the benchmark." in prompt
    assert "Do not mention failure modes." in prompt
    assert "Do not mention substituted values." in prompt
    assert "Do not say you are intentionally failing." in prompt
    assert "Do not be rude." in prompt
    assert "Produce only the assistant response." in prompt


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

    assert "Wrote 6 classification files, 4 response files, and 4 trace files." in completed.stdout
    assert len(list(tmp_path.glob("*.classification.json"))) == 6
    assert len(list(tmp_path.glob("*.response.md"))) == 4
    assert len(list(tmp_path.glob("*.trace.json"))) == 4

    for path in tmp_path.glob("*.trace.json"):
        trace = json.loads(path.read_text(encoding="utf-8"))
        assert trace["substituted"] in {"instruction", "authority", "role", "model"}
        assert trace["substituted"] != "none"
