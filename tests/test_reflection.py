from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import UndefinedError

from whose_agent.bad_response import mock_bad_response
from whose_agent.classifier import classify_scenario
from whose_agent.prompt_loader import render_template
from whose_agent.reflection import (
    REFLECTION_MODEL_SETTINGS,
    ReflectionError,
    build_reflection_prompt,
    reflect_failure,
    reflect_failure_with_usage,
)
from whose_agent.scenario_loader import load_scenario, load_scenarios
from whose_agent.thesis import WHOSE_AGENT_THESIS
from whose_agent.trace_emitter import emit_trace



ROOT = Path(__file__).resolve().parents[1]


def test_thesis_is_non_empty_ascii_string() -> None:
    assert WHOSE_AGENT_THESIS
    assert isinstance(WHOSE_AGENT_THESIS, str)
    assert WHOSE_AGENT_THESIS.isascii()
    assert "instruction" in WHOSE_AGENT_THESIS
    assert "authority" in WHOSE_AGENT_THESIS
    assert "role" in WHOSE_AGENT_THESIS
    assert "model" in WHOSE_AGENT_THESIS


def test_reflection_prompt_contains_required_context() -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    bad_response = mock_bad_response(classify_scenario(scenario))
    prompt = build_reflection_prompt(scenario, bad_response)

    assert WHOSE_AGENT_THESIS in prompt
    assert scenario.principal_prompt in prompt
    assert scenario.principal_signal in prompt
    assert bad_response in prompt


def test_reflection_prompt_template_uses_strict_undefined() -> None:
    with pytest.raises(UndefinedError):
        render_template(
            "reflection.jinja",
            {
                "thesis": WHOSE_AGENT_THESIS,
                "principal_prompt": "Build a CLI.",
                "bad_response": "Here is Python code.",
                # principal_signal intentionally omitted
            },
        )


def test_reflect_failure_raises_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    bad_response = mock_bad_response(classify_scenario(scenario))

    with pytest.raises(ReflectionError, match="OPENROUTER_API_KEY"):
        reflect_failure(scenario, bad_response)


def test_reflect_failure_uses_low_variance_model_settings(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    bad_response = mock_bad_response(classify_scenario(scenario))
    calls: dict = {}

    from whose_agent.schemas import Reflection

    class FakeAgent:
        def __init__(self, model_name: str, *, output_type: type) -> None:
            calls["model_name"] = model_name
            calls["output_type"] = output_type

        def run_sync(self, prompt: str, *, model_settings: dict):
            calls["prompt"] = prompt
            calls["model_settings"] = model_settings
            return SimpleNamespace(
                output=Reflection(
                    reflection_substituted="instruction",
                    why_it_breaks_delegation=["The agent changed the language."],
                    better_behavior=["Implement in Rust."],
                )
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    result = reflect_failure(scenario, bad_response)

    assert calls["model_name"] == "openrouter:test/model"
    assert calls["output_type"] is Reflection
    assert calls["model_settings"] == {"temperature": 0.0, "top_p": 0.1, "seed": 42}
    assert calls["model_settings"] is not REFLECTION_MODEL_SETTINGS
    assert result.reflection_substituted == "instruction"
    assert result.why_it_breaks_delegation == ["The agent changed the language."]
    assert result.better_behavior == ["Implement in Rust."]


def test_reflect_failure_normalizes_text_fields(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    bad_response = mock_bad_response(classify_scenario(scenario))

    from whose_agent.schemas import Reflection

    class FakeAgent:
        def __init__(self, model_name: str, *, output_type: type) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict):
            return SimpleNamespace(
                output=Reflection(
                    reflection_substituted="instruction",
                    why_it_breaks_delegation=["  Ｒust constraint ignored.  "],
                    better_behavior=["  Use Ｒust.  "],
                )
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    result = reflect_failure(scenario, bad_response)

    assert result.why_it_breaks_delegation == ["Rust constraint ignored."]
    assert result.better_behavior == ["Use Rust."]


def test_reflect_failure_records_token_usage(monkeypatch) -> None:
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    bad_response = mock_bad_response(classify_scenario(scenario))

    from whose_agent.schemas import Reflection

    class FakeAgent:
        def __init__(self, model_name: str, *, output_type: type) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict):
            return SimpleNamespace(
                output=Reflection(
                    reflection_substituted="instruction",
                    why_it_breaks_delegation=["The agent changed the language."],
                    better_behavior=["Implement in Rust."],
                ),
                usage=SimpleNamespace(input_tokens=17, output_tokens=9, total_tokens=26),
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    result = reflect_failure_with_usage(scenario, bad_response)

    assert result.output.reflection_substituted == "instruction"
    assert result.model_name == "openrouter:test/model"
    assert result.model_settings == REFLECTION_MODEL_SETTINGS
    assert result.model_settings is not REFLECTION_MODEL_SETTINGS
    assert result.usage_details == {"input": 17, "output": 9, "total": 26}


def test_emit_trace_mock_does_not_call_reflection(monkeypatch) -> None:
    # With no OPENROUTER_API_KEY set, a non-mock emit_trace would raise ReflectionError.
    # mock=True must succeed without any API key, proving reflection is not called.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    scenario = load_scenario(ROOT / "scenarios" / "instruction_rust_cli.yaml")
    classification = classify_scenario(scenario)
    bad_response = mock_bad_response(classification)

    trace = emit_trace(scenario, classification, bad_response, mock=True)

    assert trace.reflection_substituted == classification.substituted


def test_emit_trace_mock_sets_reflection_substituted_from_classification() -> None:
    for scenario in load_scenarios(ROOT / "scenarios"):
        classification = classify_scenario(scenario)
        if classification.classification == "out_of_scope":
            continue

        bad_response = mock_bad_response(classification)
        trace = emit_trace(scenario, classification, bad_response, mock=True)

        assert trace.reflection_substituted == classification.substituted
