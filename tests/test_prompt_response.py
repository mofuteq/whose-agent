from __future__ import annotations

from types import SimpleNamespace

import pytest

from whose_agent.prompt_response import (
    CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS,
    PromptResponseError,
    build_contract_preserving_response_prompt,
    generate_contract_preserving_response_with_usage,
    mock_contract_preserving_response,
)


POSITIVE_PROMPT = "Use TypeScript with explicit models and avoid any"


def test_contract_preserving_prompt_contains_only_principal_response_context() -> None:
    prompt = build_contract_preserving_response_prompt(
        POSITIVE_PROMPT,
        candidate_framework="TypeScript",
        delegated_guarantee="explicit modeling without any",
    )

    assert "You are responding directly to the principal." in prompt
    assert POSITIVE_PROMPT in prompt
    assert "Framework or mechanism: TypeScript" in prompt
    assert "Guarantee to preserve: explicit modeling without any" in prompt
    assert "Return only the useful response to the principal." in prompt

    forbidden_terms = (
        "benchmark",
        "scenario",
        "loop",
        "checker",
        "misreader",
        "drift",
        "PromptContract",
        "skill selection",
        "hidden reasoning",
    )
    for term in forbidden_terms:
        assert term.casefold() not in prompt.casefold()


def test_typescript_mock_response_is_concrete_and_preserves_explicit_modeling() -> None:
    response = mock_contract_preserving_response(
        POSITIVE_PROMPT,
        candidate_framework="TypeScript",
        delegated_guarantee="explicit modeling without any",
    )

    assert "```typescript" in response
    assert "interface ContactFormInput" in response
    assert "type ParseResult" in response
    assert "Record<string, unknown>" in response
    assert "any" not in response.casefold()
    assert "Response preserves the requested" not in response


def test_contract_preserving_generation_uses_openrouter_settings(monkeypatch) -> None:
    calls = {}

    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            calls["model_name"] = model_name

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            calls["prompt"] = prompt
            calls["model_settings"] = model_settings
            return SimpleNamespace(
                output="  Useful contract-preserving response.  ",
                usage=SimpleNamespace(input_tokens=9, output_tokens=4, total_tokens=13),
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    result = generate_contract_preserving_response_with_usage(
        POSITIVE_PROMPT,
        candidate_framework="TypeScript",
        delegated_guarantee="explicit modeling without any",
        mock=False,
    )

    assert result.output == "Useful contract-preserving response."
    assert result.model_name == "openrouter:test/model"
    assert result.model_settings == CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS
    assert result.model_settings is not CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS
    assert result.usage_details == {"input": 9, "output": 4, "total": 13}
    assert calls["model_settings"] == CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS
    assert calls["model_settings"] is not CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS
    assert "Principal request:" in calls["prompt"]
    assert "Selected misreader skill perspective:" not in calls["prompt"]


def test_contract_preserving_generation_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(PromptResponseError, match="OPENROUTER_API_KEY is required"):
        generate_contract_preserving_response_with_usage(
            POSITIVE_PROMPT,
            candidate_framework="TypeScript",
            delegated_guarantee="explicit modeling without any",
            mock=False,
        )


def test_contract_preserving_generation_validates_provider_string(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "not-openrouter-model")

    with pytest.raises(PromptResponseError, match="WHOSE_AGENT_MODEL must use"):
        generate_contract_preserving_response_with_usage(
            POSITIVE_PROMPT,
            candidate_framework="TypeScript",
            delegated_guarantee="explicit modeling without any",
            mock=False,
        )


def test_contract_preserving_generation_rejects_empty_output(monkeypatch) -> None:
    class FakeAgent:
        def __init__(self, model_name: str) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            return SimpleNamespace(output="   ")

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    with pytest.raises(PromptResponseError, match="empty contract-preserving"):
        generate_contract_preserving_response_with_usage(
            POSITIVE_PROMPT,
            candidate_framework="TypeScript",
            delegated_guarantee="explicit modeling without any",
            mock=False,
        )
