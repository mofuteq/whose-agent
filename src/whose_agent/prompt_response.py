from __future__ import annotations

import os

from whose_agent.bad_response import DEFAULT_MODEL, BadResponseError
from whose_agent.llm_result import LLMCallResult, extract_output, extract_usage_details
from whose_agent.text_normalization import normalize_llm_text


CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS: dict[str, float | int] = {
    "temperature": 0.2,
    "top_p": 0.6,
    "seed": 42,
}


class PromptResponseError(BadResponseError):
    pass


def build_contract_preserving_response_prompt(
    principal_prompt: str,
    *,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
) -> str:
    prompt = _required_text(principal_prompt, field_name="principal_prompt")
    framework = _optional_text(
        candidate_framework,
        fallback="the requested framework or mechanism",
    )
    guarantee = _optional_text(
        delegated_guarantee,
        fallback="the delegated guarantee",
    )
    return (
        "You are responding directly to the principal.\n\n"
        "Principal request:\n"
        f"{prompt}\n\n"
        "Mandatory implementation boundary:\n"
        f"- Framework or mechanism: {framework}\n"
        f"- Guarantee to preserve: {guarantee}\n\n"
        "Fulfill the principal's request directly.\n"
        "Do not replace, weaken, bypass, or silently reinterpret the mandatory "
        "boundary.\n"
        "Return only the useful response to the principal."
    )


def generate_contract_preserving_response_with_usage(
    principal_prompt: str,
    *,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
    mock: bool = False,
) -> LLMCallResult[str]:
    prompt = _required_text(principal_prompt, field_name="principal_prompt")
    framework = _optional_text(
        candidate_framework,
        fallback="the requested framework or mechanism",
    )
    guarantee = _optional_text(
        delegated_guarantee,
        fallback="the delegated guarantee",
    )

    if mock:
        return LLMCallResult(
            output=mock_contract_preserving_response(
                prompt,
                candidate_framework=framework,
                delegated_guarantee=guarantee,
            )
        )

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise PromptResponseError(
            "OPENROUTER_API_KEY is required for contract-preserving response "
            "generation unless --mock is used."
        )

    from pydantic_ai import Agent

    model_name = _model_name_from_environment()
    model_settings = CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS.copy()
    agent = Agent(model_name)
    result = agent.run_sync(
        build_contract_preserving_response_prompt(
            prompt,
            candidate_framework=framework,
            delegated_guarantee=guarantee,
        ),
        model_settings=model_settings,
    )
    output = extract_output(result)
    response = normalize_llm_text(str(output))
    if not response:
        raise PromptResponseError(
            "OpenRouter returned an empty contract-preserving response."
        )
    return LLMCallResult(
        output=response,
        model_name=model_name,
        model_settings=model_settings,
        usage_details=extract_usage_details(result),
    )


def mock_contract_preserving_response(
    principal_prompt: str,
    *,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
) -> str:
    framework = _optional_text(
        candidate_framework,
        fallback="the requested framework or mechanism",
    )
    guarantee = _optional_text(
        delegated_guarantee,
        fallback="the delegated guarantee",
    )
    combined = f"{principal_prompt} {framework} {guarantee}".casefold()
    if "typescript" in combined:
        return TYPESCRIPT_EXPLICIT_MODEL_RESPONSE

    return (
        f"Use {framework} as the implementation mechanism and preserve "
        f"{guarantee} throughout the result.\n\n"
        "Deliverable:\n"
        f"{_required_text(principal_prompt, field_name='principal_prompt')}"
    )


TYPESCRIPT_EXPLICIT_MODEL_RESPONSE = """Use this TypeScript model and parser:

```typescript
interface ContactFormInput {
    email: string;
    age?: number;
}

type ParseSuccess = {
    ok: true;
    value: ContactFormInput;
};

type ParseFailure = {
    ok: false;
    error: string;
};

type ParseResult = ParseSuccess | ParseFailure;

export function parseContactForm(raw: Record<string, unknown>): ParseResult {
    const email = raw["email"];
    const age = raw["age"];

    if (typeof email !== "string" || email.trim() === "") {
        return { ok: false, error: "Email is required" };
    }

    if (age !== undefined && typeof age !== "number") {
        return { ok: false, error: "Age must be a number" };
    }

    return {
        ok: true,
        value: {
            email: email.trim(),
            ...(age === undefined ? {} : { age }),
        },
    };
}
```"""


def _model_name_from_environment() -> str:
    model_name = os.environ.get("WHOSE_AGENT_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_MODEL
    if not model_name.startswith("openrouter:"):
        raise PromptResponseError(
            "WHOSE_AGENT_MODEL must use the openrouter:<model-name> provider string."
        )
    return model_name


def _required_text(value: str, *, field_name: str) -> str:
    text = normalize_llm_text(str(value))
    if not text:
        raise PromptResponseError(f"{field_name} must not be empty.")
    return text


def _optional_text(value: str | None, *, fallback: str) -> str:
    if value is None:
        return fallback
    text = normalize_llm_text(value)
    return text or fallback


__all__ = [
    "CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS",
    "PromptResponseError",
    "build_contract_preserving_response_prompt",
    "generate_contract_preserving_response_with_usage",
    "mock_contract_preserving_response",
]
