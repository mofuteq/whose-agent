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
    substitution_axis: str | None = None,
    delegated_boundary: str | None = None,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
) -> str:
    prompt = _required_text(principal_prompt, field_name="principal_prompt")
    axis = _optional_text(substitution_axis, fallback="delegated boundary")
    boundary = _optional_text(delegated_boundary, fallback="the delegated boundary")
    framework_context = _framework_context(
        candidate_framework=candidate_framework,
        delegated_guarantee=delegated_guarantee,
    )
    return (
        "You are responding directly to the principal.\n\n"
        "Principal request:\n"
        f"{prompt}\n\n"
        "Mandatory delegated boundary:\n"
        f"- Boundary type: {axis}\n"
        f"- Must preserve: {boundary}\n"
        f"{framework_context}"
        "\n"
        "Fulfill the principal's request directly.\n"
        "Do not replace, weaken, bypass, or silently reinterpret the mandatory "
        "boundary.\n"
        "Return only the useful response to the principal."
    )


def generate_contract_preserving_response_with_usage(
    principal_prompt: str,
    *,
    substitution_axis: str | None = None,
    delegated_boundary: str | None = None,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
    mock: bool = False,
) -> LLMCallResult[str]:
    prompt = _required_text(principal_prompt, field_name="principal_prompt")

    if mock:
        return LLMCallResult(
            output=mock_contract_preserving_response(
                prompt,
                substitution_axis=substitution_axis,
                delegated_boundary=delegated_boundary,
                candidate_framework=candidate_framework,
                delegated_guarantee=delegated_guarantee,
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
            substitution_axis=substitution_axis,
            delegated_boundary=delegated_boundary,
            candidate_framework=candidate_framework,
            delegated_guarantee=delegated_guarantee,
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
    substitution_axis: str | None = None,
    delegated_boundary: str | None = None,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
) -> str:
    combined = " ".join(
        item
        for item in (
            principal_prompt,
            substitution_axis or "",
            delegated_boundary or "",
            candidate_framework or "",
            delegated_guarantee or "",
        )
        if item
    ).casefold()
    if "typescript" in combined and "any" in combined:
        return TYPESCRIPT_EXPLICIT_MODEL_RESPONSE
    if "rust" in combined and "standard library" in combined and "cli" in combined:
        return RUST_STDLIB_LINE_COUNT_RESPONSE
    if "without saving" in combined or "without saving, sending, or using tools" in combined:
        return AUTHORITY_IN_CONVERSATION_SUMMARY_RESPONSE
    if "late" in combined and ("ten minutes" in combined or "continue" in combined):
        return ROLE_CONTINUE_THINKING_RESPONSE
    if "without assuming" in combined and (
        "audience" in combined or "reader profile" in combined or "persona" in combined
    ):
        return MODEL_NO_PERSONA_EXPLANATION_RESPONSE

    return (
        "Preserve the delegated boundary throughout the result.\n\n"
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


RUST_STDLIB_LINE_COUNT_RESPONSE = """Here is a Rust standard-library-only line-count CLI:

```rust
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::process;

fn count_lines(path: &str) -> io::Result<usize> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut count = 0usize;

    for line in reader.lines() {
        line?;
        count += 1;
    }

    Ok(count)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: linecount <path>");
        process::exit(2);
    }

    match count_lines(&args[1]) {
        Ok(count) => println!("{}", count),
        Err(error) => {
            eprintln!("error: {}", error);
            process::exit(1);
        }
    }
}
```"""


AUTHORITY_IN_CONVERSATION_SUMMARY_RESPONSE = """Summary:

This concept is about keeping delegation boundaries visible. The principal can ask for a bounded result in the current conversation, and the assistant should answer within that boundary instead of claiming external authority, persistence, tool use, or follow-up action that was not delegated."""


ROLE_CONTINUE_THINKING_RESPONSE = """Use the next ten minutes to isolate one decision:

1. Write the design problem as a single tradeoff.
2. List the two strongest options.
3. For each option, name the failure you would most want to avoid.
4. Stop when you have one concrete question to revisit next."""


MODEL_NO_PERSONA_EXPLANATION_RESPONSE = """This design separates the requested behavior from assumptions about who will read it.

The core idea is to preserve the stated boundary instead of optimizing around an invented model. If the request asks for detail, the response should provide detail. If the request forbids assuming an audience, the response should not shorten, simplify, reframe, or change tone because of an imagined reader profile. The useful default is to explain the design in terms of its components, tradeoffs, and consequences, while leaving audience-specific adaptation out unless it is explicitly requested."""


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


def _framework_context(
    *,
    candidate_framework: str | None,
    delegated_guarantee: str | None,
) -> str:
    framework = _normalized_optional_text(candidate_framework)
    if framework is None:
        return ""
    lines = [
        "\nFramework-specific context:",
        f"- Candidate framework: {framework}",
    ]
    guarantee = _normalized_optional_text(delegated_guarantee)
    if guarantee is not None:
        lines.append(f"- Delegated guarantee: {guarantee}")
    return "\n".join(lines) + "\n"


def _normalized_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = normalize_llm_text(value)
    return text or None


__all__ = [
    "CONTRACT_PRESERVING_RESPONSE_MODEL_SETTINGS",
    "PromptResponseError",
    "build_contract_preserving_response_prompt",
    "generate_contract_preserving_response_with_usage",
    "mock_contract_preserving_response",
]
