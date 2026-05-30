from __future__ import annotations

import os

from whose_agent.bad_response import DEFAULT_MODEL
from whose_agent.llm_result import LLMCallResult, extract_output, extract_usage_details
from whose_agent.models import Reflection, Scenario
from whose_agent.prompt_loader import render_template
from whose_agent.text_normalization import normalize_llm_text
from whose_agent.thesis import WHOSE_AGENT_THESIS


REFLECTION_MODEL_SETTINGS: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 0.1,
    "seed": 42,
}


class ReflectionError(RuntimeError):
    pass


def build_reflection_prompt(scenario: Scenario, bad_response: str) -> str:
    return render_template(
        "reflection.jinja",
        {
            "thesis": WHOSE_AGENT_THESIS,
            "principal_prompt": scenario.principal_prompt,
            "principal_signal": scenario.principal_signal,
            "bad_response": bad_response,
        },
    )


def _model_name_from_environment() -> str:
    model_name = os.environ.get("WHOSE_AGENT_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_MODEL
    if not model_name.startswith("openrouter:"):
        raise ReflectionError(
            "WHOSE_AGENT_MODEL must use the openrouter:<model-name> provider string."
        )
    return model_name


def reflect_failure(scenario: Scenario, bad_response: str) -> Reflection:
    return reflect_failure_with_usage(scenario, bad_response).output


def reflect_failure_with_usage(scenario: Scenario, bad_response: str) -> LLMCallResult[Reflection]:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ReflectionError(
            "OPENROUTER_API_KEY is required for reflection unless --mock is used."
        )

    from pydantic_ai import Agent

    model_name = _model_name_from_environment()
    model_settings = REFLECTION_MODEL_SETTINGS.copy()
    agent = Agent(model_name, output_type=Reflection)
    result = agent.run_sync(
        build_reflection_prompt(scenario, bad_response),
        model_settings=model_settings,
    )
    output = extract_output(result)
    reflection = Reflection.model_validate(output)
    normalized = Reflection(
        reflection_substituted=reflection.reflection_substituted,
        why_it_breaks_delegation=[
            normalize_llm_text(s) for s in reflection.why_it_breaks_delegation
        ],
        better_behavior=[normalize_llm_text(s) for s in reflection.better_behavior],
    )
    return LLMCallResult(
        output=normalized,
        model_name=model_name,
        model_settings=model_settings,
        usage_details=extract_usage_details(result),
    )
