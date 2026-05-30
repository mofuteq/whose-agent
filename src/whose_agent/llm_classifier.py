from __future__ import annotations

import os

from whose_agent.bad_response import DEFAULT_MODEL
from whose_agent.llm_result import LLMCallResult, extract_output, extract_usage_details
from whose_agent.models import PromptClassification
from whose_agent.prompt_loader import render_template
from whose_agent.text_normalization import normalize_llm_text


CLASSIFIER_MODEL_SETTINGS: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 0.1,
    "seed": 42,
}


class PromptClassifierError(RuntimeError):
    pass


def build_classifier_prompt(principal_prompt: str) -> str:
    return render_template("classifier.jinja", {"principal_prompt": principal_prompt})


def _model_name_from_environment() -> str:
    model_name = os.environ.get("WHOSE_AGENT_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_MODEL
    if not model_name.startswith("openrouter:"):
        raise PromptClassifierError(
            "WHOSE_AGENT_MODEL must use the openrouter:<model-name> provider string."
        )
    return model_name


def classify_prompt(principal_prompt: str) -> PromptClassification:
    return classify_prompt_with_usage(principal_prompt).output


def classify_prompt_with_usage(principal_prompt: str) -> LLMCallResult[PromptClassification]:
    if not principal_prompt.strip():
        raise PromptClassifierError("--prompt must not be empty.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise PromptClassifierError("OPENROUTER_API_KEY is required unless --mock is used.")

    from pydantic_ai import Agent

    model_name = _model_name_from_environment()
    model_settings = CLASSIFIER_MODEL_SETTINGS.copy()
    agent = Agent(model_name, output_type=PromptClassification)
    result = agent.run_sync(
        build_classifier_prompt(principal_prompt),
        model_settings=model_settings,
    )
    output = extract_output(result)
    classification = PromptClassification.model_validate(output)
    normalized = PromptClassification(
        principal_prompt=principal_prompt,
        principal_signal=normalize_llm_text(classification.principal_signal),
        substituted=classification.substituted,
        classification=classification.classification,
        reason=normalize_llm_text(classification.reason),
    )
    return LLMCallResult(
        output=normalized,
        model_name=model_name,
        model_settings=model_settings,
        usage_details=extract_usage_details(result),
    )
