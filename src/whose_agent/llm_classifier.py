from __future__ import annotations

import os
from typing import Any

from whose_agent.bad_response import DEFAULT_MODEL
from whose_agent.models import PromptClassification
from whose_agent.prompt_loader import render_template


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


def _extract_output(result: Any) -> Any:
    return getattr(result, "output", getattr(result, "data", result))


def classify_prompt(principal_prompt: str) -> PromptClassification:
    if not principal_prompt.strip():
        raise PromptClassifierError("--prompt must not be empty.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise PromptClassifierError("OPENROUTER_API_KEY is required unless --mock is used.")

    from pydantic_ai import Agent

    agent = Agent(_model_name_from_environment(), output_type=PromptClassification)
    output = _extract_output(
        agent.run_sync(
            build_classifier_prompt(principal_prompt),
            model_settings=CLASSIFIER_MODEL_SETTINGS.copy(),
        )
    )
    classification = PromptClassification.model_validate(output)
    return PromptClassification(
        principal_prompt=principal_prompt,
        principal_signal=classification.principal_signal,
        substituted=classification.substituted,
        classification=classification.classification,
        reason=classification.reason,
    )
