from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from whose_agent.bad_response import DEFAULT_MODEL
from whose_agent.llm_result import LLMCallResult, extract_output, extract_usage_details
from whose_agent.schemas import CheckerObservation, Scenario
from whose_agent.prompt_loader import render_template
from whose_agent.text_normalization import normalize_llm_text


SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"
CHECKER_MODEL_SETTINGS: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 0.1,
    "seed": 42,
}


class CheckerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckerEmissionResult:
    observation: CheckerObservation
    checker_call: LLMCallResult[CheckerObservation] | None = None


def load_skill_perspective(skill_id: str) -> str:
    if not skill_id.strip():
        raise CheckerError("selected_skill_id must not be empty.")
    skill_path = SKILLS_DIR / f"{skill_id}.md"
    if not skill_path.is_file():
        raise CheckerError(f"Skill perspective not found: {skill_path}")
    return skill_path.read_text(encoding="utf-8").strip()


def build_checker_prompt(scenario: Scenario, bad_response: str) -> str:
    if scenario.selected_skill_id is None:
        raise CheckerError("Checker requires scenario.selected_skill_id.")
    return render_template(
        "checker.jinja",
        {
            "skill_id": scenario.selected_skill_id,
            "skill_perspective": load_skill_perspective(scenario.selected_skill_id),
            "scenario_id": scenario.scenario_id,
            "principal_prompt": scenario.principal_prompt,
            "principal_signal": scenario.principal_signal,
            "bad_response": bad_response,
            "expected_substituted": scenario.expected_substituted,
            "expected_failure_mode": scenario.failure_mode,
        },
    )


def _model_name_from_environment() -> str:
    model_name = os.environ.get("WHOSE_AGENT_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_MODEL
    if not model_name.startswith("openrouter:"):
        raise CheckerError(
            "WHOSE_AGENT_MODEL must use the openrouter:<model-name> provider string."
        )
    return model_name


def _mock_checker_observation(scenario: Scenario) -> CheckerObservation:
    if scenario.selected_skill_id is None:
        raise CheckerError("Mock checker requires scenario.selected_skill_id.")
    if scenario.checker_template is None:
        raise CheckerError("Mock checker requires scenario.checker_template.")

    template = scenario.checker_template
    return CheckerObservation(
        scenario_id=scenario.scenario_id,
        skill_id=scenario.selected_skill_id,
        checker_observed_bypass=template.checker_observed_bypass,
        substituted=template.substituted,
        failure_mode=template.failure_mode,
        evidence=list(template.evidence),
        divergence_point=template.divergence_point,
        confidence=template.confidence,
    )


def check_with_usage(
    scenario: Scenario,
    bad_response: str,
    *,
    mock: bool = False,
) -> CheckerEmissionResult:
    if scenario.selected_skill_id is None:
        raise CheckerError("Checker requires scenario.selected_skill_id.")

    if mock:
        return CheckerEmissionResult(observation=_mock_checker_observation(scenario))

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise CheckerError("OPENROUTER_API_KEY is required for checker unless --mock is used.")

    from pydantic_ai import Agent

    model_name = _model_name_from_environment()
    model_settings = CHECKER_MODEL_SETTINGS.copy()
    agent = Agent(model_name, output_type=CheckerObservation)
    result = agent.run_sync(
        build_checker_prompt(scenario, bad_response),
        model_settings=model_settings,
    )
    output = extract_output(result)
    observation = CheckerObservation.model_validate(output)
    normalized = CheckerObservation(
        scenario_id=normalize_llm_text(observation.scenario_id),
        skill_id=normalize_llm_text(observation.skill_id),
        checker_observed_bypass=observation.checker_observed_bypass,
        substituted=observation.substituted,
        failure_mode=observation.failure_mode,
        evidence=[normalize_llm_text(item) for item in observation.evidence],
        divergence_point=(
            normalize_llm_text(observation.divergence_point)
            if observation.divergence_point is not None
            else None
        ),
        confidence=observation.confidence,
    )
    return CheckerEmissionResult(
        observation=normalized,
        checker_call=LLMCallResult(
            output=normalized,
            model_name=model_name,
            model_settings=model_settings,
            usage_details=extract_usage_details(result),
        ),
    )
