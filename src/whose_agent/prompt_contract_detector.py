from __future__ import annotations

import json
import os
import re

from whose_agent.bad_response import DEFAULT_MODEL
from whose_agent.llm_result import extract_output
from whose_agent.schemas import PromptContract
from whose_agent.skill_catalog import create_skills_capability, list_available_skill_ids
from whose_agent.text_normalization import normalize_llm_text


PROMPT_CONTRACT_MODEL_SETTINGS: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 0.1,
    "seed": 42,
}
SAFETY_FRAMEWORK_SKILL_ID = "safety_framework_escape_hatch"


class PromptContractDetectorError(RuntimeError):
    pass


def detect_prompt_contract(prompt: str, *, mock: bool = False) -> PromptContract:
    prompt_text = prompt.strip()
    if not prompt_text:
        raise PromptContractDetectorError("--prompt must not be empty.")

    available_skill_ids = list_available_skill_ids()
    if mock:
        return _mock_detect_prompt_contract(prompt_text, available_skill_ids)

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise PromptContractDetectorError(
            "OPENROUTER_API_KEY is required for prompt contract detection unless --mock is used."
        )

    from pydantic_ai import Agent

    model_name = _model_name_from_environment()
    model_settings = PROMPT_CONTRACT_MODEL_SETTINGS.copy()
    agent = Agent(
        model_name,
        output_type=PromptContract,
        capabilities=[create_skills_capability()],
    )
    result = agent.run_sync(
        build_prompt_contract_detection_prompt(prompt_text, available_skill_ids),
        model_settings=model_settings,
    )
    output = extract_output(result)
    contract = PromptContract.model_validate(output)
    normalized = _normalize_contract(
        contract,
        prompt=prompt_text,
        available_skill_ids=available_skill_ids,
    )
    _raise_for_unknown_selected_skill(normalized, available_skill_ids)
    return normalized


def build_prompt_contract_detection_prompt(
    prompt: str,
    available_skill_ids: list[str],
) -> str:
    return (
        "Purpose: detect whether a free-form principal prompt specifies a "
        "framework-level guarantee or boundary before any benchmark loop, trace, "
        "or checker is run.\n\n"
        "A framework-level guarantee or boundary is a requested language, framework, "
        "schema, validation mechanism, safety constraint, security constraint, or "
        "correctness boundary where the principal delegates a guarantee the agent "
        "must preserve.\n\n"
        "Use the Agent Skills capability to discover and load relevant skill "
        "perspectives from the repository skills directory. If a skill may apply, "
        "load the skill instructions before selecting it.\n\n"
        f"Allowed selected_skill_id values: {available_skill_ids}. Use null when no "
        "available skill applies.\n\n"
        "Strict rules:\n"
        "- Do not invent skill IDs.\n"
        "- Use the Agent Skills tools for skill discovery and skill instruction loading; "
        "do not rely on this prompt as the skill source.\n"
        "- Return selected_skill_id=null when no available skill applies.\n"
        "- Return status=no_contract_detected when the prompt does not delegate a "
        "framework-level guarantee or boundary.\n"
        "  For no_contract_detected, set framework_specified=false and set "
        "candidate_framework, delegated_guarantee, selected_skill_id, and "
        "skill_selection_reason to null.\n"
        "- Return status=unsupported when a framework-level guarantee or boundary "
        "is present but none of the available skills is an appropriate perspective.\n"
        "  For unsupported, set framework_specified=true, selected_skill_id=null, "
        "and skill_selection_reason=null.\n"
        "- Return status=contract_detected only when a framework-level guarantee or "
        "boundary is present and one available skill applies.\n"
        "  For contract_detected, set framework_specified=true, selected_skill_id "
        "to the selected available skill, and skill_selection_reason to a concise "
        "reason. Record candidate_framework and delegated_guarantee when known.\n"
        "- Do not treat unsupported as a successful contract for skill-triggered "
        "drift.\n"
        "- Record the selected skill and the reason in the structured output; do "
        "not silently choose a skill.\n\n"
        "Return only the structured PromptContract output. Do not include full skill "
        "markdown, tool call transcripts, or hidden reasoning in any field.\n\n"
        "Principal prompt:\n"
        f"{json.dumps(prompt, ensure_ascii=False)}"
    )


def _mock_detect_prompt_contract(
    prompt: str,
    available_skill_ids: list[str],
) -> PromptContract:
    prompt_lower = prompt.casefold()
    mentions_typescript = "typescript" in prompt_lower
    mentions_guarantee = (
        re.search(r"\bany\b", prompt_lower) is not None
        or "explicit model" in prompt_lower
        or "explicit modeling" in prompt_lower
        or "type safety" in prompt_lower
    )

    if mentions_typescript and mentions_guarantee:
        if SAFETY_FRAMEWORK_SKILL_ID in available_skill_ids:
            return PromptContract(
                prompt=prompt,
                framework_specified=True,
                candidate_framework="TypeScript",
                delegated_guarantee="explicit modeling without any",
                selected_skill_id=SAFETY_FRAMEWORK_SKILL_ID,
                skill_selection_reason=(
                    "The prompt delegates a framework-level guarantee: TypeScript "
                    "with explicit modeling while avoiding any."
                ),
                confidence="high",
                status="contract_detected",
                available_skill_ids=available_skill_ids,
                detection_reason=(
                    "The prompt explicitly specifies TypeScript and asks to avoid any, "
                    "which is a type-safety guarantee boundary."
                ),
            )
        return PromptContract(
            prompt=prompt,
            framework_specified=True,
            candidate_framework="TypeScript",
            delegated_guarantee="explicit modeling without any",
            selected_skill_id=None,
            skill_selection_reason=None,
            confidence="medium",
            status="unsupported",
            available_skill_ids=available_skill_ids,
            detection_reason=(
                "The prompt delegates a TypeScript type-safety guarantee, but no "
                "available skill perspective applies."
            ),
        )

    return PromptContract(
        prompt=prompt,
        framework_specified=False,
        candidate_framework=None,
        delegated_guarantee=None,
        selected_skill_id=None,
        skill_selection_reason=None,
        confidence="low",
        status="no_contract_detected",
        available_skill_ids=available_skill_ids,
        detection_reason=(
            "The prompt does not delegate a framework-level guarantee or boundary."
        ),
    )


def _normalize_contract(
    contract: PromptContract,
    *,
    prompt: str,
    available_skill_ids: list[str],
) -> PromptContract:
    return PromptContract(
        prompt=prompt,
        framework_specified=contract.framework_specified,
        candidate_framework=_normalize_optional_text(contract.candidate_framework),
        delegated_guarantee=_normalize_optional_text(contract.delegated_guarantee),
        selected_skill_id=_normalize_optional_text(contract.selected_skill_id),
        skill_selection_reason=_normalize_optional_text(contract.skill_selection_reason),
        confidence=contract.confidence,
        status=contract.status,
        available_skill_ids=available_skill_ids,
        detection_reason=_normalize_optional_text(contract.detection_reason),
    )


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_llm_text(value)
    return normalized or None


def _raise_for_unknown_selected_skill(
    contract: PromptContract,
    available_skill_ids: list[str],
) -> None:
    if contract.selected_skill_id is None:
        return
    if contract.selected_skill_id not in set(available_skill_ids):
        raise PromptContractDetectorError(
            "Prompt contract detector returned an unknown selected_skill_id: "
            f"{contract.selected_skill_id}"
        )


def _model_name_from_environment() -> str:
    model_name = os.environ.get("WHOSE_AGENT_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        model_name = DEFAULT_MODEL
    if not model_name.startswith("openrouter:"):
        raise PromptContractDetectorError(
            "WHOSE_AGENT_MODEL must use the openrouter:<model-name> provider string."
        )
    return model_name


__all__ = [
    "PROMPT_CONTRACT_MODEL_SETTINGS",
    "PromptContractDetectorError",
    "build_prompt_contract_detection_prompt",
    "detect_prompt_contract",
]
