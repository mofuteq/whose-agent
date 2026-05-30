from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


Substituted = Literal["instruction", "authority", "role", "model", "none"]
TraceSubstituted = Literal["instruction", "authority", "role", "model"]
FailureMode = Literal[
    "constraint_override",
    "unauthorized_autonomy",
    "protective_shutdown",
    "persona_hallucination",
    "none",
]
TraceFailureMode = Literal[
    "constraint_override",
    "unauthorized_autonomy",
    "protective_shutdown",
    "persona_hallucination",
]
Confidence = Literal["low", "medium", "high"]
ClassificationKind = Literal["in_scope", "out_of_scope"]

EXPECTED_FAILURE_BY_SUBSTITUTED: Final[dict[Substituted, FailureMode]] = {
    "instruction": "constraint_override",
    "authority": "unauthorized_autonomy",
    "role": "protective_shutdown",
    "model": "persona_hallucination",
    "none": "none",
}


class ScenarioTraceTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    divergence_point: str
    why_it_breaks_delegation: list[str]
    better_behavior: list[str]

    @field_validator("divergence_point")
    @classmethod
    def trim_yaml_block_terminal_newline(cls, value: str) -> str:
        return value.rstrip("\n")


class ScenarioCheckerTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checker_observed_bypass: bool
    substituted: Substituted
    failure_mode: FailureMode
    evidence: list[str]
    divergence_point: str | None
    confidence: Confidence

    @field_validator("divergence_point")
    @classmethod
    def trim_yaml_block_terminal_newline(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.rstrip("\n")

    @model_validator(mode="after")
    def validate_failure_mode_mapping(self) -> "ScenarioCheckerTemplate":
        expected_failure_mode = EXPECTED_FAILURE_BY_SUBSTITUTED[self.substituted]
        if self.failure_mode != expected_failure_mode:
            raise ValueError(
                "checker_template.failure_mode must match checker_template.substituted: "
                f"{self.substituted} -> {expected_failure_mode} "
                f"(got {self.failure_mode})"
            )
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    expected_substituted: Substituted
    failure_mode: FailureMode
    selected_skill_id: str | None = None
    principal_prompt: str
    principal_signal: str
    generation_instruction: str
    trace_template: ScenarioTraceTemplate | None = None
    checker_template: ScenarioCheckerTemplate | None = None

    @field_validator("principal_prompt", "principal_signal", "generation_instruction")
    @classmethod
    def trim_yaml_block_terminal_newline(cls, value: str) -> str:
        return value.rstrip("\n")

    @model_validator(mode="after")
    def validate_failure_mode_mapping(self) -> "Scenario":
        expected_failure_mode = EXPECTED_FAILURE_BY_SUBSTITUTED[self.expected_substituted]
        if self.failure_mode != expected_failure_mode:
            raise ValueError(
                "failure_mode must match expected_substituted: "
                f"{self.expected_substituted} -> {expected_failure_mode} "
                f"(got {self.failure_mode})"
            )
        if self.expected_substituted == "none":
            if self.generation_instruction != "":
                raise ValueError("none scenarios must have an empty generation_instruction")
            if self.checker_template is not None:
                raise ValueError("none scenarios must not define checker_template")
        if self.selected_skill_id is not None and self.checker_template is None:
            raise ValueError("selected_skill_id requires checker_template")
        if self.checker_template is not None and self.selected_skill_id is None:
            raise ValueError("checker_template requires selected_skill_id")
        return self


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    principal_signal: str
    substituted: Substituted
    classification: ClassificationKind
    reason: str


class PromptClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal_prompt: str
    principal_signal: str
    substituted: Substituted
    classification: ClassificationKind
    reason: str

    @field_validator("principal_prompt", "principal_signal", "reason")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def validate_classification_mapping(self) -> "PromptClassification":
        expected_classification: ClassificationKind = (
            "out_of_scope" if self.substituted == "none" else "in_scope"
        )
        if self.classification != expected_classification:
            raise ValueError(
                "classification must match substituted: "
                f"{self.substituted} -> {expected_classification} "
                f"(got {self.classification})"
            )
        return self


class Reflection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reflection_substituted: TraceSubstituted
    why_it_breaks_delegation: list[str]
    better_behavior: list[str]


class CheckerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    skill_id: str
    checker_observed_bypass: bool
    substituted: TraceSubstituted | Literal["none"]
    failure_mode: FailureMode
    evidence: list[str]
    divergence_point: str | None
    confidence: Confidence


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    substituted: TraceSubstituted
    failure_mode: TraceFailureMode
    principal_signal: str
    bad_response: str
    divergence_point: str
    why_it_breaks_delegation: list[str]
    better_behavior: list[str]
    reflection_substituted: TraceSubstituted
