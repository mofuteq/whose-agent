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
ClassificationKind = Literal["in_scope", "out_of_scope"]

EXPECTED_FAILURE_BY_SUBSTITUTED: Final[dict[Substituted, FailureMode]] = {
    "instruction": "constraint_override",
    "authority": "unauthorized_autonomy",
    "role": "protective_shutdown",
    "model": "persona_hallucination",
    "none": "none",
}


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    expected_substituted: Substituted
    failure_mode: FailureMode
    principal_prompt: str
    principal_signal: str
    generation_instruction: str

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
