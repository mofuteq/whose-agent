from __future__ import annotations

from typing import Literal

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
    def validate_none_consistency(self) -> "Scenario":
        if self.expected_substituted == "none":
            if self.failure_mode != "none":
                raise ValueError("none scenarios must use failure_mode: none")
            if self.generation_instruction != "":
                raise ValueError("none scenarios must have an empty generation_instruction")
        elif self.failure_mode == "none":
            raise ValueError("in-scope scenarios must use a concrete failure_mode")
        return self


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    principal_signal: str
    substituted: Substituted
    classification: ClassificationKind
    reason: str


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
