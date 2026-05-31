from __future__ import annotations

from operator import add
from typing import Annotated, Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Principal = str
AgentId = str
StepKind = Literal["plan", "do", "check"]
NextAction = Literal["continue", "stop", "handoff"]
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
BoundaryNextAction = Literal["trace_ready", "review_reflection"]
ObservationOutcome = Literal[
    "observation_succeeded",
    "checker_missed_boundary_event",
    "checker_over_detected",
    "not_applicable",
]

FAILURE_MODES: Final[tuple[FailureMode, ...]] = (
    "constraint_override",
    "unauthorized_autonomy",
    "protective_shutdown",
    "persona_hallucination",
    "none",
)
TRACE_FAILURE_MODES: Final[tuple[TraceFailureMode, ...]] = (
    "constraint_override",
    "unauthorized_autonomy",
    "protective_shutdown",
    "persona_hallucination",
)
EXPECTED_FAILURE_BY_SUBSTITUTED: Final[dict[Substituted, FailureMode]] = {
    "instruction": "constraint_override",
    "authority": "unauthorized_autonomy",
    "role": "protective_shutdown",
    "model": "persona_hallucination",
    "none": "none",
}


class ControlState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: Principal
    agent: AgentId
    principal_instruction: str
    principal_signal: str
    step_kind: StepKind
    step_index: int
    next_action: NextAction | None = None
    handoff_ready: bool = False
    selected_skill_id: str | None = None
    selected_skill_perspective: str | None = None
    framework_specified: bool = False
    guarantee_bypass_observed: bool = False
    guarantee_bypass_evidence: list[str] = Field(default_factory=list)
    checker_id: str | None = None
    checker_confidence: str | None = None
    boundary_flags: list[str] = Field(default_factory=list)


class StepTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    step_kind: StepKind
    principal: Principal
    agent: AgentId
    misreader_skill_fired: bool = False
    selected_skill_id: str | None = None
    checker_ran: bool = False
    checker_observed_bypass: bool = False
    trigger_evidence: list[str] = Field(default_factory=list)
    substituted: TraceSubstituted | None = None
    boundary_flags: list[str] = Field(default_factory=list)
    divergence_point: str | None = None


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


class CheckerComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    expected_checker_observed_bypass: bool | None
    actual_checker_observed_bypass: bool | None
    expected_substituted: Substituted | None
    actual_substituted: Substituted | None
    expected_failure_mode: FailureMode | None
    actual_failure_mode: FailureMode | None
    matches_expected: bool
    mismatch_reasons: list[str] = Field(default_factory=list)
    observation_outcome: ObservationOutcome


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


class BoundaryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    principal_prompt: str
    principal_signal: str
    expected_substituted: TraceSubstituted
    failure_mode: TraceFailureMode
    bad_response: str | None = None
    reflection_substituted: TraceSubstituted | None = None
    reflection_matches_expected: bool | None = None
    boundary_flags: list[TraceFailureMode] = Field(default_factory=list)
    why_it_breaks_delegation: list[str] = Field(default_factory=list)
    better_behavior: list[str] = Field(default_factory=list)
    next_action: BoundaryNextAction | None = None


class BoundaryStateTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    state: BoundaryState


class BoundaryStateTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    transitions: list[BoundaryStateTransition]


class WhoseAgentState(TypedDict, total=False):
    principal: str
    agent: str
    principal_instruction: str
    principal_signal: str

    scenario: Scenario
    classification: Classification | None
    bad_response: str | None
    generation_used_skill: bool
    generation_skill_id: str | None
    trace: Trace | None
    state_trace: BoundaryStateTrace | None
    checker_observation: CheckerObservation | None
    checker_comparison: CheckerComparison | None

    step_kind: StepKind
    step_index: int
    next_action: NextAction | None
    handoff_ready: bool
    completed: bool

    # Minimal loop fields. LangGraph state is the runtime source of truth; these
    # extend WhoseAgentState directly rather than wiring ControlState as a nested
    # runtime object. They support a minimal plan -> do -> check loop path.
    loop_iteration: int
    loop_phase: StepKind
    max_iterations: int
    framework_specified: bool
    loop_completed: bool
    loop_stop_reason: str | None

    selected_skill_id: str | None
    selected_skill_perspective: str | None
    skill_triggered: bool
    misreader_skill_fired: bool
    trigger_evidence: Annotated[list[str], add]

    checker_ran: bool
    checker_observed_bypass: bool
    checker_id: str | None
    checker_confidence: str | None
    checker_matches_expected: bool | None
    observation_outcome: ObservationOutcome | None
    guarantee_bypass_observed: bool
    guarantee_bypass_evidence: Annotated[list[str], add]

    substituted: Substituted | None
    failure_mode: FailureMode | None
    divergence_point: str | None
    boundary_flags: Annotated[list[str], add]

    step_traces: Annotated[list[StepTrace], add]
    errors: Annotated[list[str], add]


__all__ = [
    "AgentId",
    "BoundaryNextAction",
    "BoundaryState",
    "BoundaryStateTrace",
    "BoundaryStateTransition",
    "CheckerComparison",
    "CheckerObservation",
    "Classification",
    "ClassificationKind",
    "Confidence",
    "ControlState",
    "EXPECTED_FAILURE_BY_SUBSTITUTED",
    "FAILURE_MODES",
    "FailureMode",
    "NextAction",
    "ObservationOutcome",
    "Principal",
    "PromptClassification",
    "Reflection",
    "Scenario",
    "ScenarioCheckerTemplate",
    "ScenarioTraceTemplate",
    "StepKind",
    "StepTrace",
    "Substituted",
    "TRACE_FAILURE_MODES",
    "Trace",
    "TraceFailureMode",
    "TraceSubstituted",
    "WhoseAgentState",
]
