from __future__ import annotations

from operator import add
from typing import Annotated, Final, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from whose_agent.firing_signals import FiringSignals, PromptFiringReason


Principal = str
AgentId = str
ConversationRole = Literal["user", "assistant", "tool", "system"]
StepKind = Literal["plan", "do", "check", "explain"]
NextAction = Literal["continue", "stop", "handoff"]
SubstitutionAxis = Literal["instruction", "authority", "role", "model"]
Substituted = Literal["instruction", "authority", "role", "model", "none"]
TraceSubstituted = SubstitutionAxis
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
    "matched_no_boundary_event",
    "checker_missed_boundary_event",
    "checker_over_detected",
    "not_applicable",
]
LoopSource = Literal["fixed_scenario", "prompt_contract"]
PromptContractStatus = Literal[
    "contract_detected",
    "no_contract_detected",
    "unsupported",
]
ExternalActionKind = Literal["external_persistence"]
AuthorityGrantStatus = Literal[
    "not_granted",
    "granted",
    "no_agent_proposal",
    "target_mismatch",
    "no_action_attempt",
]
AuthorityResult = Literal[
    "self_originated_delegation_laundering",
    "authorized",
    "not_applicable",
]
ExplanationStatus = Literal["provided", "refused", "unavailable"]

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
    boundary_detected: bool = False
    framework_specified: bool = False
    guarantee_bypass_observed: bool = False
    guarantee_bypass_evidence: list[str] = Field(default_factory=list)
    checker_id: str | None = None
    checker_confidence: str | None = None
    boundary_flags: list[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    role: ConversationRole
    content: str

    @field_validator("message_id", mode="before")
    @classmethod
    def require_message_id(cls, value: str) -> str:
        message_id = str(value).strip()
        if not message_id:
            raise ValueError("message_id must not be empty")
        return message_id

    @field_validator("content", mode="before")
    @classmethod
    def require_content(cls, value: str) -> str:
        content = str(value).strip()
        if not content:
            raise ValueError("content must not be empty")
        return content


class ExternalPersistenceActionAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_kind: ExternalActionKind = "external_persistence"
    target: str = Field(max_length=80)

    @field_validator("target", mode="before")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        target = str(value).strip().casefold()
        if not target:
            raise ValueError("target must not be empty")
        return target


class AuthorityProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_kind: ExternalActionKind = "external_persistence"
    target: str = Field(max_length=80)
    prior_agent_proposal_turn: int | None = None
    principal_grant_turn: int | None = None
    grant_status: AuthorityGrantStatus
    action_attempt_turn: int | None = None
    result: AuthorityResult = "not_applicable"

    @field_validator("target", mode="before")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        target = str(value).strip().casefold()
        if not target:
            raise ValueError("target must not be empty")
        return target

    @model_validator(mode="after")
    def validate_turn_indexes(self) -> "AuthorityProvenance":
        for field_name in (
            "prior_agent_proposal_turn",
            "principal_grant_turn",
            "action_attempt_turn",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be a 1-based turn index")
        if (
            self.result == "self_originated_delegation_laundering"
            and self.grant_status != "not_granted"
        ):
            raise ValueError(
                "self_originated_delegation_laundering requires grant_status=not_granted"
            )
        if self.result == "authorized" and self.grant_status != "granted":
            raise ValueError("authorized requires grant_status=granted")
        return self


class AuthorityCauseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provenance: AuthorityProvenance
    action_attempt: ExternalPersistenceActionAttempt | None = None
    drift_fired: bool
    trigger_evidence: tuple[str, ...] = Field(default_factory=tuple)


class SelfExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ExplanationStatus
    action_or_adaptation_summary: str | None = Field(
        default=None,
        max_length=300,
    )
    treated_as_sufficient_basis: str | None = Field(
        default=None,
        max_length=300,
    )
    relied_on_turn_indexes: tuple[int, ...] = Field(default_factory=tuple)
    rationale_summary: str | None = Field(
        default=None,
        max_length=500,
    )
    checker_acknowledgement: str | None = Field(
        default=None,
        max_length=300,
    )

    @field_validator(
        "action_or_adaptation_summary",
        "treated_as_sufficient_basis",
        "rationale_summary",
        "checker_acknowledgement",
        mode="before",
    )
    @classmethod
    def normalize_optional_explanation_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("relied_on_turn_indexes")
    @classmethod
    def validate_relied_on_turn_indexes(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(index < 1 for index in value):
            raise ValueError("relied_on_turn_indexes must be 1-based")
        if len(value) != len(set(value)):
            raise ValueError("relied_on_turn_indexes must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("relied_on_turn_indexes must be ascending")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self) -> "SelfExplanation":
        explanatory_fields = (
            self.action_or_adaptation_summary,
            self.treated_as_sufficient_basis,
            self.rationale_summary,
            self.checker_acknowledgement,
        )
        if self.status == "provided":
            if any(field is None for field in explanatory_fields):
                raise ValueError("provided self_explanation requires all summaries")
            if not self.relied_on_turn_indexes:
                raise ValueError(
                    "provided self_explanation requires relied_on_turn_indexes"
                )
            return self

        if any(field is not None for field in explanatory_fields):
            raise ValueError(
                "refused and unavailable self_explanation must not include summaries"
            )
        if self.relied_on_turn_indexes:
            raise ValueError(
                "refused and unavailable self_explanation must not include turn indexes"
            )
        return self


class AuthorityCheckerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_kind: ExternalActionKind = "external_persistence"
    target: str = Field(max_length=80)
    prior_agent_proposal_turn: int | None = None
    principal_grant_turn: int | None = None
    generated_action_attempt_turn: int | None = None

    @field_validator("target", mode="before")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        target = str(value).strip().casefold()
        if not target:
            raise ValueError("target must not be empty")
        return target


class StepTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int
    step_kind: StepKind
    principal: Principal
    agent: AgentId
    misreader_skill_fired: bool = False
    selected_skill_id: str | None = None
    generation_used_skill: bool = False
    generation_skill_id: str | None = None
    checker_ran: bool = False
    checker_observed_bypass: bool = False
    trigger_evidence: list[str] = Field(default_factory=list)
    authority_provenance: AuthorityProvenance | None = None
    drift_evidence: str | None = Field(default=None, max_length=300)
    drift_artifact_kind: str | None = Field(default=None, max_length=80)
    substituted: TraceSubstituted | None = None
    boundary_flags: list[str] = Field(default_factory=list)
    divergence_point: str | None = None

    @field_validator(
        "generation_skill_id",
        "drift_evidence",
        "drift_artifact_kind",
        mode="before",
    )
    @classmethod
    def trim_optional_step_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None


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
    initial_messages: list[dict[str, object]] = Field(default_factory=list)
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


class PromptContract(BaseModel):
    """Detected arbitrary-prompt boundary contract.

    Status semantics intentionally separate a supported contract from a prompt
    that names a boundary but has no applicable skill perspective.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    boundary_detected: bool
    substitution_axis: SubstitutionAxis | None = None
    delegated_boundary: str | None = Field(default=None, max_length=500)
    framework_specified: bool
    candidate_framework: str | None = Field(default=None, max_length=200)
    delegated_guarantee: str | None = Field(default=None, max_length=500)
    selected_skill_id: str | None = Field(default=None, max_length=128)
    skill_selection_reason: str | None = Field(default=None, max_length=1000)
    confidence: Confidence
    status: PromptContractStatus
    available_skill_ids: list[str] = Field(default_factory=list)
    detection_reason: str | None = Field(default=None, max_length=1000)

    @field_validator(
        "candidate_framework",
        "delegated_boundary",
        "delegated_guarantee",
        "selected_skill_id",
        "skill_selection_reason",
        "detection_reason",
        mode="before",
    )
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("prompt", mode="before")
    @classmethod
    def require_prompt(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("prompt must not be empty")
        value = str(value).strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value

    @field_validator("available_skill_ids")
    @classmethod
    def require_non_empty_skill_ids(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_status_consistency(self) -> "PromptContract":
        if self.status == "contract_detected":
            if not self.boundary_detected:
                raise ValueError("contract_detected requires boundary_detected=true")
            if self.substitution_axis is None:
                raise ValueError("contract_detected requires substitution_axis")
            if self.delegated_boundary is None:
                raise ValueError("contract_detected requires delegated_boundary")
            if self.selected_skill_id is None:
                raise ValueError("contract_detected requires selected_skill_id")
            if self.skill_selection_reason is None:
                raise ValueError("contract_detected requires skill_selection_reason")
        if self.status == "no_contract_detected":
            if self.boundary_detected:
                raise ValueError("no_contract_detected requires boundary_detected=false")
            if self.substitution_axis is not None:
                raise ValueError("no_contract_detected requires substitution_axis=null")
            if self.delegated_boundary is not None:
                raise ValueError("no_contract_detected requires delegated_boundary=null")
            if self.framework_specified:
                raise ValueError("no_contract_detected requires framework_specified=false")
            if self.selected_skill_id is not None:
                raise ValueError("no_contract_detected requires selected_skill_id=null")
            if self.skill_selection_reason is not None:
                raise ValueError("no_contract_detected requires skill_selection_reason=null")
            if self.candidate_framework is not None:
                raise ValueError("no_contract_detected requires candidate_framework=null")
            if self.delegated_guarantee is not None:
                raise ValueError("no_contract_detected requires delegated_guarantee=null")
        if self.status == "unsupported":
            if not self.boundary_detected:
                raise ValueError("unsupported requires boundary_detected=true")
            if self.selected_skill_id is not None:
                raise ValueError("unsupported requires selected_skill_id=null")
            if self.skill_selection_reason is not None:
                raise ValueError("unsupported requires skill_selection_reason=null")
            if self.delegated_boundary is not None and self.substitution_axis is None:
                raise ValueError(
                    "unsupported with delegated_boundary requires substitution_axis"
                )
        return self


class Reflection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reflection_substituted: TraceSubstituted
    why_it_breaks_delegation: list[str]
    better_behavior: list[str]


class CheckerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    skill_id: str
    checker_observed_bypass: bool
    substituted: TraceSubstituted | Literal["none"]
    failure_mode: FailureMode
    evidence: tuple[str, ...] = Field(default_factory=tuple)
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
    authority_provenance: AuthorityProvenance | None = None
    principal_signal: str
    bad_response: str
    divergence_point: str
    why_it_breaks_delegation: list[str]
    better_behavior: list[str]
    reflection_substituted: TraceSubstituted
    self_explanation: SelfExplanation | None = None


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


class LoopTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    loop_source: LoopSource = "fixed_scenario"
    boundary_detected: bool = False
    substitution_axis: SubstitutionAxis | None = None
    delegated_boundary: str | None = None
    prompt_contract_status: PromptContractStatus | None = None
    prompt_contract_boundary_detected: bool | None = None
    prompt_contract_substitution_axis: SubstitutionAxis | None = None
    prompt_contract_delegated_boundary: str | None = None
    prompt_contract_candidate_framework: str | None = None
    prompt_contract_delegated_guarantee: str | None = None
    prompt_contract_artifact: str | None = None
    prompt_loop_generated_artifact: str | None = None
    prompt_loop_generated_step_index: int | None = None
    authority_provenance: AuthorityProvenance | None = None
    principal: str
    agent: str
    max_iterations: int
    final_loop_iteration: int
    loop_completed: bool
    loop_stop_reason: str | None
    framework_specified: bool
    selected_skill_id: str | None
    misreader_firing_decision: bool | None = None
    firing_signals: FiringSignals | None = None
    firing_reason: PromptFiringReason | None = None
    generation_used_skill: bool
    checker_ran: bool
    checker_observed_bypass: bool
    guarantee_bypass_observed: bool
    checker_matches_expected: bool | None
    observation_outcome: ObservationOutcome | None
    step_traces: list[StepTrace]
    checker_comparison: CheckerComparison | None
    self_explanation: SelfExplanation | None = None


class WhoseAgentState(TypedDict, total=False):
    principal: str
    agent: str
    messages: list[ConversationMessage]
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
    self_explanation: SelfExplanation | None

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
    loop_source: LoopSource
    boundary_detected: bool
    substitution_axis: SubstitutionAxis | None
    delegated_boundary: str | None
    prompt_contract_status: PromptContractStatus | None
    prompt_contract_boundary_detected: bool | None
    prompt_contract_substitution_axis: SubstitutionAxis | None
    prompt_contract_delegated_boundary: str | None
    prompt_contract_candidate_framework: str | None
    prompt_contract_delegated_guarantee: str | None
    prompt_contract_artifact: str | None
    prompt_loop_generated_artifact: str | None
    prompt_loop_generated_step_index: int | None
    authority_provenance: AuthorityProvenance | None
    authority_cause_record: AuthorityCauseRecord | None

    selected_skill_id: str | None
    selected_skill_perspective: str | None
    misreader_firing_decision: bool | None
    firing_signals: FiringSignals | None
    firing_reason: PromptFiringReason | None
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
    "AuthorityCauseRecord",
    "AuthorityCheckerContext",
    "AuthorityGrantStatus",
    "AuthorityProvenance",
    "AuthorityResult",
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
    "ConversationMessage",
    "ConversationRole",
    "EXPECTED_FAILURE_BY_SUBSTITUTED",
    "ExternalActionKind",
    "ExternalPersistenceActionAttempt",
    "FAILURE_MODES",
    "FailureMode",
    "ExplanationStatus",
    "LoopSource",
    "LoopTrace",
    "NextAction",
    "ObservationOutcome",
    "Principal",
    "PromptContract",
    "PromptContractStatus",
    "Reflection",
    "Scenario",
    "ScenarioCheckerTemplate",
    "ScenarioTraceTemplate",
    "SelfExplanation",
    "StepKind",
    "SubstitutionAxis",
    "StepTrace",
    "Substituted",
    "TRACE_FAILURE_MODES",
    "Trace",
    "TraceFailureMode",
    "TraceSubstituted",
    "WhoseAgentState",
]
