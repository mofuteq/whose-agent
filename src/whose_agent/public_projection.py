from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from whose_agent.history_adapter import initial_conversation_messages
from whose_agent.prompt_loop_presets import PromptLoopPreset
from whose_agent.schemas import (
    AuthorityCauseRecord,
    AuthorityProvenance,
    CheckerObservation,
    ConversationRole,
    FailureMode,
    ObservationOutcome,
    Scenario,
    SelfExplanation,
    StepKind,
    Substituted,
    WhoseAgentState,
)


RunMode = Literal["fixed", "prompt_loop"]
RunStatus = Literal["in_progress", "completed", "failed", "cancelled"]
SafeErrorCode = Literal[
    "invalid_request",
    "unknown_scenario",
    "run_failed",
    "stream_cancelled",
]


class ScenarioPreviewMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ConversationRole
    content: str


class ScenarioDisplayProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    preview_messages: list[ScenarioPreviewMessage] = Field(default_factory=list)


class ScenarioMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    display_title: str
    selected_skill_id: str | None = None
    substitution_axis: Substituted
    description: str
    display: ScenarioDisplayProjection


class PromptLoopPresetMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset_id: str
    display_title: str
    description: str
    prior_completed_agent_turns: int
    preview_messages: list[ScenarioPreviewMessage] = Field(default_factory=list)
    suggested_next_prompt: str


class PhaseProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: StepKind


class AuthorityProvenanceProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_kind: str
    target: str
    prior_agent_proposal_turn: int | None = None
    principal_grant_turn: int | None = None
    grant_status: str
    action_attempt_turn: int | None = None
    result: str


class ActionAttemptSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_kind: str
    target: str
    attempted: bool


class CauseProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    misreader_skill_fired: bool
    selected_skill_id: str | None = None
    authority_provenance: AuthorityProvenanceProjection | None = None
    action_attempt_summary: ActionAttemptSummary | None = None


class CheckerProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checker_ran: bool
    checker_observed_bypass: bool
    substituted: Substituted | None = None
    failure_mode: FailureMode | None = None
    confidence: str | None = None
    divergence_summary: str | None = Field(default=None, max_length=500)


class ExplainProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    action_or_adaptation_summary: str | None = None
    treated_as_sufficient_basis: str | None = None
    relied_on_turn_indexes: tuple[int, ...] = ()
    rationale_summary: str | None = None
    checker_acknowledgement: str | None = None


class CompletedProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["completed"] = "completed"
    mode: RunMode
    selected_skill_id: str | None = None
    observation_outcome: ObservationOutcome | None = None
    artifact_names: list[str] = Field(default_factory=list)


class RunProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    thread_id: str
    status: RunStatus
    mode: RunMode
    result: CompletedProjection | None = None
    artifact_names: list[str] = Field(default_factory=list)
    safe_error_code: SafeErrorCode | None = None


def project_scenario_metadata(scenario: Scenario) -> ScenarioMetadata:
    selected_skill_id = scenario.selected_skill_id
    skill_label = selected_skill_id if selected_skill_id is not None else "no selected skill"
    display = project_scenario_display(scenario)
    return ScenarioMetadata(
        scenario_id=scenario.scenario_id,
        display_title=display.title,
        selected_skill_id=selected_skill_id,
        substitution_axis=scenario.expected_substituted,
        description=(
            f"{scenario.expected_substituted} scenario with {skill_label}; "
            f"failure mode {scenario.failure_mode}."
        ),
        display=display,
    )


def project_scenario_display(scenario: Scenario) -> ScenarioDisplayProjection:
    preview_messages = initial_conversation_messages(
        scenario.initial_messages,
        prompt=scenario.principal_prompt,
    )
    return ScenarioDisplayProjection(
        title=scenario.display_title or _title_from_scenario_id(scenario.scenario_id),
        preview_messages=[
            ScenarioPreviewMessage(role=message.role, content=message.content)
            for message in preview_messages
        ],
    )


def project_prompt_loop_preset_metadata(
    preset: PromptLoopPreset,
) -> PromptLoopPresetMetadata:
    return PromptLoopPresetMetadata(
        preset_id=preset.preset_id,
        display_title=preset.display_title,
        description=preset.description,
        prior_completed_agent_turns=preset.prior_completed_agent_turns,
        suggested_next_prompt=preset.suggested_next_prompt,
        preview_messages=[
            ScenarioPreviewMessage(role=message.role, content=message.content)
            for message in preset.initial_messages
        ],
    )


def project_authority_provenance(
    provenance: AuthorityProvenance | None,
) -> AuthorityProvenanceProjection | None:
    if provenance is None:
        return None
    return AuthorityProvenanceProjection(
        action_kind=provenance.action_kind,
        target=provenance.target,
        prior_agent_proposal_turn=provenance.prior_agent_proposal_turn,
        principal_grant_turn=provenance.principal_grant_turn,
        grant_status=provenance.grant_status,
        action_attempt_turn=provenance.action_attempt_turn,
        result=provenance.result,
    )


def project_cause(state: WhoseAgentState) -> CauseProjection:
    cause_record = state.get("authority_cause_record")
    provenance = _authority_provenance_from_state(state, cause_record)
    action_attempt_summary = None
    if provenance is not None:
        action_attempt_summary = ActionAttemptSummary(
            action_kind=provenance.action_kind,
            target=provenance.target,
            attempted=(
                cause_record.action_attempt is not None
                if cause_record is not None
                else provenance.action_attempt_turn is not None
            ),
        )
    return CauseProjection(
        misreader_skill_fired=bool(state.get("misreader_skill_fired", False)),
        selected_skill_id=state.get("selected_skill_id"),
        authority_provenance=project_authority_provenance(provenance),
        action_attempt_summary=action_attempt_summary,
    )


def project_checker(state: WhoseAgentState) -> CheckerProjection:
    observation = state.get("checker_observation")
    return CheckerProjection(
        checker_ran=bool(state.get("checker_ran", False)),
        checker_observed_bypass=bool(state.get("checker_observed_bypass", False)),
        substituted=_checker_substituted(observation, state),
        failure_mode=_checker_failure_mode(observation, state),
        confidence=(
            observation.confidence
            if observation is not None
            else state.get("checker_confidence")
        ),
        divergence_summary=(
            observation.divergence_point if observation is not None else None
        ),
    )


def project_explain(explanation: SelfExplanation) -> ExplainProjection:
    return ExplainProjection(
        status=explanation.status,
        action_or_adaptation_summary=explanation.action_or_adaptation_summary,
        treated_as_sufficient_basis=explanation.treated_as_sufficient_basis,
        relied_on_turn_indexes=explanation.relied_on_turn_indexes,
        rationale_summary=explanation.rationale_summary,
        checker_acknowledgement=explanation.checker_acknowledgement,
    )


def project_completed(
    *,
    run_id: str,
    mode: RunMode,
    state: WhoseAgentState,
    artifact_names: list[str],
) -> CompletedProjection:
    return CompletedProjection(
        run_id=run_id,
        mode=mode,
        selected_skill_id=state.get("selected_skill_id"),
        observation_outcome=state.get("observation_outcome"),
        artifact_names=artifact_names,
    )


def _title_from_scenario_id(scenario_id: str) -> str:
    return " ".join(
        segment.capitalize()
        for segment in scenario_id.split("_")
        if segment
    )


def _authority_provenance_from_state(
    state: WhoseAgentState,
    cause_record: AuthorityCauseRecord | None,
) -> AuthorityProvenance | None:
    if cause_record is not None:
        return cause_record.provenance
    return state.get("authority_provenance")


def _checker_substituted(
    observation: CheckerObservation | None,
    state: WhoseAgentState,
) -> Substituted | None:
    if observation is not None:
        return observation.substituted
    return state.get("substituted")


def _checker_failure_mode(
    observation: CheckerObservation | None,
    state: WhoseAgentState,
) -> FailureMode | None:
    if observation is not None:
        return observation.failure_mode
    return state.get("failure_mode")
