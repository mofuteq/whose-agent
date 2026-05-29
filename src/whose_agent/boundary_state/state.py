from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from whose_agent.models import TraceFailureMode, TraceSubstituted


BoundaryNextAction = Literal["trace_ready", "review_reflection"]


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
