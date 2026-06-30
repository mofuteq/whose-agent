"""External pressure signals for prompt-derived misreader firing."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime, time as ClockTime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field


HEAVY_TIME_WINDOWS: Final[tuple[tuple[ClockTime, ClockTime], ...]] = (
    (ClockTime(6, 0), ClockTime(9, 0)),
    (ClockTime(22, 0), ClockTime(2, 0)),
)
QUOTA_PRESSURE_THRESHOLD: Final[float] = 0.9
QUOTA_LIMIT_ENV_VAR: Final[str] = "WHOSE_AGENT_QUOTA_LIMIT"
_POSITIVE_WHOLE_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9]+$")
PromptFiringReason = Literal[
    "explicit_decision",
    "heavy_time",
    "quota_pressure",
    "heavy_time_and_quota_pressure",
    "no_pressure",
    "not_applicable",
]


class QuotaSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    used: float = Field(ge=0)
    limit: float = Field(gt=0)

    @property
    def ratio(self) -> float:
        return self.used / self.limit


class FiringSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: datetime | None = None
    quota: QuotaSignal | None = None


class FiringSignalOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: datetime | None = None
    quota: QuotaSignal | None = None


class PromptFiringEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_fire: bool
    reason: PromptFiringReason
    explicit_decision: bool | None
    firing_signals: FiringSignals | None


FiringSignalsInput = FiringSignals | Mapping[str, Any] | None


class FiringSignalConfigurationError(ValueError):
    """Invalid prompt-loop firing signal configuration."""


def production_firing_signals(
    *,
    iteration_used: int,
    clock: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
) -> FiringSignals:
    """Build production firing signals from real runtime inputs."""
    _validate_iteration_used(iteration_used)
    return FiringSignals(
        time=_current_time(clock),
        quota=_quota_signal_from_env(
            iteration_used=iteration_used,
            environ=environ,
        ),
    )


def resolve_firing_signals(
    *,
    iteration_used: int,
    clock: Callable[[], datetime] | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: FiringSignalOverrides | None = None,
) -> FiringSignals:
    """Resolve production signals with optional component-level overrides."""
    _validate_iteration_used(iteration_used)
    overrides = overrides or FiringSignalOverrides()
    if overrides.time is None and overrides.quota is None:
        return production_firing_signals(
            iteration_used=iteration_used,
            clock=clock,
            environ=environ,
        )

    return FiringSignals(
        time=overrides.time if overrides.time is not None else _current_time(clock),
        quota=(
            overrides.quota
            if overrides.quota is not None
            else _quota_signal_from_env(
                iteration_used=iteration_used,
                environ=environ,
            )
        ),
    )


def normalize_firing_signals(signals: FiringSignalsInput) -> FiringSignals | None:
    if signals is None:
        return None
    if isinstance(signals, FiringSignals):
        return signals
    return FiringSignals.model_validate(signals)


def external_pressure_active(signals: FiringSignalsInput) -> bool:
    normalized = normalize_firing_signals(signals)
    if normalized is None:
        return False
    return is_heavy_time(normalized.time) or is_quota_pressure(normalized.quota)


def is_heavy_time(moment: datetime | None) -> bool:
    if moment is None:
        return False
    current_time = moment.time()
    return any(
        _time_is_in_window(current_time, start, end)
        for start, end in HEAVY_TIME_WINDOWS
    )


def is_quota_pressure(quota: QuotaSignal | None) -> bool:
    if quota is None:
        return False
    return quota.ratio >= QUOTA_PRESSURE_THRESHOLD


def _validate_iteration_used(iteration_used: int) -> None:
    if (
        isinstance(iteration_used, bool)
        or not isinstance(iteration_used, int)
        or iteration_used < 0
    ):
        raise ValueError("iteration_used must be a non-negative integer")


def _current_time(clock: Callable[[], datetime] | None) -> datetime:
    return clock() if clock is not None else datetime.now().astimezone()


def _quota_signal_from_env(
    *,
    iteration_used: int,
    environ: Mapping[str, str] | None,
) -> QuotaSignal | None:
    source = os.environ if environ is None else environ
    raw_limit = source.get(QUOTA_LIMIT_ENV_VAR)
    if raw_limit is None or raw_limit.strip() == "":
        return None

    stripped_limit = raw_limit.strip()
    if not _POSITIVE_WHOLE_NUMBER_PATTERN.fullmatch(stripped_limit):
        raise FiringSignalConfigurationError(
            f"{QUOTA_LIMIT_ENV_VAR} must be a positive whole number of iterations"
        )
    limit = int(stripped_limit)
    if limit <= 0:
        raise FiringSignalConfigurationError(
            f"{QUOTA_LIMIT_ENV_VAR} must be a positive whole number of iterations"
        )
    return QuotaSignal(used=iteration_used, limit=limit)


def _time_is_in_window(
    current_time: ClockTime,
    start: ClockTime,
    end: ClockTime,
) -> bool:
    if start <= end:
        return start <= current_time < end
    return current_time >= start or current_time < end


__all__ = [
    "FiringSignalConfigurationError",
    "FiringSignalOverrides",
    "FiringSignals",
    "FiringSignalsInput",
    "HEAVY_TIME_WINDOWS",
    "PromptFiringEvaluation",
    "PromptFiringReason",
    "QUOTA_PRESSURE_THRESHOLD",
    "QUOTA_LIMIT_ENV_VAR",
    "QuotaSignal",
    "external_pressure_active",
    "is_heavy_time",
    "is_quota_pressure",
    "production_firing_signals",
    "resolve_firing_signals",
]
