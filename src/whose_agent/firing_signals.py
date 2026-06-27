"""External pressure signals for prompt-derived misreader firing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, time as ClockTime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field


HEAVY_TIME_WINDOWS: Final[tuple[tuple[ClockTime, ClockTime], ...]] = (
    (ClockTime(6, 0), ClockTime(9, 0)),
    (ClockTime(22, 0), ClockTime(2, 0)),
)
QUOTA_PRESSURE_THRESHOLD: Final[float] = 0.9
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


class PromptFiringEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_fire: bool
    reason: PromptFiringReason
    explicit_decision: bool | None
    firing_signals: FiringSignals | None


FiringSignalsInput = FiringSignals | Mapping[str, Any] | None


def production_firing_signals(
    *,
    clock: Callable[[], datetime] | None = None,
) -> FiringSignals:
    """Build production firing signals from real runtime inputs."""
    now = clock() if clock is not None else datetime.now().astimezone()
    return FiringSignals(time=now, quota=None)


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


def _time_is_in_window(
    current_time: ClockTime,
    start: ClockTime,
    end: ClockTime,
) -> bool:
    if start <= end:
        return start <= current_time < end
    return current_time >= start or current_time < end


__all__ = [
    "FiringSignals",
    "FiringSignalsInput",
    "HEAVY_TIME_WINDOWS",
    "PromptFiringEvaluation",
    "PromptFiringReason",
    "QUOTA_PRESSURE_THRESHOLD",
    "QuotaSignal",
    "external_pressure_active",
    "is_heavy_time",
    "is_quota_pressure",
    "production_firing_signals",
]
