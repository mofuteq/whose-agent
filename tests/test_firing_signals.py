from __future__ import annotations

from datetime import datetime

import pytest

from whose_agent.firing_signals import (
    FiringSignalConfigurationError,
    FiringSignals,
    QuotaSignal,
    production_firing_signals,
)


NON_HEAVY_TIME = datetime.fromisoformat("2026-01-01T12:00:00+09:00")


def _clock() -> datetime:
    return NON_HEAVY_TIME


def test_production_firing_signals_absent_env_has_no_quota() -> None:
    signals = production_firing_signals(
        iteration_used=2,
        clock=_clock,
        environ={},
    )

    assert signals == FiringSignals(time=NON_HEAVY_TIME, quota=None)


def test_production_firing_signals_whitespace_env_has_no_quota() -> None:
    signals = production_firing_signals(
        iteration_used=2,
        clock=_clock,
        environ={"WHOSE_AGENT_QUOTA_LIMIT": "   "},
    )

    assert signals == FiringSignals(time=NON_HEAVY_TIME, quota=None)


def test_production_firing_signals_env_limit_uses_actual_iteration_count() -> None:
    signals = production_firing_signals(
        iteration_used=2,
        clock=_clock,
        environ={"WHOSE_AGENT_QUOTA_LIMIT": "3"},
    )

    assert signals == FiringSignals(
        time=NON_HEAVY_TIME,
        quota=QuotaSignal(used=2, limit=3),
    )


@pytest.mark.parametrize("limit", ["0", "-1", "1.5", "abc"])
def test_production_firing_signals_invalid_env_limit_fails_clearly(
    limit: str,
) -> None:
    with pytest.raises(
        FiringSignalConfigurationError,
        match="WHOSE_AGENT_QUOTA_LIMIT",
    ):
        production_firing_signals(
            iteration_used=2,
            clock=_clock,
            environ={"WHOSE_AGENT_QUOTA_LIMIT": limit},
        )


@pytest.mark.parametrize("iteration_used", [-1, 1.5, True])
def test_production_firing_signals_requires_non_negative_integer_iteration_used(
    iteration_used: object,
) -> None:
    with pytest.raises(ValueError, match="iteration_used"):
        production_firing_signals(
            iteration_used=iteration_used,  # type: ignore[arg-type]
            clock=_clock,
            environ={"WHOSE_AGENT_QUOTA_LIMIT": "3"},
        )
