from __future__ import annotations

from collections.abc import Iterable

from whose_agent.conversation_view import ConversationView
from whose_agent.schemas import SelfExplanation


RAW_HISTORY_LEAKAGE_ERROR = "self_explanation_unavailable:RawHistoryLeakage"
MIN_HISTORY_LEAKAGE_CHARS = 12


def normalize_for_history_leakage(text: str) -> str:
    return " ".join(text.casefold().split())


def public_safe_self_explanation(
    candidate: SelfExplanation,
    *,
    history: ConversationView,
) -> SelfExplanation:
    if _contains_raw_history_leakage(candidate, history=history):
        return SelfExplanation(status="unavailable")
    return candidate


def _contains_raw_history_leakage(
    candidate: SelfExplanation,
    *,
    history: ConversationView,
) -> bool:
    if candidate.status != "provided":
        return False

    normalized_turns = tuple(_normalized_leak_candidates(history))
    if not normalized_turns:
        return False

    for field_value in (
        candidate.action_or_adaptation_summary,
        candidate.treated_as_sufficient_basis,
        candidate.rationale_summary,
        candidate.checker_acknowledgement,
    ):
        if field_value is None:
            continue
        normalized_field = normalize_for_history_leakage(field_value)
        if any(turn in normalized_field for turn in normalized_turns):
            return True
    return False


def _normalized_leak_candidates(history: ConversationView) -> Iterable[str]:
    for message in history:
        if len("".join(message.content.split())) < MIN_HISTORY_LEAKAGE_CHARS:
            continue
        normalized = normalize_for_history_leakage(message.content)
        if normalized:
            yield normalized


__all__ = [
    "MIN_HISTORY_LEAKAGE_CHARS",
    "RAW_HISTORY_LEAKAGE_ERROR",
    "normalize_for_history_leakage",
    "public_safe_self_explanation",
]
