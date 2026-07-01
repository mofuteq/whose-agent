from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from whose_agent.history_adapter import (
    conversation_from_prompt,
    current_principal_prompt,
    require_unique_message_ids,
)
from whose_agent.prompt_loop_presets import load_prompt_loop_preset_by_id
from whose_agent.schemas import ConversationMessage, HistorySource


DEFAULT_PROMPT_LOOP_PRESETS_DIR = Path("prompt_loop_presets")


class PromptLoopSeedError(ValueError):
    pass


@dataclass(frozen=True)
class PromptLoopSeed:
    messages: list[ConversationMessage]
    current_principal_prompt: str
    history_source: HistorySource
    prompt_loop_preset_id: str | None
    prior_completed_agent_turns: int


def resolve_prompt_loop_seed(
    *,
    prompt: str | None = None,
    messages: Sequence[ConversationMessage] | None = None,
    preset_id: str | None = None,
    presets_dir: Path = DEFAULT_PROMPT_LOOP_PRESETS_DIR,
) -> PromptLoopSeed:
    provided_sources = sum(
        source is not None for source in (prompt, messages, preset_id)
    )
    if provided_sources != 1:
        raise PromptLoopSeedError(
            "prompt-loop input requires exactly one of prompt, messages, or preset_id"
        )

    if prompt is not None:
        canonical_messages = conversation_from_prompt(prompt)
        return PromptLoopSeed(
            messages=canonical_messages,
            current_principal_prompt=current_principal_prompt(canonical_messages),
            history_source="caller_supplied",
            prompt_loop_preset_id=None,
            prior_completed_agent_turns=0,
        )

    if messages is not None:
        canonical_messages = list(messages)
        require_unique_message_ids(canonical_messages)
        return PromptLoopSeed(
            messages=canonical_messages,
            current_principal_prompt=current_principal_prompt(canonical_messages),
            history_source="caller_supplied",
            prompt_loop_preset_id=None,
            prior_completed_agent_turns=0,
        )

    if preset_id is None:
        raise PromptLoopSeedError(
            "prompt-loop input requires exactly one of prompt, messages, or preset_id"
        )
    preset = load_prompt_loop_preset_by_id(presets_dir, preset_id)
    canonical_messages = preset.to_conversation_messages()
    return PromptLoopSeed(
        messages=canonical_messages,
        current_principal_prompt=current_principal_prompt(canonical_messages),
        history_source="server_owned_preset",
        prompt_loop_preset_id=preset.preset_id,
        prior_completed_agent_turns=preset.prior_completed_agent_turns,
    )


__all__ = [
    "DEFAULT_PROMPT_LOOP_PRESETS_DIR",
    "PromptLoopSeed",
    "PromptLoopSeedError",
    "resolve_prompt_loop_seed",
]
