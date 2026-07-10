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
from whose_agent.schemas import (
    ConversationMessage,
    HistorySource,
    PromptLoopActorMode,
)


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
    actor_mode: PromptLoopActorMode | None


def resolve_prompt_loop_seed(
    *,
    prompt: str | None = None,
    messages: Sequence[ConversationMessage] | None = None,
    preset_id: str | None = None,
    presets_dir: Path = DEFAULT_PROMPT_LOOP_PRESETS_DIR,
) -> PromptLoopSeed:
    if messages is not None and (prompt is not None or preset_id is not None):
        raise PromptLoopSeedError(
            "caller-supplied messages cannot be combined with prompt or preset_id"
        )
    if preset_id is not None and prompt is None:
        raise PromptLoopSeedError("preset_id requires a current prompt")
    if prompt is None and messages is None:
        raise PromptLoopSeedError(
            "prompt-loop input requires prompt, messages, or preset_id plus prompt"
        )

    if prompt is not None and preset_id is None:
        canonical_messages = conversation_from_prompt(prompt)
        return PromptLoopSeed(
            messages=canonical_messages,
            current_principal_prompt=current_principal_prompt(canonical_messages),
            history_source="caller_supplied",
            prompt_loop_preset_id=None,
            prior_completed_agent_turns=0,
            actor_mode=None,
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
            actor_mode=None,
        )

    if preset_id is None or prompt is None:
        raise PromptLoopSeedError(
            "prompt-loop input requires prompt, messages, or preset_id plus prompt"
        )
    preset = load_prompt_loop_preset_by_id(presets_dir, preset_id)
    current_user_message = conversation_from_prompt(prompt)[0]
    canonical_messages = [
        *preset.to_conversation_messages(),
        current_user_message,
    ]
    return PromptLoopSeed(
        messages=canonical_messages,
        current_principal_prompt=current_principal_prompt(canonical_messages),
        history_source="server_owned_preset",
        prompt_loop_preset_id=preset.preset_id,
        prior_completed_agent_turns=preset.prior_completed_agent_turns,
        actor_mode=preset.actor_mode,
    )


__all__ = [
    "DEFAULT_PROMPT_LOOP_PRESETS_DIR",
    "PromptLoopSeed",
    "PromptLoopSeedError",
    "resolve_prompt_loop_seed",
]
