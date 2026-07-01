from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from whose_agent.history_adapter import normalize_role_tagged_messages
from whose_agent.schemas import ConversationMessage


ALLOWED_PRESET_FIELDS = {
    "preset_id",
    "display_title",
    "description",
    "prior_completed_agent_turns",
    "initial_messages",
}
ALLOWED_MESSAGE_FIELDS = {"role", "content"}


class PromptLoopPresetError(ValueError):
    pass


class PromptLoopPresetMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("role must be a string")
        return value.strip().casefold()

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("content must be a string")
        content = value.strip()
        if not content:
            raise ValueError("content must not be empty")
        return content


class PromptLoopPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    preset_id: str
    display_title: str
    description: str
    prior_completed_agent_turns: int = Field(ge=0)
    initial_messages: tuple[PromptLoopPresetMessage, ...] = Field(min_length=1)

    @field_validator("preset_id", "display_title", "description", mode="before")
    @classmethod
    def normalize_required_text(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        text = value.strip()
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("prior_completed_agent_turns", mode="before")
    @classmethod
    def require_declared_integer_count(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("prior_completed_agent_turns must be an integer")
        return value

    @model_validator(mode="after")
    def validate_message_shape_and_prior_count(self) -> "PromptLoopPreset":
        roles = [message.role for message in self.initial_messages]
        if roles[0] != "user":
            raise ValueError("initial_messages must begin with a user message")
        if roles[-1] != "user":
            raise ValueError("initial_messages must end with a user message")
        for index, role in enumerate(roles):
            expected = "user" if index % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError(
                    "initial_messages must alternate user then assistant"
                )
        assistant_turns = roles.count("assistant")
        if self.prior_completed_agent_turns != assistant_turns:
            raise ValueError(
                "prior_completed_agent_turns must equal the number of "
                "assistant messages"
            )
        return self

    def to_conversation_messages(self) -> list[ConversationMessage]:
        return normalize_role_tagged_messages(
            [
                {"role": message.role, "content": message.content}
                for message in self.initial_messages
            ]
        )


def load_prompt_loop_preset(path: Path) -> PromptLoopPreset:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as exc:
        raise PromptLoopPresetError(
            f"could not read prompt-loop preset {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise PromptLoopPresetError(
            f"Prompt-loop preset {path} must contain a YAML mapping."
        )
    _validate_raw_fixture_fields(data, path)
    try:
        return PromptLoopPreset.model_validate(data)
    except ValueError as exc:
        raise PromptLoopPresetError(
            f"Prompt-loop preset {path} is invalid: {exc}"
        ) from exc


def load_prompt_loop_presets(directory: Path) -> list[PromptLoopPreset]:
    paths = sorted(directory.glob("*.yaml"))
    presets: list[PromptLoopPreset] = []
    seen: dict[str, Path] = {}
    for path in paths:
        preset = load_prompt_loop_preset(path)
        previous = seen.get(preset.preset_id)
        if previous is not None:
            raise PromptLoopPresetError(
                "duplicate prompt-loop preset_id "
                f"{preset.preset_id!r} in {previous} and {path}"
            )
        seen[preset.preset_id] = path
        presets.append(preset)
    return presets


def load_prompt_loop_preset_by_id(
    directory: Path,
    preset_id: str,
) -> PromptLoopPreset:
    normalized_id = preset_id.strip()
    if not normalized_id:
        raise PromptLoopPresetError("prompt-loop preset id must not be empty")
    for preset in load_prompt_loop_presets(directory):
        if preset.preset_id == normalized_id:
            return preset
    raise PromptLoopPresetError(
        f"unknown prompt-loop preset {normalized_id!r} in {directory}"
    )


def _validate_raw_fixture_fields(data: dict[object, object], path: Path) -> None:
    extra_fields = sorted(str(key) for key in data if key not in ALLOWED_PRESET_FIELDS)
    if extra_fields:
        raise PromptLoopPresetError(
            f"Prompt-loop preset {path} contains forbidden fixture field "
            f"{extra_fields[0]!r}."
        )
    raw_messages = data.get("initial_messages")
    if not isinstance(raw_messages, list):
        return
    for index, message in enumerate(raw_messages, start=1):
        if not isinstance(message, dict):
            continue
        message_extra = sorted(
            str(key) for key in message if key not in ALLOWED_MESSAGE_FIELDS
        )
        if message_extra:
            raise PromptLoopPresetError(
                f"Prompt-loop preset {path} message {index} contains forbidden "
                f"fixture field {message_extra[0]!r}."
            )


__all__ = [
    "PromptLoopPreset",
    "PromptLoopPresetError",
    "PromptLoopPresetMessage",
    "load_prompt_loop_preset",
    "load_prompt_loop_preset_by_id",
    "load_prompt_loop_presets",
]
