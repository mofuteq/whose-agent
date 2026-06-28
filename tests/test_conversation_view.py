from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from whose_agent.conversation_view import MessageView, project_messages
from whose_agent.schemas import ConversationMessage


def test_project_messages_returns_frozen_tuple_view() -> None:
    messages = [
        ConversationMessage(
            message_id="msg_user",
            role="user",
            content="Principal request.",
        ),
        ConversationMessage(
            message_id="msg_assistant",
            role="assistant",
            content="Agent response.",
        ),
        ConversationMessage(
            message_id="msg_tool",
            role="tool",
            content="Tool output.",
        ),
        ConversationMessage(
            message_id="msg_system",
            role="system",
            content="System instruction.",
        ),
    ]

    view = project_messages(messages)

    assert isinstance(view, tuple)
    assert all(isinstance(message, MessageView) for message in view)
    assert [message.turn_index for message in view] == [1, 2, 3, 4]
    assert [message.message_id for message in view] == [
        "msg_user",
        "msg_assistant",
        "msg_tool",
        "msg_system",
    ]
    assert [message.speaker for message in view] == [
        "principal",
        "agent",
        "tool",
        "system",
    ]
    assert [message.content for message in view] == [
        "Principal request.",
        "Agent response.",
        "Tool output.",
        "System instruction.",
    ]

    with pytest.raises(FrozenInstanceError):
        view[0].content = "mutated"  # type: ignore[misc]
