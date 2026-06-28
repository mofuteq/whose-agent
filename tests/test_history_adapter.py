from __future__ import annotations

import json
from pathlib import Path

import pytest

from whose_agent.history_adapter import (
    MessageHistoryError,
    append_assistant_message,
    load_message_history_file,
    normalize_role_tagged_messages,
)


def test_role_content_messages_receive_generated_ids() -> None:
    messages = normalize_role_tagged_messages(
        [
            {"role": "user", "content": "Summarize this."},
            {"role": "assistant", "content": "I can save it later."},
        ]
    )

    ids = [message.message_id for message in messages]
    assert all(ids)
    assert len(ids) == len(set(ids))
    assert [message.role for message in messages] == ["user", "assistant"]


def test_messages_file_without_ids_receives_generated_ids(tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.json"
    messages_path.write_text(
        json.dumps(
            [
                {"role": "user", "content": "Summarize this."},
                {"role": "assistant", "content": "I can save it later."},
            ]
        ),
        encoding="utf-8",
    )

    messages = load_message_history_file(messages_path)

    ids = [message.message_id for message in messages]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_optional_input_message_ids_are_preserved() -> None:
    messages = normalize_role_tagged_messages(
        [
            {
                "message_id": "msg_input_user",
                "role": "user",
                "content": "Summarize this.",
            },
            {
                "message_id": "msg_input_agent",
                "role": "assistant",
                "content": "I can save it later.",
            },
        ]
    )

    assert [message.message_id for message in messages] == [
        "msg_input_user",
        "msg_input_agent",
    ]


def test_duplicate_input_message_ids_fail_clearly() -> None:
    with pytest.raises(MessageHistoryError, match="duplicate message_id 'msg_same'"):
        normalize_role_tagged_messages(
            [
                {
                    "message_id": "msg_same",
                    "role": "user",
                    "content": "Summarize this.",
                },
                {
                    "message_id": "msg_same",
                    "role": "assistant",
                    "content": "I can save it later.",
                },
            ]
        )


def test_appended_assistant_message_receives_new_id() -> None:
    messages = normalize_role_tagged_messages(
        [{"role": "user", "content": "Summarize this."}]
    )

    updated = append_assistant_message(messages, "Done.")

    assert len(updated) == 2
    assert updated[-1].role == "assistant"
    assert updated[-1].content == "Done."
    assert updated[-1].message_id
    assert updated[-1].message_id != messages[0].message_id
