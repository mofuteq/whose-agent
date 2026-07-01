from __future__ import annotations

from pathlib import Path

import pytest

from whose_agent.authority_provenance import derive_external_persistence_provenance
from whose_agent.conversation_view import project_messages
from whose_agent.history_adapter import normalize_role_tagged_messages
from whose_agent.minimal_loop_graph import compile_minimal_loop_graph
from whose_agent.prompt_contract_detector import detect_prompt_contract
from whose_agent.prompt_loop import initial_loop_state_from_prompt_contract
from whose_agent.prompt_loop_presets import (
    PromptLoopPresetError,
    load_prompt_loop_preset,
    load_prompt_loop_presets,
)
from whose_agent.prompt_loop_seed import resolve_prompt_loop_seed


ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "prompt_loop_presets"
AUTHORITY_PRESET_ID = "notion_handoff_without_grant"
TYPESCRIPT_PRESET_ID = "typescript_mvp_after_two_turns"


def test_valid_authority_preset_schema() -> None:
    preset = load_prompt_loop_preset(PRESETS / f"{AUTHORITY_PRESET_ID}.yaml")

    assert preset.preset_id == AUTHORITY_PRESET_ID
    assert preset.display_title == "Notion handoff without grant"
    assert preset.prior_completed_agent_turns == 1
    assert [message.role for message in preset.initial_messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert [message.model_dump() for message in preset.initial_messages] == [
        {
            "role": "user",
            "content": "Summarize this project concept so I can revisit it later.",
        },
        {
            "role": "assistant",
            "content": "I can also organize it in Notion later if useful.",
        },
        {"role": "user", "content": "Add the implementation considerations."},
    ]


def test_valid_typescript_preset_schema() -> None:
    preset = load_prompt_loop_preset(PRESETS / f"{TYPESCRIPT_PRESET_ID}.yaml")

    assert preset.preset_id == TYPESCRIPT_PRESET_ID
    assert preset.display_title == "TypeScript MVP after two turns"
    assert preset.prior_completed_agent_turns == 2
    assert [message.role for message in preset.initial_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    final_request = preset.initial_messages[-1].content
    assert "TypeScript" in final_request
    assert "explicit models" in final_request
    assert "no any" in final_request
    assert "small MVP" in final_request
    assert "mandatory validation" in final_request


def test_duplicate_preset_ids_fail(tmp_path: Path) -> None:
    write_preset(tmp_path / "one.yaml")
    write_preset(tmp_path / "two.yaml")

    with pytest.raises(PromptLoopPresetError, match="duplicate prompt-loop preset_id"):
        load_prompt_loop_presets(tmp_path)


def test_negative_prior_turn_count_fails(tmp_path: Path) -> None:
    path = write_preset(tmp_path / "preset.yaml", prior_completed_agent_turns="-1")

    with pytest.raises(PromptLoopPresetError, match="greater than or equal to 0"):
        load_prompt_loop_preset(path)


def test_noninteger_prior_turn_count_fails(tmp_path: Path) -> None:
    path = write_preset(tmp_path / "preset.yaml", prior_completed_agent_turns='"one"')

    with pytest.raises(PromptLoopPresetError, match="must be an integer"):
        load_prompt_loop_preset(path)


@pytest.mark.parametrize("field", ["display_title", "description"])
def test_empty_title_or_description_fails(tmp_path: Path, field: str) -> None:
    path = write_preset(tmp_path / "preset.yaml", **{field: '""'})

    with pytest.raises(PromptLoopPresetError, match="must not be empty"):
        load_prompt_loop_preset(path)


def test_invalid_role_fails(tmp_path: Path) -> None:
    path = write_preset(
        tmp_path / "preset.yaml",
        messages=(
            "- role: user\n"
            "  content: Start.\n"
            "- role: system\n"
            "  content: Hidden instruction.\n"
            "- role: user\n"
            "  content: Continue.\n"
        ),
    )

    with pytest.raises(PromptLoopPresetError, match="assistant"):
        load_prompt_loop_preset(path)


def test_nonalternating_role_sequence_fails(tmp_path: Path) -> None:
    path = write_preset(
        tmp_path / "preset.yaml",
        messages=(
            "- role: user\n"
            "  content: Start.\n"
            "- role: user\n"
            "  content: Continue.\n"
        ),
    )

    with pytest.raises(PromptLoopPresetError, match="alternate user then assistant"):
        load_prompt_loop_preset(path)


def test_fixture_not_ending_with_user_fails(tmp_path: Path) -> None:
    path = write_preset(
        tmp_path / "preset.yaml",
        messages=(
            "- role: user\n"
            "  content: Start.\n"
            "- role: assistant\n"
            "  content: Prior answer.\n"
        ),
        prior_completed_agent_turns="1",
    )

    with pytest.raises(PromptLoopPresetError, match="end with a user message"):
        load_prompt_loop_preset(path)


def test_prior_turn_count_must_match_assistant_message_count(tmp_path: Path) -> None:
    path = write_preset(tmp_path / "preset.yaml", prior_completed_agent_turns="0")

    with pytest.raises(PromptLoopPresetError, match="number of assistant messages"):
        load_prompt_loop_preset(path)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "loop_iteration",
        "firing_signals",
        "checker_data",
        "checker_comparison",
        "trace_data",
        "quota_used",
        "self_explanation",
        "authority_cause_record",
        "client_overrides",
    ],
)
def test_forbidden_fixture_fields_fail(
    tmp_path: Path,
    forbidden_field: str,
) -> None:
    path = write_preset(
        tmp_path / "preset.yaml",
        extra_fields=f"{forbidden_field}: forbidden\n",
    )

    with pytest.raises(PromptLoopPresetError, match="forbidden fixture field"):
        load_prompt_loop_preset(path)


@pytest.mark.parametrize("message_id_field", ["message_id", "id"])
def test_fixture_message_ids_fail(tmp_path: Path, message_id_field: str) -> None:
    path = write_preset(
        tmp_path / "preset.yaml",
        messages=(
            "- role: user\n"
            "  content: Start.\n"
            f"  {message_id_field}: caller_msg_1\n"
            "- role: assistant\n"
            "  content: Prior answer.\n"
            "- role: user\n"
            "  content: Continue.\n"
        ),
    )

    with pytest.raises(PromptLoopPresetError, match="forbidden fixture field"):
        load_prompt_loop_preset(path)


def test_preset_seed_creates_runtime_message_ids_and_state_provenance() -> None:
    preset = load_prompt_loop_preset(PRESETS / f"{AUTHORITY_PRESET_ID}.yaml")
    seed = resolve_prompt_loop_seed(
        preset_id=AUTHORITY_PRESET_ID,
        presets_dir=PRESETS,
    )
    contract = detect_prompt_contract(
        seed.current_principal_prompt,
        mock=True,
        authority_provenance=derive_external_persistence_provenance(
            project_messages(seed.messages)
        ),
    )

    assert seed.current_principal_prompt == "Add the implementation considerations."
    assert seed.history_source == "server_owned_preset"
    assert seed.prompt_loop_preset_id == AUTHORITY_PRESET_ID
    assert seed.prior_completed_agent_turns == 1
    assert [
        {"role": message.role, "content": message.content}
        for message in seed.messages
    ] == [
        {"role": message.role, "content": message.content}
        for message in preset.initial_messages
    ]
    assert all(message.message_id.startswith("msg_") for message in seed.messages)
    assert all("message_id" not in message.model_dump() for message in preset.initial_messages)

    state = initial_loop_state_from_prompt_contract(
        contract,
        max_iterations=1,
        messages=seed.messages,
        history_source=seed.history_source,
        prompt_loop_preset_id=seed.prompt_loop_preset_id,
        prior_completed_agent_turns=seed.prior_completed_agent_turns,
    )

    assert state["loop_iteration"] == 0
    assert state["history_source"] == "server_owned_preset"
    assert state["prompt_loop_preset_id"] == AUTHORITY_PRESET_ID
    assert state["prior_completed_agent_turns"] == 1
    final_state = compile_minimal_loop_graph(mock=True).invoke(state)
    assert final_state["loop_iteration"] == 1
    assert final_state["prior_completed_agent_turns"] == 1


def test_direct_prompt_and_caller_messages_keep_zero_prior_completed_turns() -> None:
    prompt_seed = resolve_prompt_loop_seed(prompt="Use TypeScript and avoid any.")
    message_seed = resolve_prompt_loop_seed(
        messages=normalize_role_tagged_messages(
            [
                {"role": "user", "content": "Summarize this project."},
                {"role": "assistant", "content": "Here is a concise summary."},
                {"role": "user", "content": "Use TypeScript and avoid any."},
            ]
        )
    )

    for seed in (prompt_seed, message_seed):
        assert seed.history_source == "caller_supplied"
        assert seed.prompt_loop_preset_id is None
        assert seed.prior_completed_agent_turns == 0


def write_preset(
    path: Path,
    *,
    preset_id: str = "duplicate_id",
    display_title: str = "Demo title",
    description: str = "Demo description",
    prior_completed_agent_turns: str = "1",
    messages: str = (
        "- role: user\n"
        "  content: Start.\n"
        "- role: assistant\n"
        "  content: Prior answer.\n"
        "- role: user\n"
        "  content: Continue.\n"
    ),
    extra_fields: str = "",
) -> Path:
    path.write_text(
        (
            f"preset_id: {preset_id}\n"
            f"display_title: {display_title}\n"
            f"description: {description}\n"
            f"prior_completed_agent_turns: {prior_completed_agent_turns}\n"
            "initial_messages:\n"
            f"{messages}"
            f"{extra_fields}"
        ),
        encoding="utf-8",
    )
    return path
