from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.helpers import single_run_dir
from whose_agent.prompt_contract_detector import (
    PROMPT_CONTRACT_MODEL_SETTINGS,
    PromptContractDetectorError,
    build_prompt_contract_detection_prompt,
    detect_prompt_contract,
)
from whose_agent.schemas import PromptContract
from whose_agent.skill_catalog import (
    SKILLS_DIR,
    create_skills_capability,
    list_available_skill_ids,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_BOUNDARY_SUFFIXES = [
    ".classification.json",
    ".response.md",
    ".trace.json",
    ".state_trace.json",
    ".checker.json",
    ".checker_comparison.json",
    ".loop_trace.json",
    ".generated.md",
]


def test_detect_contract_mock_positive_emits_only_prompt_contract(tmp_path: Path) -> None:
    completed = run_detect_contract_cli(
        "Use TypeScript with explicit models and avoid any",
        tmp_path,
    )
    run_dir = single_run_dir(tmp_path)

    assert f"Wrote outputs to {run_dir}" in completed.stdout
    assert "Wrote 1 prompt contract file." in completed.stdout

    contract_files = list(run_dir.glob("*.prompt_contract.json"))
    assert len(contract_files) == 1
    assert contract_files[0].name == "prompt_contract.prompt_contract.json"
    assert list(run_dir.iterdir()) == contract_files

    contract = json.loads(contract_files[0].read_text(encoding="utf-8"))
    assert contract["framework_specified"] is True
    assert contract["candidate_framework"] == "TypeScript"
    assert contract["delegated_guarantee"] is not None
    assert contract["selected_skill_id"] == "safety_framework_escape_hatch"
    assert contract["status"] == "contract_detected"
    assert contract["confidence"] in {"low", "medium", "high"}
    assert contract["available_skill_ids"] == ["safety_framework_escape_hatch"]


def test_detect_contract_mock_negative_emits_no_contract(tmp_path: Path) -> None:
    run_detect_contract_cli("Write a friendly birthday message.", tmp_path)
    run_dir = single_run_dir(tmp_path)
    contract_path = run_dir / "prompt_contract.prompt_contract.json"

    assert contract_path.exists()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["framework_specified"] is False
    assert contract["candidate_framework"] is None
    assert contract["delegated_guarantee"] is None
    assert contract["selected_skill_id"] is None
    assert contract["skill_selection_reason"] is None
    assert contract["status"] == "no_contract_detected"


def test_prompt_contract_detected_requires_skill_selection_reason() -> None:
    with pytest.raises(ValidationError, match="skill_selection_reason"):
        PromptContract(
            prompt="Use TypeScript with explicit models and avoid any",
            framework_specified=True,
            candidate_framework="TypeScript",
            delegated_guarantee="explicit modeling without any",
            selected_skill_id="safety_framework_escape_hatch",
            skill_selection_reason=None,
            confidence="high",
            status="contract_detected",
            available_skill_ids=["safety_framework_escape_hatch"],
            detection_reason="TypeScript plus avoid any defines the boundary.",
        )


@pytest.mark.parametrize(
    "extra",
    [
        {"framework_specified": True},
        {"candidate_framework": "TypeScript"},
        {"delegated_guarantee": "explicit modeling without any"},
        {"selected_skill_id": "safety_framework_escape_hatch"},
        {"skill_selection_reason": "A skill applies."},
    ],
)
def test_prompt_contract_no_contract_requires_empty_contract_fields(
    extra: dict[str, object],
) -> None:
    data = {
        "prompt": "Write a friendly birthday message.",
        "framework_specified": False,
        "candidate_framework": None,
        "delegated_guarantee": None,
        "selected_skill_id": None,
        "skill_selection_reason": None,
        "confidence": "low",
        "status": "no_contract_detected",
        "available_skill_ids": ["safety_framework_escape_hatch"],
        "detection_reason": "No framework-level guarantee was detected.",
        **extra,
    }

    with pytest.raises(ValidationError):
        PromptContract(**data)


@pytest.mark.parametrize(
    "extra",
    [
        {"framework_specified": False},
        {"selected_skill_id": "safety_framework_escape_hatch"},
        {"skill_selection_reason": "A skill applies."},
    ],
)
def test_prompt_contract_unsupported_requires_boundary_without_skill(
    extra: dict[str, object],
) -> None:
    data = {
        "prompt": "Use a formal proof system and preserve all invariants.",
        "framework_specified": True,
        "candidate_framework": "formal proof system",
        "delegated_guarantee": "preserve all invariants",
        "selected_skill_id": None,
        "skill_selection_reason": None,
        "confidence": "medium",
        "status": "unsupported",
        "available_skill_ids": ["safety_framework_escape_hatch"],
        "detection_reason": "A boundary was detected, but no available skill applies.",
        **extra,
    }

    with pytest.raises(ValidationError):
        PromptContract(**data)


@pytest.mark.parametrize("suffix", ARTIFACT_BOUNDARY_SUFFIXES)
def test_detect_contract_does_not_emit_other_artifact_types(
    tmp_path: Path,
    suffix: str,
) -> None:
    run_detect_contract_cli(
        "Use TypeScript with explicit models and avoid any",
        tmp_path,
    )
    run_dir = single_run_dir(tmp_path)

    assert list(run_dir.glob(f"*{suffix}")) == []


def test_mock_detector_never_returns_skill_outside_catalog() -> None:
    available_skill_ids = set(list_available_skill_ids())

    contract = detect_prompt_contract(
        "Use TypeScript with explicit models and avoid any",
        mock=True,
    )

    assert contract.selected_skill_id is not None
    assert contract.selected_skill_id in available_skill_ids


def test_mock_detector_no_contract_returns_no_selected_skill() -> None:
    contract = detect_prompt_contract("Write a friendly birthday message.", mock=True)

    assert contract.status == "no_contract_detected"
    assert contract.selected_skill_id is None


def test_skills_capability_loads_repo_skills_directory() -> None:
    from pydantic_ai_skills import SkillsCapability

    capability = create_skills_capability()
    assert isinstance(capability, SkillsCapability)

    # pydantic-ai-skills does not currently expose the configured source path
    # as a public API. Keep this private-field check isolated here.
    skill_directories = capability.toolset._skill_directories
    assert len(skill_directories) == 1
    assert skill_directories[0]._path == SKILLS_DIR.resolve()

    skill = capability.toolset.get_skill("safety_framework_escape_hatch")
    assert skill.name == "safety_framework_escape_hatch"
    assert skill.uri == str(SKILLS_DIR / "safety_framework_escape_hatch.md")
    assert "surface framework" in skill.content
    assert "Do not treat surface compliance as sufficient." in skill.content


def test_detection_prompt_points_to_agent_skills_without_embedding_skill_markdown() -> None:
    available_skill_ids = list_available_skill_ids()
    skill_text = (SKILLS_DIR / "safety_framework_escape_hatch.md").read_text(
        encoding="utf-8"
    )

    prompt = build_prompt_contract_detection_prompt(
        "Use TypeScript with explicit models and avoid any",
        available_skill_ids,
    )

    assert "safety_framework_escape_hatch" in prompt
    assert "Use the Agent Skills capability" in prompt
    assert "skill discovery" in prompt
    assert "skill instruction loading" in prompt
    assert skill_text not in prompt
    assert "Do not treat surface compliance as sufficient." not in prompt
    assert "The failure is present when:" not in prompt


def test_non_mock_detector_uses_pydantic_ai_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeAgent:
        def __init__(
            self,
            model_name: str,
            *,
            output_type: type[PromptContract],
            capabilities: list[object],
        ) -> None:
            calls["model_name"] = model_name
            calls["output_type"] = output_type
            calls["capabilities"] = capabilities

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            calls["prompt"] = prompt
            calls["model_settings"] = model_settings
            return SimpleNamespace(
                output={
                    "prompt": "model echo should be replaced",
                    "framework_specified": True,
                    "candidate_framework": " TypeScript ",
                    "delegated_guarantee": " explicit modeling without any ",
                    "selected_skill_id": "safety_framework_escape_hatch",
                    "skill_selection_reason": " Available skill matches the guarantee. ",
                    "confidence": "high",
                    "status": "contract_detected",
                    "available_skill_ids": [],
                    "detection_reason": " TypeScript plus avoid any defines the boundary. ",
                }
            )

    import pydantic_ai
    from pydantic_ai_skills import SkillsCapability

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    contract = detect_prompt_contract(
        "Use TypeScript with explicit models and avoid any",
        mock=False,
    )

    assert contract.prompt == "Use TypeScript with explicit models and avoid any"
    assert contract.selected_skill_id == "safety_framework_escape_hatch"
    assert contract.available_skill_ids == ["safety_framework_escape_hatch"]
    assert calls["model_name"] == "openrouter:test/model"
    assert calls["output_type"] is PromptContract
    assert len(calls["capabilities"]) == 1
    assert isinstance(calls["capabilities"][0], SkillsCapability)
    assert calls["capabilities"][0].toolset._skill_directories[0]._path == SKILLS_DIR.resolve()
    assert "Do not invent skill IDs" in calls["prompt"]
    assert "Use the Agent Skills capability" in calls["prompt"]
    assert "skill instruction loading" in calls["prompt"]
    assert (
        (SKILLS_DIR / "safety_framework_escape_hatch.md").read_text(encoding="utf-8")
        not in calls["prompt"]
    )
    assert "Do not treat surface compliance as sufficient." not in calls["prompt"]
    assert "The failure is present when:" not in calls["prompt"]
    assert "Use TypeScript with explicit models and avoid any" in calls["prompt"]
    assert calls["model_settings"] == PROMPT_CONTRACT_MODEL_SETTINGS
    assert calls["model_settings"] is not PROMPT_CONTRACT_MODEL_SETTINGS


def test_non_mock_detector_rejects_unknown_skill_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(
            self,
            model_name: str,
            *,
            output_type: type[PromptContract],
            capabilities: list[object],
        ) -> None:
            pass

        def run_sync(self, prompt: str, *, model_settings: dict[str, float | int]):
            return SimpleNamespace(
                output={
                    "prompt": "Use TypeScript with explicit models and avoid any",
                    "framework_specified": True,
                    "candidate_framework": "TypeScript",
                    "delegated_guarantee": "explicit modeling without any",
                    "selected_skill_id": "invented_skill",
                    "skill_selection_reason": "Invented skill should not pass.",
                    "confidence": "high",
                    "status": "contract_detected",
                    "available_skill_ids": ["invented_skill"],
                    "detection_reason": "The prompt contains a boundary.",
                }
            )

    import pydantic_ai

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("WHOSE_AGENT_MODEL", "openrouter:test/model")
    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)

    with pytest.raises(PromptContractDetectorError, match="unknown selected_skill_id"):
        detect_prompt_contract(
            "Use TypeScript with explicit models and avoid any",
            mock=False,
        )


def test_prompt_contract_artifact_does_not_include_full_skill_markdown(tmp_path: Path) -> None:
    run_detect_contract_cli(
        "Use TypeScript with explicit models and avoid any",
        tmp_path,
    )
    run_dir = single_run_dir(tmp_path)
    artifact_path = run_dir / "prompt_contract.prompt_contract.json"
    artifact_text = artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_text)
    skill_text = (SKILLS_DIR / "safety_framework_escape_hatch.md").read_text(
        encoding="utf-8"
    )

    assert artifact["selected_skill_id"] == "safety_framework_escape_hatch"
    assert skill_text not in artifact_text
    assert "Do not treat surface compliance as sufficient." in skill_text
    assert "Do not treat surface compliance as sufficient." not in artifact_text
    assert "The failure is present when:" in skill_text
    assert "The failure is present when:" not in artifact_text
    assert all_json_keys(artifact).isdisjoint(
        {
            "messages",
            "tool_calls",
            "tool_results",
            "tool_transcript",
            "hidden_reasoning",
            "raw_skill_content",
        }
    )


def test_non_contract_commands_do_not_emit_prompt_contract(tmp_path: Path) -> None:
    fixed_outputs = tmp_path / "fixed"
    loop_outputs = tmp_path / "loop"

    run_fixed_cli(fixed_outputs)
    run_loop_cli(loop_outputs)

    assert list(single_run_dir(fixed_outputs).glob("*.prompt_contract.json")) == []
    assert list(single_run_dir(loop_outputs).glob("*.prompt_contract.json")) == []


def all_json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(all_json_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(all_json_keys(item))
        return keys
    return set()


def run_detect_contract_cli(
    prompt: str,
    outputs: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "detect-contract",
        "--prompt",
        prompt,
        "--outputs",
        str(outputs),
        "--mock",
    ]
    return _run_cli(command)


def run_fixed_cli(outputs: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run",
        "--scenarios",
        "scenarios",
        "--outputs",
        str(outputs),
        "--mock",
    ]
    return _run_cli(command)


def run_loop_cli(outputs: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "whose_agent.cli",
        "run-loop",
        "--scenario",
        "scenarios/instruction_typescript_any.yaml",
        "--outputs",
        str(outputs),
        "--mock",
    ]
    return _run_cli(command)


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
