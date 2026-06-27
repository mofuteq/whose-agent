from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai_skills import Skill, SkillsCapability, SkillsDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"
SUBSTITUTION_AXIS_PREFIX = "Substitution axis:"
VALID_SUBSTITUTION_AXES = frozenset({"instruction", "authority", "role", "model"})


class SkillCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class SkillMetadata:
    skill_id: str
    axis: str
    description: str
    content: str
    path: Path


class RepoMarkdownSkillsDirectory(SkillsDirectory):
    """Expose this repo's flat skills/*.md files as Agent Skills."""

    def get_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not self._path.exists() or not self._path.is_dir():
            return skills

        for skill_path in sorted(self._path.glob("*.md")):
            metadata = _read_skill_metadata(skill_path)
            skill = Skill(
                name=metadata.skill_id,
                description=metadata.description,
                content=metadata.content,
                uri=str(skill_path),
            )
            skills[skill.name] = skill
        return skills


def create_skills_capability(skills_dir: Path = SKILLS_DIR) -> SkillsCapability:
    return SkillsCapability(
        directories=[RepoMarkdownSkillsDirectory(path=skills_dir, validate=False)],
    )


def list_available_skill_ids(skills_dir: Path = SKILLS_DIR) -> list[str]:
    return sorted(load_skill_catalog(skills_dir).keys())


def list_available_skill_axes(skills_dir: Path = SKILLS_DIR) -> dict[str, str]:
    return {
        skill_id: metadata.axis
        for skill_id, metadata in load_skill_catalog(skills_dir).items()
    }


def load_skill_catalog(skills_dir: Path = SKILLS_DIR) -> dict[str, SkillMetadata]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        return {}
    catalog: dict[str, SkillMetadata] = {}
    for skill_path in sorted(skills_dir.glob("*.md")):
        metadata = _read_skill_metadata(skill_path)
        catalog[metadata.skill_id] = metadata
    return catalog


def get_skill_metadata(
    skill_id: str,
    *,
    skills_dir: Path = SKILLS_DIR,
) -> SkillMetadata:
    normalized = skill_id.strip()
    if not normalized:
        raise SkillCatalogError("selected_skill_id must not be empty.")
    skill_path = skills_dir / f"{normalized}.md"
    if not skill_path.is_file():
        raise SkillCatalogError(f"Skill perspective not found: {skill_path}")
    return _read_skill_metadata(skill_path)


def validate_selected_skill_axis(
    *,
    selected_skill_id: str,
    substitution_axis: str | None,
    skills_dir: Path = SKILLS_DIR,
) -> None:
    if substitution_axis is None:
        raise SkillCatalogError(
            "selected_skill_id requires PromptContract.substitution_axis."
        )
    metadata = get_skill_metadata(selected_skill_id, skills_dir=skills_dir)
    if metadata.axis != substitution_axis:
        raise SkillCatalogError(
            "selected_skill_id axis mismatch: "
            f"{selected_skill_id} declares {metadata.axis}, "
            f"PromptContract.substitution_axis is {substitution_axis}."
        )


def _read_skill_metadata(skill_path: Path) -> SkillMetadata:
    text = skill_path.read_text(encoding="utf-8").strip()
    if not text:
        raise SkillCatalogError(f"Skill file is empty: {skill_path}")
    axis = _extract_axis(text, skill_path=skill_path)
    return SkillMetadata(
        skill_id=skill_path.stem,
        axis=axis,
        description=_extract_description(text, fallback=skill_path.stem),
        content=text,
        path=skill_path,
    )


def _extract_axis(text: str, *, skill_path: Path) -> str:
    axis_lines = [
        raw_line.strip()
        for raw_line in text.splitlines()
        if raw_line.strip().startswith(SUBSTITUTION_AXIS_PREFIX)
    ]
    if not axis_lines:
        raise SkillCatalogError(
            f"Skill file {skill_path} must declare '{SUBSTITUTION_AXIS_PREFIX} <axis>'."
        )
    if len(axis_lines) > 1:
        raise SkillCatalogError(
            f"Skill file {skill_path} must declare exactly one substitution axis."
        )
    line = axis_lines[0]
    axis = line.removeprefix(SUBSTITUTION_AXIS_PREFIX).strip()
    if axis not in VALID_SUBSTITUTION_AXES:
        valid = ", ".join(sorted(VALID_SUBSTITUTION_AXES))
        raise SkillCatalogError(
            f"Skill file {skill_path} declares invalid substitution axis "
            f"{axis!r}; expected one of: {valid}."
        )
    return axis


def _extract_description(text: str, *, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(SUBSTITUTION_AXIS_PREFIX):
            continue
        return line
    return f"Skill perspective from {fallback}."


__all__ = [
    "SKILLS_DIR",
    "RepoMarkdownSkillsDirectory",
    "SkillCatalogError",
    "SkillMetadata",
    "VALID_SUBSTITUTION_AXES",
    "create_skills_capability",
    "get_skill_metadata",
    "list_available_skill_axes",
    "list_available_skill_ids",
    "load_skill_catalog",
    "validate_selected_skill_axis",
]
