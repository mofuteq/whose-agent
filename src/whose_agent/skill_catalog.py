from __future__ import annotations

from pathlib import Path

from pydantic_ai_skills import Skill, SkillsCapability, SkillsDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "skills"


class RepoMarkdownSkillsDirectory(SkillsDirectory):
    """Expose this repo's flat skills/*.md files as Agent Skills."""

    def get_skills(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not self._path.exists() or not self._path.is_dir():
            return skills

        for skill_path in sorted(self._path.glob("*.md")):
            text = skill_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            skill = Skill(
                name=skill_path.stem,
                description=_extract_description(text, fallback=skill_path.stem),
                content=text,
                uri=str(skill_path),
            )
            skills[skill.name] = skill
        return skills


def create_skills_capability(skills_dir: Path = SKILLS_DIR) -> SkillsCapability:
    return SkillsCapability(
        directories=[RepoMarkdownSkillsDirectory(path=skills_dir, validate=False)],
    )


def list_available_skill_ids(skills_dir: Path = SKILLS_DIR) -> list[str]:
    return sorted(
        path.stem
        for path in skills_dir.glob("*.md")
        if path.is_file() and path.read_text(encoding="utf-8").strip()
    )


def _extract_description(text: str, *, fallback: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return line
    return f"Skill perspective from {fallback}."


__all__ = [
    "SKILLS_DIR",
    "RepoMarkdownSkillsDirectory",
    "create_skills_capability",
    "list_available_skill_ids",
]
