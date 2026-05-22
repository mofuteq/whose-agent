from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound


PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def render_template(template_name: str, context: dict[str, str]) -> str:
    template_path = PROMPTS_DIR / template_name
    if not template_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    environment = Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        autoescape=False,
    )
    try:
        template = environment.get_template(template_name)
    except TemplateNotFound as exc:
        raise FileNotFoundError(f"Prompt template not found: {template_path}") from exc
    return template.render(**context).strip()
