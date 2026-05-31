from __future__ import annotations

import json
from pathlib import Path

from whose_agent.schemas import PromptContract


def write_prompt_contract(
    contract: PromptContract,
    output_dir: Path,
    stem: str = "prompt_contract",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.prompt_contract.json"
    path.write_text(
        json.dumps(contract.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = ["write_prompt_contract"]
