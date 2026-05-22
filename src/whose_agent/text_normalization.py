from __future__ import annotations

import unicodedata


def normalize_llm_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()
