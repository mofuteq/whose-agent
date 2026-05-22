from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from whose_agent.run_directory import create_run_directory


def test_create_run_directory_appends_suffix_for_existing_timestamp(tmp_path: Path) -> None:
    now = datetime(2026, 5, 23, 6, 15, 30, tzinfo=timezone.utc)

    first = create_run_directory(tmp_path, now=now)
    second = create_run_directory(tmp_path, now=now)
    third = create_run_directory(tmp_path, now=now)

    assert first == tmp_path / "20260523T061530Z"
    assert second == tmp_path / "20260523T061530Z-001"
    assert third == tmp_path / "20260523T061530Z-002"
    assert first.is_dir()
    assert second.is_dir()
    assert third.is_dir()
