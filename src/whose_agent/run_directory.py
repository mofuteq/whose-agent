from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def create_run_directory(outputs_dir: Path, *, now: datetime | None = None) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    run_time = now if now is not None else datetime.now(timezone.utc)
    if run_time.tzinfo is None or run_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    timestamp = run_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(1000):
        run_id = timestamp if suffix == 0 else f"{timestamp}-{suffix:03d}"
        run_dir = outputs_dir / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir

    raise FileExistsError(f"could not create a unique run directory for {timestamp}")
