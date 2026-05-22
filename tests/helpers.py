from __future__ import annotations

import re
from pathlib import Path


def single_run_dir(outputs: Path) -> Path:
    entries = list(outputs.iterdir())
    run_dirs = [entry for entry in entries if entry.is_dir()]
    assert len(run_dirs) == 1
    assert entries == run_dirs
    assert re.fullmatch(r"\d{8}T\d{6}Z(?:-\d{3})?", run_dirs[0].name)
    return run_dirs[0]
