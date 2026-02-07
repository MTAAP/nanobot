"""Atomic file write utilities."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON data atomically via tempfile + os.replace().

    Guarantees that the file is either fully written or untouched.
    On crash during write, the original file remains intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically via tempfile + os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
