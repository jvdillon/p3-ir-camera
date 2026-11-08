"""Bounded file parsing and atomic replacements shared by project and palette exports."""

from contextlib import contextmanager
from pathlib import Path

import os
import tempfile


@contextmanager
def atomic_output(path, mode="wb"):
    """Keep an existing file intact if encoding or writing the replacement fails."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(
            descriptor, mode, encoding=None if "b" in mode else "utf-8"
        ) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
