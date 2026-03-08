"""Filesystem tools — read and inspect files on the local filesystem."""

from pathlib import Path


def read_file(path: str) -> str:
    """Return the text content of *path*, or an error message if it does not exist."""
    p = Path(path)

    if not p.exists():
        return f"File not found: {path}"

    return p.read_text()
