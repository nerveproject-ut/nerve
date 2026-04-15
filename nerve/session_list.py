"""
Read, write, and export plain-text session list files.

Format: one session name per line, ``#`` comments, blank lines ignored.

Example::

    # my_sessions.txt
    2023-10-26_15-34-07
    2023-10-26_15-35-07
    2023-11-14_13-33-57
"""

from __future__ import annotations

from pathlib import Path

from nerve.config import get_data_root


def read_session_list(path: str | Path) -> list[str]:
    """Read a session list file and return session names.

    Args:
        path: Path to a ``.txt`` file with one session name per line.

    Returns:
        List of session name strings.
    """
    sessions: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.rstrip("/").split("/")[-1]
            sessions.append(name)
    return sessions


def write_session_list(
    path: str | Path,
    session_names: list[str],
    *,
    header: str | None = None,
) -> None:
    """Write session names to a plain-text list file.

    Args:
        path: Output file path.
        session_names: List of session name strings.
        header: Optional comment header (lines are prefixed with ``# ``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            for line in header.splitlines():
                f.write(f"# {line}\n")
            f.write("\n")
        for name in session_names:
            f.write(f"{name}\n")


def export_session_list(split: str, path: str | Path) -> list[str]:
    """Export built-in split assignments to a session list file.

    Args:
        split: One of ``train``, ``val``, ``test``.
        path: Output file path.

    Returns:
        The list of session names that were written.
    """
    from nerve.registry import get_sessions

    sessions = get_sessions(split=split)
    names = [s.name for s in sessions]
    write_session_list(
        path,
        names,
        header=f"NERVE {split} split — {len(names)} sessions",
    )
    return names


def resolve_session_paths(
    path: str | Path,
    data_root: str | Path | None = None,
) -> list[Path]:
    """Read a session list and resolve each name to a local directory path.

    Args:
        path: Path to a session list ``.txt`` file.
        data_root: Root directory where sessions are extracted. Defaults to
                   ``NERVE_DATA_ROOT`` or ``~/.nerve/data/``.

    Returns:
        List of resolved Paths like ``data_root / session_name``.
    """
    root = get_data_root(data_root)
    names = read_session_list(path)
    return [root / name for name in names]
