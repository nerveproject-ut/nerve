#!/usr/bin/env python3
"""
Remove unexpected files and directories from NERVE session folders.

Uses the same allowlists as ``validate_sessions.py`` to decide what is
extra.  Before deleting a ``tmp/`` directory that contains an
``annotations/annotations.json``, the script verifies that it matches
the sibling ``annotations/annotations.json`` byte-for-byte.  If it
does not match the ``tmp/`` directory is kept and a warning is printed.

Usage::

    # Dry run (default) — only show what would be removed
    python -m nerve.clean_sessions --data-root /scratch/data

    # Actually delete
    python -m nerve.clean_sessions --data-root /scratch/data --delete

    # Single session
    python -m nerve.clean_sessions --data-root /scratch/data --session 2023-12-04_12-03-26 --delete
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

from nerve.config import get_data_root
from nerve.validate_sessions import (
    ALLOWED_ANNOTATIONS,
    ALLOWED_DAVIS,
    ALLOWED_INFINEON_RADAR,
    ALLOWED_PROPHESEE,
    ALLOWED_RGB,
    ALLOWED_TI_RADAR,
    ALLOWED_TOP_DIRS,
    ALLOWED_TOP_LEVEL,
    _is_session_dir,
)


def _collect_extras(directory: Path, allowed: set) -> list[Path]:
    extras = []
    if not directory.is_dir():
        return extras
    for entry in sorted(directory.iterdir()):
        if entry.name not in allowed:
            extras.append(entry)
    return extras


def _is_duplicate_tmp(tmp_dir: Path, parent_dir: Path) -> bool:
    """Return True if *tmp_dir* is a redundant staging copy of *parent_dir*.

    Checks ``annotations/annotations.json`` inside both directories.
    If the files are byte-identical the whole ``tmp/`` is considered safe
    to remove.
    """
    tmp_ann = tmp_dir / "annotations" / "annotations.json"
    real_ann = parent_dir / "annotations" / "annotations.json"
    if not tmp_ann.is_file() or not real_ann.is_file():
        return False
    return filecmp.cmp(str(tmp_ann), str(real_ann), shallow=False)


def clean_session(session: Path, *, delete: bool = False) -> list[str]:
    """Identify (and optionally remove) unexpected entries in a session.

    Returns a list of human-readable action strings.
    """
    actions: list[str] = []
    verb = "Removing" if delete else "Would remove"

    subdirs_and_allowlists: list[tuple[str, set]] = [
        ("davis", ALLOWED_DAVIS),
        ("prophesee", ALLOWED_PROPHESEE),
        ("rgb", ALLOWED_RGB),
        ("ti_radar", ALLOWED_TI_RADAR),
        ("infineon_radar", ALLOWED_INFINEON_RADAR),
    ]

    # ── Top-level extras ────────────────────────────────────────────
    for entry in _collect_extras(session, ALLOWED_TOP_LEVEL | ALLOWED_TOP_DIRS):
        actions.append(f"  {verb} {entry.name} ({'dir' if entry.is_dir() else 'file'})")
        if delete:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    # ── Sensor subdirectories ───────────────────────────────────────
    for subname, allowed in subdirs_and_allowlists:
        subdir = session / subname
        if not subdir.is_dir():
            continue

        for entry in _collect_extras(subdir, allowed):
            label = f"{subname}/{entry.name}"

            if entry.is_dir() and entry.name == "tmp":
                if _is_duplicate_tmp(entry, subdir):
                    actions.append(f"  {verb} {label}/ (duplicate staging dir)")
                    if delete:
                        shutil.rmtree(entry)
                else:
                    actions.append(f"  KEEPING  {label}/ (annotations differ from parent — manual review needed)")
                continue

            actions.append(f"  {verb} {label} ({'dir' if entry.is_dir() else 'file'})")
            if delete:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        # Extras inside annotations/
        ann_dir = subdir / "annotations"
        if ann_dir.is_dir():
            for entry in _collect_extras(ann_dir, ALLOWED_ANNOTATIONS):
                label = f"{subname}/annotations/{entry.name}"
                actions.append(f"  {verb} {label} ({'dir' if entry.is_dir() else 'file'})")
                if delete:
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()

    return actions


def _discover_sessions(data_root: Path, session_name: str | None) -> list[Path]:
    if session_name:
        candidate = data_root / session_name
        if candidate.is_dir():
            return [candidate]
        for sub in ("train", "val", "test"):
            candidate = data_root / sub / session_name
            if candidate.is_dir():
                return [candidate]
        print(f"Session not found: {session_name}", file=sys.stderr)
        sys.exit(1)

    sessions = sorted(p for p in data_root.iterdir() if _is_session_dir(p))
    for sub in ("train", "val", "test"):
        sub_dir = data_root / sub
        if sub_dir.is_dir():
            sessions.extend(sorted(p for p in sub_dir.iterdir() if _is_session_dir(p)))
    return sessions


def main():
    parser = argparse.ArgumentParser(
        description="Remove unexpected files/dirs from NERVE sessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--session", type=str, default=None,
                        help="Clean only this session")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)

    sessions = _discover_sessions(data_root, args.session)
    if not sessions:
        print("No sessions found.")
        sys.exit(0)

    mode = "DELETE mode" if args.delete else "DRY RUN (pass --delete to actually remove)"
    print(f"Data root: {data_root}")
    print(f"Mode:      {mode}")
    print(f"Sessions:  {len(sessions)}\n")

    total_actions = 0
    for session in sessions:
        actions = clean_session(session, delete=args.delete)
        if actions:
            print(f"[{'CLEANED' if args.delete else 'DIRTY'}] {session.name}")
            for a in actions:
                print(a)
            print()
            total_actions += len(actions)

    print("=" * 60)
    if args.delete:
        print(f"Removed {total_actions} item(s) across {len(sessions)} session(s).")
    else:
        print(f"Found {total_actions} item(s) to remove across {len(sessions)} session(s).")
        if total_actions:
            print("Re-run with --delete to actually remove them.")


if __name__ == "__main__":
    main()
