#!/usr/bin/env python3
"""
Migrate NERVE session directories to the updated structure.

For each extracted session under the data root:

1. Move ``evk4_events.hdf5`` into ``prophesee/events.hdf5``
   (also moves ``evk4_events.bias`` into ``prophesee/`` if present).
2. Convert the Infineon Position2Go radar data from the ``.rad`` file
   into the radardb HDF5 format under ``infineon_radar/``.

Both steps are **idempotent**: if a session already has the new layout,
it is left untouched.

Usage::

    # Migrate all sessions under the default data root (~/.nerve/data/)
    python -m nerve.migrate_sessions

    # Explicit data root
    python -m nerve.migrate_sessions --data-root /scratch/nerve_data

    # Dry run (report what would be done without changing anything)
    python -m nerve.migrate_sessions --dry-run

    # Process a single session
    python -m nerve.migrate_sessions --session 2023-10-26_15-37-59
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from nerve.config import get_data_root


def _is_session_dir(p: Path) -> bool:
    """Heuristic: a session directory contains session_metadata.json or a
    recognisable sensor subdirectory."""
    if not p.is_dir():
        return False
    if (p / "session_metadata.json").exists():
        return True
    if (p / "davis").is_dir() or (p / "prophesee").is_dir():
        return True
    if (p / "ti_radar").is_dir():
        return True
    return False


def migrate_evk4(session: Path, *, dry_run: bool = False) -> list[str]:
    """Move evk4_events.hdf5 (and .bias) into prophesee/."""
    actions = []

    old_hdf5 = session / "evk4_events.hdf5"
    new_hdf5 = session / "prophesee" / "events.hdf5"
    old_bias = session / "evk4_events.bias"
    new_bias = session / "prophesee" / "evk4_events.bias"
    old_sync = session / "evk4_events.sync"
    new_sync = session / "prophesee" / "evk4_events.sync"

    if not old_hdf5.exists():
        if new_hdf5.exists():
            actions.append("  [ok] prophesee/events.hdf5 already in place")
        else:
            actions.append("  [skip] no evk4_events.hdf5 found (no Prophesee data)")
        return actions

    if new_hdf5.exists():
        actions.append("  [skip] prophesee/events.hdf5 already exists, keeping both")
        return actions

    prophesee_dir = session / "prophesee"
    if not prophesee_dir.is_dir():
        actions.append(f"  mkdir {prophesee_dir.name}/")
        if not dry_run:
            prophesee_dir.mkdir(exist_ok=True)

    actions.append(f"  move evk4_events.hdf5 -> prophesee/events.hdf5")
    if not dry_run:
        shutil.move(str(old_hdf5), str(new_hdf5))

    if old_bias.exists() and not new_bias.exists():
        actions.append(f"  move evk4_events.bias -> prophesee/evk4_events.bias")
        if not dry_run:
            shutil.move(str(old_bias), str(new_bias))

    if old_sync.exists() and not new_sync.exists():
        actions.append(f"  move evk4_events.sync -> prophesee/evk4_events.sync")
        if not dry_run:
            shutil.move(str(old_sync), str(new_sync))

    return actions


def migrate_infineon_radar(session: Path, *, dry_run: bool = False) -> list[str]:
    """Convert .rad radar frames to radardb format under infineon_radar/."""
    actions = []

    rad_file = session / "radar_and_davis346_events.rad"
    infineon_dir = session / "infineon_radar"
    recording_xml = infineon_dir / "recording.xml"

    if recording_xml.exists():
        actions.append("  [ok] infineon_radar/ already exists")
        return actions

    if not rad_file.exists():
        actions.append("  [skip] no radar_and_davis346_events.rad found")
        return actions

    actions.append(f"  convert {rad_file.name} -> infineon_radar/")

    if dry_run:
        return actions

    from nerve.extraction.access.radar_dvs.rad_radar_to_radardb import convert

    try:
        _, n_frames = convert(
            input_path=str(rad_file),
            output_dir=str(infineon_dir),
            verbose=False,
        )
        actions.append(f"  [done] wrote {n_frames} Infineon radar frames")
    except Exception as e:
        actions.append(f"  [error] conversion failed: {e}")

    return actions


def migrate_session(session: Path, *, dry_run: bool = False) -> list[str]:
    """Run all migrations on a single session directory."""
    all_actions = []
    all_actions.extend(migrate_evk4(session, dry_run=dry_run))
    all_actions.extend(migrate_infineon_radar(session, dry_run=dry_run))
    return all_actions


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NERVE session directories to the updated structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Process only this session (name, not full path)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be done without making changes",
    )
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}")
        sys.exit(1)

    print(f"Data root: {data_root}")
    if args.dry_run:
        print("*** DRY RUN — no changes will be made ***\n")

    if args.session:
        sessions = [data_root / args.session]
        if not sessions[0].is_dir():
            print(f"Session directory not found: {sessions[0]}")
            sys.exit(1)
    else:
        sessions = sorted(
            p for p in data_root.iterdir() if _is_session_dir(p)
        )

    if not sessions:
        print("No session directories found.")
        sys.exit(0)

    total_sessions = len(sessions)
    migrated = 0

    for session in sessions:
        print(f"\n[{session.name}]")
        actions = migrate_session(session, dry_run=args.dry_run)
        for a in actions:
            print(a)
        if any("[done]" in a or "move" in a for a in actions):
            migrated += 1

    print(f"\n{'=' * 60}")
    print(f"Processed {total_sessions} session(s), {migrated} modified.")
    if args.dry_run:
        print("(dry run — rerun without --dry-run to apply changes)")


if __name__ == "__main__":
    main()
