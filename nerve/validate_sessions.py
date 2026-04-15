#!/usr/bin/env python3
"""
Validate NERVE session directory structure.

Checks every extracted session under the data root for structural
correctness: required files, expected subdirectories, HDF5 integrity,
unexpected files, and consistency with the session registry metadata.

Usage::

    # Validate all sessions under default data root
    python -m nerve.validate_sessions

    # Explicit data root
    python -m nerve.validate_sessions --data-root /scratch/nerve_data

    # Validate a single session
    python -m nerve.validate_sessions --session 2023-10-26_15-37-59

    # Show passing sessions too (default: only problems)
    python -m nerve.validate_sessions --verbose

    # JSON output for automation
    python -m nerve.validate_sessions --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from nerve.config import get_data_root


# ── Expected session structure ──────────────────────────────────────────

REQUIRED_FILES = [
    "session_metadata.json",
    "timings.json",
]

REQUIRED_DIRS = [
    "davis",
    "prophesee",
    "rgb",
]

DAVIS_EXPECTED = {
    "files": ["events.hdf5"],
    "dirs": ["annotations"],
    "annotation_file": "annotations/annotations.json",
}

PROPHESEE_EXPECTED = {
    "files": ["events.hdf5"],
    "dirs": ["annotations"],
    "annotation_file": "annotations/annotations.json",
}

RGB_EXPECTED = {
    "dirs": ["annotations", "images"],
    "annotation_file": "annotations/annotations.json",
}

TI_RADAR_EXPECTED = {
    "dirs": ["captured_data"],
    "files_deep": ["captured_data/set000/data.h5"],
}

INFINEON_RADAR_EXPECTED = {
    "files": ["recording.xml"],
    "dirs": ["captured_data"],
    "files_deep": ["captured_data/set000/data.h5"],
}

DEPTH_FILES = [
    "L515_depth.mp4",
    "L515_depth_confidence.mp4",
    "L515_depth_unit.txt",
]

LEGACY_FILES_TO_FLAG = [
    ("evk4_events.hdf5", "should be moved to prophesee/events.hdf5 (run: python -m nerve.migrate_sessions)"),
    ("evk4_events.bias", "should be moved to prophesee/evk4_events.bias (run: python -m nerve.migrate_sessions)"),
]

# ── Allowlists for extra-file detection ─────────────────────────────────

ALLOWED_TOP_LEVEL = {
    "session_metadata.json",
    "timings.json",
    "metadata_cache.json",
    "L515_depth.mp4",
    "L515_depth_confidence.mp4",
    "L515_depth_unit.txt",
    "L515_rgb.txt",
    "radar_and_davis346_events.rad",
    "evk4_events.hdf5",
    "evk4_events.bias",
    "evk4_events.sync",
}

ALLOWED_TOP_DIRS = {
    "davis",
    "prophesee",
    "rgb",
    "ti_radar",
    "infineon_radar",
}

ALLOWED_DAVIS = {"events.hdf5", "annotations", "images"}
ALLOWED_PROPHESEE = {"events.hdf5", "evk4_events.bias", "annotations", "images"}
ALLOWED_RGB = {"annotations", "images"}
ALLOWED_TI_RADAR = {"captured_data", "meta_data", "recording.xml", "scenario.xml"}
ALLOWED_INFINEON_RADAR = {"captured_data", "meta_data", "recording.xml", "scenario.xml", "infineon_p2g.xml"}
ALLOWED_ANNOTATIONS = {"annotations.json"}


def _check_hdf5(path: Path) -> list[str]:
    """Basic HDF5 integrity check."""
    issues = []
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            if len(f.keys()) == 0:
                issues.append(f"  [warn] {path.name}: HDF5 file has no datasets")
    except ImportError:
        pass
    except Exception as e:
        issues.append(f"  [error] {path.name}: HDF5 cannot be opened: {e}")
    return issues


def _check_json(path: Path) -> list[str]:
    """Basic JSON integrity check."""
    issues = []
    try:
        file_size = path.stat().st_size
        if file_size == 0:
            issues.append(f"  [warn] {path.name}: JSON file is empty")
        elif file_size > 10 * 1024 * 1024:
            with open(path, "rb") as f:
                head = f.read(64)
                f.seek(-64, 2)
                tail = f.read(64)
            if not head.lstrip().startswith(b"{") and not head.lstrip().startswith(b"["):
                issues.append(f"  [warn] {path.name}: large file does not start with JSON object/array")
            if not tail.rstrip().endswith(b"}") and not tail.rstrip().endswith(b"]"):
                issues.append(f"  [warn] {path.name}: large file does not end with JSON close")
        else:
            with open(path) as f:
                data = json.load(f)
            if not data:
                issues.append(f"  [warn] {path.name}: JSON file is empty")
    except json.JSONDecodeError as e:
        issues.append(f"  [error] {path.name}: invalid JSON: {e}")
    except Exception as e:
        issues.append(f"  [error] {path.name}: cannot read: {e}")
    return issues


def _check_subdir(session: Path, subdir_name: str, expected: dict) -> list[str]:
    """Check a subdirectory against its expected structure."""
    issues = []
    subdir = session / subdir_name

    if not subdir.is_dir():
        issues.append(f"  [missing] {subdir_name}/ directory not found")
        return issues

    for fname in expected.get("files", []):
        fpath = subdir / fname
        if not fpath.exists():
            issues.append(f"  [missing] {subdir_name}/{fname}")
        elif fname.endswith(".hdf5") or fname.endswith(".h5"):
            issues.extend(_check_hdf5(fpath))

    for dname in expected.get("dirs", []):
        dpath = subdir / dname
        if not dpath.is_dir():
            issues.append(f"  [missing] {subdir_name}/{dname}/ directory")

    for deep_path in expected.get("files_deep", []):
        fpath = subdir / deep_path
        if not fpath.exists():
            issues.append(f"  [missing] {subdir_name}/{deep_path}")
        elif deep_path.endswith(".h5") or deep_path.endswith(".hdf5"):
            issues.extend(_check_hdf5(fpath))

    ann_file = expected.get("annotation_file")
    if ann_file:
        ann_path = subdir / ann_file
        if ann_path.exists():
            issues.extend(_check_json(ann_path))

    return issues


def _check_extra_files(directory: Path, allowed: set, prefix: str) -> list[str]:
    """Flag entries in *directory* that are not in the allowlist."""
    issues = []
    try:
        for entry in sorted(directory.iterdir()):
            if entry.name not in allowed:
                kind = "dir" if entry.is_dir() else "file"
                issues.append(f"  [extra] {prefix}{entry.name} (unexpected {kind})")
    except PermissionError:
        issues.append(f"  [error] {prefix}: cannot list directory (permission denied)")
    return issues


def validate_session(session: Path, registry_info: dict = None) -> list[str]:
    """Validate a single session directory.

    Returns a list of issue strings. Empty list means all checks pass.
    """
    issues = []

    # ── Required top-level files ────────────────────────────────────
    for fname in REQUIRED_FILES:
        fpath = session / fname
        if not fpath.exists():
            issues.append(f"  [missing] {fname}")
        elif fname.endswith(".json"):
            issues.extend(_check_json(fpath))

    # ── Legacy files that should have been migrated ─────────────────
    for fname, msg in LEGACY_FILES_TO_FLAG:
        if (session / fname).exists():
            issues.append(f"  [legacy] {fname}: {msg}")

    # ── Sensor directories ──────────────────────────────────────────
    sensors = {}
    if registry_info and "sensors_available" in registry_info:
        sensors = registry_info["sensors_available"]

    # DAVIS
    if sensors.get("davis346", True):
        issues.extend(_check_subdir(session, "davis", DAVIS_EXPECTED))

    # Prophesee
    if sensors.get("evk4", True):
        issues.extend(_check_subdir(session, "prophesee", PROPHESEE_EXPECTED))

    # RGB
    if sensors.get("l515_rgb", True):
        issues.extend(_check_subdir(session, "rgb", RGB_EXPECTED))

    # Depth
    if sensors.get("l515_depth", True):
        for fname in DEPTH_FILES:
            if not (session / fname).exists():
                issues.append(f"  [missing] {fname}")

    # TI Radar
    if sensors.get("ti_radar", True):
        issues.extend(_check_subdir(session, "ti_radar", TI_RADAR_EXPECTED))

    # Infineon Radar (optional -- only check if .rad source exists or dir exists)
    rad_file = session / "radar_and_davis346_events.rad"
    infineon_dir = session / "infineon_radar"
    if rad_file.exists() and not infineon_dir.exists():
        issues.append(
            f"  [missing] infineon_radar/: .rad file exists but radardb conversion "
            f"not done (run: python -m nerve.migrate_sessions)"
        )
    if infineon_dir.exists():
        issues.extend(_check_subdir(session, "infineon_radar", INFINEON_RADAR_EXPECTED))

    # ── Extra / unexpected files ─────────────────────────────────────
    issues.extend(_check_extra_files(session, ALLOWED_TOP_LEVEL | ALLOWED_TOP_DIRS, ""))

    davis_dir = session / "davis"
    if davis_dir.is_dir():
        issues.extend(_check_extra_files(davis_dir, ALLOWED_DAVIS, "davis/"))
        ann_dir = davis_dir / "annotations"
        if ann_dir.is_dir():
            issues.extend(_check_extra_files(ann_dir, ALLOWED_ANNOTATIONS, "davis/annotations/"))

    proph_dir = session / "prophesee"
    if proph_dir.is_dir():
        issues.extend(_check_extra_files(proph_dir, ALLOWED_PROPHESEE, "prophesee/"))
        ann_dir = proph_dir / "annotations"
        if ann_dir.is_dir():
            issues.extend(_check_extra_files(ann_dir, ALLOWED_ANNOTATIONS, "prophesee/annotations/"))

    rgb_dir = session / "rgb"
    if rgb_dir.is_dir():
        issues.extend(_check_extra_files(rgb_dir, ALLOWED_RGB, "rgb/"))
        ann_dir = rgb_dir / "annotations"
        if ann_dir.is_dir():
            issues.extend(_check_extra_files(ann_dir, ALLOWED_ANNOTATIONS, "rgb/annotations/"))

    ti_dir = session / "ti_radar"
    if ti_dir.is_dir():
        issues.extend(_check_extra_files(ti_dir, ALLOWED_TI_RADAR, "ti_radar/"))

    inf_dir = session / "infineon_radar"
    if inf_dir.is_dir():
        issues.extend(_check_extra_files(inf_dir, ALLOWED_INFINEON_RADAR, "infineon_radar/"))

    # ── Registry consistency checks ─────────────────────────────────
    if registry_info:
        if "split" not in registry_info or not registry_info["split"]:
            issues.append(f"  [registry] no split assigned")
        if "group" not in registry_info or not registry_info["group"]:
            issues.append(f"  [registry] no group assigned")

        # Check annotation counts match between registry and actual files.
        # Full JSON parsing is skipped for files >10 MB to avoid multi-minute
        # stalls on networked filesystems; use --skip-annotations to disable
        # this check entirely.
        if "annotations" in registry_info:
            for pov, ann_info in registry_info["annotations"].items():
                pov_lower = pov.lower()
                if pov_lower == "davis346":
                    ann_path = session / "davis" / "annotations" / "annotations.json"
                elif pov_lower == "evk4":
                    ann_path = session / "prophesee" / "annotations" / "annotations.json"
                elif pov_lower == "intel_l515":
                    ann_path = session / "rgb" / "annotations" / "annotations.json"
                else:
                    continue

                if ann_path.exists():
                    try:
                        if ann_path.stat().st_size > 10 * 1024 * 1024:
                            continue
                        with open(ann_path) as f:
                            coco = json.load(f)
                        actual_anns = len(coco.get("annotations", []))
                        expected_anns = ann_info.get("num_annotations", 0)
                        if actual_anns != expected_anns:
                            issues.append(
                                f"  [mismatch] {pov} annotations: registry says "
                                f"{expected_anns}, file has {actual_anns}"
                            )
                    except Exception:
                        pass

    return issues


def _is_session_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if (p / "session_metadata.json").exists():
        return True
    if (p / "davis").is_dir() or (p / "prophesee").is_dir():
        return True
    if (p / "timings.json").exists():
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate NERVE session directory structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--session", type=str, default=None,
        help="Validate only this session (name, not full path)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show passing sessions too",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--skip-annotations", action="store_true",
        help="Skip annotation count verification (faster)",
    )
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    if not data_root.is_dir():
        print(f"Data root does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)

    # Load registry for metadata cross-checks
    registry_path = Path(__file__).parent / "data" / "session_registry.json"
    registry = {}
    if registry_path.exists():
        with open(registry_path) as f:
            reg_data = json.load(f)
        registry = reg_data.get("sessions", {})

    # Discover sessions
    if args.session:
        sessions = [data_root / args.session]
        if not sessions[0].is_dir():
            for sub in ("train", "val", "test"):
                candidate = data_root / sub / args.session
                if candidate.is_dir():
                    sessions = [candidate]
                    break
            if not sessions[0].is_dir():
                print(f"Session directory not found: {sessions[0]}", file=sys.stderr)
                sys.exit(1)
    else:
        sessions = sorted(p for p in data_root.iterdir() if _is_session_dir(p))
        for sub in ("train", "val", "test"):
            sub_dir = data_root / sub
            if sub_dir.is_dir():
                sessions.extend(sorted(p for p in sub_dir.iterdir() if _is_session_dir(p)))

    if not sessions:
        print("No session directories found.")
        sys.exit(0)

    # Validate
    results = {}
    total_issues = 0
    sessions_with_issues = 0

    for session in sessions:
        reg_info = registry.get(session.name)
        if args.skip_annotations and reg_info:
            reg_info = {k: v for k, v in reg_info.items() if k != "annotations"}

        issues = validate_session(session, reg_info)
        results[session.name] = issues
        if issues:
            sessions_with_issues += 1
            total_issues += len(issues)

    # Output
    if args.json:
        json_out = {
            name: {"ok": len(iss) == 0, "issues": iss}
            for name, iss in results.items()
        }
        print(json.dumps(json_out, indent=2))
    else:
        if not args.verbose:
            print(f"Data root: {data_root}")
            print(f"Checking {len(sessions)} session(s)...\n")

        for name, issues in results.items():
            if issues:
                print(f"[FAIL] {name}")
                for iss in issues:
                    print(iss)
                print()
            elif args.verbose:
                print(f"[OK]   {name}")

        print("=" * 60)
        print(f"Sessions checked:     {len(sessions)}")
        print(f"Sessions with issues: {sessions_with_issues}")
        print(f"Total issues:         {total_issues}")
        if sessions_with_issues == 0:
            print("All sessions are valid.")


if __name__ == "__main__":
    main()
