"""
Remote download client for the NERVE dataset hosted on 4TU.ResearchData.

Uses Referer-header authentication discovered via API exploration:
the 4TU server grants access to individual files when the HTTP request
includes the private dataset page URL as the Referer header.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from nerve.config import (
    ARTICLE_UUID,
    FILE_URL_TEMPLATE,
    REFERER,
    get_data_root,
)


def _file_download_url(file_uuid: str) -> str:
    return FILE_URL_TEMPLATE.format(file_uuid=file_uuid)


def _verify_md5(path: Path, expected_md5: str) -> bool:
    if not expected_md5:
        return True
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() == expected_md5


def download_session(
    name: str,
    data_root: str | Path | None = None,
    *,
    extract: bool = True,
    verify: bool = True,
    keep_archive: bool = False,
) -> Path:
    """Download and optionally extract a single session archive.

    Args:
        name: Session name (e.g. ``"2023-10-26_15-34-07"``).
        data_root: Directory to store data. Defaults to ``NERVE_DATA_ROOT``.
        extract: Whether to extract the .tar.gz after downloading.
        verify: Whether to verify the MD5 checksum.
        keep_archive: If True, keep the .tar.gz after extraction.

    Returns:
        Path to the extracted session directory (or the archive if not extracted).
    """
    from nerve.registry import get_session

    root = get_data_root(data_root)
    root.mkdir(parents=True, exist_ok=True)

    info = get_session(name)
    session_dir = root / name

    if session_dir.is_dir():
        print(f"  Session already exists: {session_dir}")
        return session_dir

    archive_path = root / f"{name}.tar.gz"
    url = _file_download_url(info.uuid)

    if not archive_path.exists():
        _download_file(url, archive_path, expected_size=info.size_bytes)

    if verify and info.md5:
        if not _verify_md5(archive_path, info.md5):
            archive_path.unlink()
            raise RuntimeError(
                f"MD5 mismatch for {name}. Archive deleted. Try again."
            )

    if extract:
        _extract_archive(archive_path, root)
        if not keep_archive:
            archive_path.unlink(missing_ok=True)

        from nerve.metadata import cache_session
        cache_session(name, session_dir, data_root=data_root)

        return session_dir

    return archive_path


def download_sessions(
    names: list[str],
    data_root: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    """Download multiple sessions sequentially.

    Args:
        names: List of session names.
        data_root: Target directory.
        **kwargs: Forwarded to :func:`download_session`.

    Returns:
        List of paths to extracted session directories.
    """
    from nerve.registry import total_size, get_session

    sessions = [get_session(n) for n in names]
    total_gb = total_size(sessions)
    print(f"Downloading {len(names)} sessions ({total_gb:.1f} GB total)")

    paths = []
    for name in names:
        print(f"\n[{len(paths)+1}/{len(names)}] {name}")
        p = download_session(name, data_root=data_root, **kwargs)
        paths.append(p)

    print(f"\nDone. {len(paths)} sessions downloaded.")
    return paths


def download_split(
    split: str,
    data_root: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    """Download all sessions for a given split (train/val/test).

    Args:
        split: One of ``"train"``, ``"val"``, ``"test"``.
        data_root: Target directory.
        **kwargs: Forwarded to :func:`download_session`.
    """
    from nerve.registry import get_sessions

    sessions = get_sessions(split=split)
    names = [s.name for s in sessions]
    print(f"Split '{split}': {len(names)} sessions")
    return download_sessions(names, data_root=data_root, **kwargs)


def download_from_file(
    path: str | Path,
    data_root: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    """Download sessions listed in a plain-text file.

    Args:
        path: Path to a session list file (one name per line).
        data_root: Target directory.
        **kwargs: Forwarded to :func:`download_session`.
    """
    from nerve.session_list import read_session_list

    names = read_session_list(path)
    print(f"Session list '{path}': {len(names)} sessions")
    return download_sessions(names, data_root=data_root, **kwargs)


def download_filtered(
    data_root: str | Path | None = None,
    *,
    dry_run: bool = False,
    **filter_kwargs,
) -> list[Path]:
    """Download sessions matching metadata filter criteria.

    Args:
        data_root: Target directory.
        dry_run: If True, print what would be downloaded without downloading.
        **filter_kwargs: Forwarded to :func:`nerve.registry.filter_sessions`.

    Returns:
        List of paths (empty list if dry_run).
    """
    from nerve.registry import filter_sessions, total_size

    sessions = filter_sessions(data_root=data_root, **filter_kwargs)

    if dry_run:
        gb = total_size(sessions)
        print(f"Would download {len(sessions)} sessions ({gb:.1f} GB):")
        for s in sessions:
            print(f"  {s.name}  ({s.size_gb:.2f} GB, {s.split})")
        return []

    names = [s.name for s in sessions]
    return download_sessions(names, data_root=data_root)


def download_utils(data_root: str | Path | None = None) -> Path:
    """Download and extract the utils.tar.gz archive.

    Contains mapping JSONs, timings, split lists, and dataset statistics.

    Returns:
        Path to the extracted utils/ directory.
    """
    from nerve.registry import get_utils_info

    root = get_data_root(data_root)
    root.mkdir(parents=True, exist_ok=True)

    info = get_utils_info()
    utils_dir = root / "utils"

    if utils_dir.is_dir():
        print(f"  Utils already exists: {utils_dir}")
        return utils_dir

    archive_path = root / "utils.tar.gz"
    url = _file_download_url(info["uuid"])

    if not archive_path.exists():
        _download_file(url, archive_path, expected_size=info.get("size_bytes", 0))

    if info.get("md5") and not _verify_md5(archive_path, info["md5"]):
        archive_path.unlink()
        raise RuntimeError("MD5 mismatch for utils.tar.gz. Deleted. Try again.")

    _extract_archive(archive_path, root)
    archive_path.unlink(missing_ok=True)
    return utils_dir


def _download_file(url: str, dest: Path, expected_size: int = 0) -> None:
    """Download a file with progress bar and Referer auth.

    Supports resuming partially downloaded files via HTTP Range requests.
    Falls back to a full re-download if the server does not support ranges.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {"Referer": REFERER}

    resume_pos = 0
    if dest.exists():
        resume_pos = dest.stat().st_size
        if expected_size and resume_pos >= expected_size:
            return
        headers["Range"] = f"bytes={resume_pos}-"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)
    resp.raise_for_status()

    if resume_pos and resp.status_code != 206:
        # Server ignored the Range header — restart from scratch
        resume_pos = 0

    total = int(resp.headers.get("content-length", 0)) + resume_pos
    mode = "ab" if resume_pos else "wb"

    with (
        open(dest, mode) as f,
        tqdm(
            total=total,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into dest_dir."""
    print(f"  Extracting {archive_path.name} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
