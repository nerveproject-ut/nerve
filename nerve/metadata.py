"""
Local metadata cache for NERVE sessions.

Each session archive contains a ``session_metadata.json`` with rich
per-sensor and annotation information (duration, sensors, categories, etc.).
This module reads those files from locally extracted sessions and maintains
a lightweight JSON cache so that ``nerve list`` and the registry can display
metadata without re-scanning the filesystem every time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from nerve.config import get_data_root

CACHE_FILENAME = "metadata_cache.json"


def _cache_path(data_root: Optional[Union[str, Path]] = None) -> Path:
    return get_data_root(data_root) / CACHE_FILENAME


def load_cache(data_root: Optional[Union[str, Path]] = None) -> dict[str, dict]:
    """Load the metadata cache from disk.

    Returns:
        Mapping of session name -> metadata dict.  Empty dict if no cache exists.
    """
    path = _cache_path(data_root)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(
    cache: dict[str, dict],
    data_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Persist the metadata cache to disk.

    Returns:
        Path to the cache file.
    """
    path = _cache_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_session_metadata(session_dir: Path) -> Optional[dict[str, Any]]:
    """Read ``session_metadata.json`` from an extracted session directory.

    Returns:
        Parsed metadata dict, or ``None`` if the file doesn't exist.
    """
    meta_file = session_dir / "session_metadata.json"
    if not meta_file.is_file():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _extract_cache_entry(raw_meta: dict[str, Any]) -> dict[str, Any]:
    """Distill the full session_metadata.json into the fields we cache.

    The metadata uses a SensorML/OGC-namespaced JSON schema.  Key paths::

        duration   -> sml:System / sml:validTime / gml:TimePeriod / duration_seconds
        start_time -> sml:System / sml:validTime / gml:TimePeriod / gml:beginPosition
        sensors    -> swe:DataRecord / sensors_available
        aggregate  -> swe:DataRecord / aggregate_statistics
        components -> sml:System / sml:components[]  (list of sml:Component)
    """
    system = raw_meta.get("sml:System", {})
    time_period = (
        system
        .get("sml:validTime", {})
        .get("gml:TimePeriod", {})
    )
    duration = time_period.get("duration_seconds", 0.0)
    start_time = time_period.get("gml:beginPosition", "")

    top_data = raw_meta.get("swe:DataRecord", {})
    sensors_available = top_data.get("sensors_available", {})
    aggregate = top_data.get("aggregate_statistics", {})

    annotations: dict[str, Any] = {}
    sensor_details: dict[str, Any] = {}

    for component_wrapper in system.get("sml:components", []):
        comp = component_wrapper.get("sml:Component", {})
        name = comp.get("gml:name", "")
        if not name:
            continue

        ann = comp.get("swe:DataRecord", {}).get("annotations")
        if ann:
            annotations[name] = ann

        chars = comp.get("sml:characteristics", {})
        obs = comp.get("sml:observedProperty", {})
        detail: dict[str, Any] = {}
        for field in ("manufacturer", "model", "sensor_type",
                      "resolution_width", "resolution_height"):
            if field in chars:
                detail[field] = chars[field]
        if obs:
            detail["observation"] = obs
        if detail:
            sensor_details[name] = detail

    return {
        "duration_seconds": duration,
        "start_time": start_time,
        "sensors_available": sensors_available,
        "annotations": annotations,
        "sensor_details": sensor_details,
        "aggregate": aggregate,
    }


def cache_session(
    session_name: str,
    session_dir: Path,
    data_root: Optional[Union[str, Path]] = None,
) -> bool:
    """Read metadata from a single extracted session and add it to the cache.

    Returns:
        ``True`` if metadata was found and cached, ``False`` otherwise.
    """
    raw = read_session_metadata(session_dir)
    if raw is None:
        return False

    cache = load_cache(data_root)
    cache[session_name] = _extract_cache_entry(raw)
    save_cache(cache, data_root)
    return True


def _find_session_dir(
    session_name: str,
    roots: list[Path],
) -> Optional[Path]:
    """Locate an extracted session directory across multiple roots.

    Checks each root directly (``root/session_name``) and also inside
    ``train/``, ``val/``, ``test/`` subdirectories so that sessions
    organised by split are discovered automatically.
    """
    for root in roots:
        candidate = root / session_name
        if (candidate / "session_metadata.json").is_file():
            return candidate
        for split in ("train", "val", "test"):
            candidate = root / split / session_name
            if (candidate / "session_metadata.json").is_file():
                return candidate
    return None


def enrich_from_local(
    data_root: Optional[Union[str, Path]] = None,
    extra_roots: Optional[list[Union[str, Path]]] = None,
    verbose: bool = False,
) -> int:
    """Scan all locally extracted sessions and rebuild the metadata cache.

    Session directories are searched in the primary *data_root* and any
    *extra_roots*, both directly and inside ``train/``, ``val/``, ``test/``
    subdirectories.

    Args:
        data_root: Primary data root (resolved via config if ``None``).
        extra_roots: Additional directories to scan for extracted sessions.
        verbose: Print progress to stdout.

    Returns:
        Number of sessions whose metadata was found and cached.
    """
    from nerve.registry import all_sessions

    root = get_data_root(data_root)
    search_roots = [root]
    if extra_roots:
        search_roots.extend(Path(r).expanduser().resolve() for r in extra_roots)

    cache = load_cache(data_root)
    count = 0

    for session in all_sessions():
        if session.name in cache and cache[session.name].get("duration_seconds", 0) > 0:
            count += 1
            continue

        session_dir = _find_session_dir(session.name, search_roots)
        if session_dir is not None:
            raw = read_session_metadata(session_dir)
            if raw is not None:
                cache[session.name] = _extract_cache_entry(raw)
                count += 1
                if verbose:
                    dur = cache[session.name].get("duration_seconds", 0)
                    print(f"  Cached {session.name}  ({dur:.1f}s)")
                continue

        if verbose:
            print(f"  Skipped {session.name}  (not downloaded)")

    save_cache(cache, data_root)
    return count
