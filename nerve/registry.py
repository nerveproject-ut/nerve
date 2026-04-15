"""
Session registry for the NERVE dataset.

Provides query and filter access to all 116 recording sessions and the
utils archive, backed by a bundled JSON file containing download metadata
(UUID, MD5, size) and per-session sensor/annotation statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional, Union


_REGISTRY: dict | None = None


def _load_registry() -> dict:
    global _REGISTRY
    if _REGISTRY is None:
        registry_path = files("nerve.data").joinpath("session_registry.json")
        _REGISTRY = json.loads(registry_path.read_text(encoding="utf-8"))
    return _REGISTRY


@dataclass
class SessionInfo:
    """Structured view of a single session's registry entry."""

    name: str
    uuid: str
    md5: str
    size_bytes: int
    split: str
    group: str
    duration_seconds: float = 0.0
    start_time: str = ""
    sensors_available: dict[str, bool] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    sensor_details: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @classmethod
    def from_entry(cls, name: str, entry: dict) -> SessionInfo:
        dl = entry.get("download", {})
        return cls(
            name=name,
            uuid=dl.get("uuid", ""),
            md5=dl.get("md5", ""),
            size_bytes=dl.get("size_bytes", 0),
            split=entry.get("split", "unassigned"),
            group=entry.get("group", ""),
            duration_seconds=entry.get("duration_seconds", 0.0),
            start_time=entry.get("start_time", ""),
            sensors_available=entry.get("sensors_available", {}),
            annotations=entry.get("annotations", {}),
            sensor_details=entry.get("sensor_details", {}),
            aggregate=entry.get("aggregate", {}),
        )


def all_sessions(
    data_root: str | Path | None = None,
) -> list[SessionInfo]:
    """Return a list of all sessions in the registry.

    If *data_root* is provided (or ``NERVE_DATA_ROOT`` is set), locally
    cached metadata (duration, sensors, annotations) is merged in.
    """
    reg = _load_registry()

    cache: dict[str, dict] = {}
    try:
        from nerve.metadata import load_cache
        cache = load_cache(data_root)
    except Exception:
        pass

    sessions = []
    for name, entry in sorted(reg["sessions"].items()):
        merged = dict(entry)
        if name in cache:
            merged.update(cache[name])
        sessions.append(SessionInfo.from_entry(name, merged))
    return sessions


def get_session(name: str) -> SessionInfo:
    """Look up a single session by name.

    Raises:
        KeyError: If the session name is not in the registry.
    """
    reg = _load_registry()
    if name not in reg["sessions"]:
        raise KeyError(
            f"Session '{name}' not found in registry. "
            f"Use nerve.registry.all_sessions() to list available sessions."
        )
    return SessionInfo.from_entry(name, reg["sessions"][name])


def get_sessions(
    split: str | None = None,
    data_root: str | Path | None = None,
) -> list[SessionInfo]:
    """Return sessions, optionally filtered by split (train/val/test)."""
    sessions = all_sessions(data_root=data_root)
    if split is not None:
        sessions = [s for s in sessions if s.split == split]
    return sessions


def get_utils_info() -> dict:
    """Return download metadata for the utils.tar.gz archive."""
    reg = _load_registry()
    return reg.get("utils", {})


def filter_sessions(
    *,
    split: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    min_persons: int | None = None,
    categories: list[str] | None = None,
    sensors: list[str] | None = None,
    groups: list[str] | None = None,
    max_size_gb: float | None = None,
    names: list[str] | None = None,
    data_root: str | Path | None = None,
) -> list[SessionInfo]:
    """Filter sessions by metadata criteria.

    All criteria are AND-combined. Pass only the filters you need.

    Args:
        split: Restrict to a specific split (train/val/test).
        min_duration: Minimum recording duration in seconds.
        max_duration: Maximum recording duration in seconds.
        min_persons: Minimum number of unique persons detected (aggregate).
        categories: Session must contain annotations for all these categories
                    in at least one POV.
        sensors: Session must have all these sensors available. Valid names:
                 davis346, evk4, ti_radar, l515_rgb, l515_depth.
        groups: Restrict to specific recording groups (e.g. ["2023_10_26"]).
        max_size_gb: Maximum archive size in GB.
        names: Restrict to a specific list of session names.

    Returns:
        List of SessionInfo objects matching all criteria.
    """
    sessions = all_sessions(data_root=data_root)

    if split is not None:
        sessions = [s for s in sessions if s.split == split]

    if names is not None:
        name_set = set(names)
        sessions = [s for s in sessions if s.name in name_set]

    if groups is not None:
        group_set = set(groups)
        sessions = [s for s in sessions if s.group in group_set]

    if min_duration is not None:
        sessions = [s for s in sessions if s.duration_seconds >= min_duration]

    if max_duration is not None:
        sessions = [s for s in sessions if s.duration_seconds <= max_duration]

    if max_size_gb is not None:
        sessions = [s for s in sessions if s.size_gb <= max_size_gb]

    if min_persons is not None:
        sessions = [
            s for s in sessions
            if s.aggregate.get("unique_persons_detected", 0) >= min_persons
        ]

    if categories is not None:
        def has_categories(session: SessionInfo) -> bool:
            for pov_data in session.annotations.values():
                cats = set(pov_data.get("categories", {}).keys())
                if all(c in cats for c in categories):
                    return True
            return False
        sessions = [s for s in sessions if has_categories(s)]

    if sensors is not None:
        def has_sensors(session: SessionInfo) -> bool:
            avail = session.sensors_available
            return all(avail.get(sensor, False) for sensor in sensors)
        sessions = [s for s in sessions if has_sensors(s)]

    return sessions


def session_names(
    sessions: list[SessionInfo] | None = None,
    data_root: str | Path | None = None,
) -> list[str]:
    """Extract session names from a list, or return all names."""
    if sessions is None:
        sessions = all_sessions(data_root=data_root)
    return [s.name for s in sessions]


def total_size(sessions: list[SessionInfo]) -> float:
    """Return total download size in GB for a list of sessions."""
    return sum(s.size_bytes for s in sessions) / 1e9
