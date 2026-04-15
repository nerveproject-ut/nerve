"""
Abstract radar interface with pluggable backends.

Backends are registered via :func:`register_backend` and retrieved with
:func:`get_backend`.  When no backends have been explicitly registered the
module performs a one-time auto-discovery of any bundled backends whose
dependencies are available.

Two backends are bundled:

* :class:`nerve.radar.cached_backend.CachedBackend` -- always available;
  reads precomputed point clouds and Range-Doppler maps from a
  ``radar_cache.h5`` file.  Built once with
  ``nerve precompute-radar-cache``.
* :class:`nerve.radar.pycore_backend.PyCoreBackend` -- reads raw TI
  AWR1443 recordings via a proprietary DSP library; only registered if
  that library is installed.

Auto-discovery registers ``cached`` first so that
:func:`open_recording` (and :func:`get_backend` with no name) prefers a
ready-made cache when one exists.  Use :func:`open_recording` if you
want automatic fallback to the next registered backend when a cache is
absent::

    from nerve.radar import open_recording

    radar = open_recording("/path/to/session/ti_radar")

Custom backends can be registered at any time::

    from nerve.radar import register_backend, RadarBackend

    class MyBackend(RadarBackend):
        ...

    register_backend("my_backend", MyBackend)
"""

from __future__ import annotations

from nerve.radar.interface import RadarBackend

_REGISTRY: dict[str, type[RadarBackend]] = {}
_AUTODISCOVERED = False


def register_backend(name: str, cls: type[RadarBackend]) -> None:
    """Register a :class:`RadarBackend` subclass under *name*.

    Args:
        name: Short identifier used to look up the backend later.
        cls:  A concrete subclass of :class:`RadarBackend`.

    Raises:
        TypeError: If *cls* is not a subclass of :class:`RadarBackend`.
    """
    if not (isinstance(cls, type) and issubclass(cls, RadarBackend)):
        raise TypeError(
            f"Expected a RadarBackend subclass, got {cls!r}"
        )
    _REGISTRY[name] = cls


def _autodiscover() -> None:
    """Try to import bundled backends whose dependencies are available.

    The cached backend is registered first so that it becomes the
    default when no name is given to :func:`get_backend` or
    :func:`open_recording`.  This is intentional: a cache hit is cheap
    and produces results identical to the source backend, while
    falling back to the source backend only requires raising
    :class:`FileNotFoundError` from the cached backend's
    ``from_recording``.
    """
    global _AUTODISCOVERED
    if _AUTODISCOVERED:
        return
    _AUTODISCOVERED = True

    try:
        from nerve.radar.cached_backend import CachedBackend  # noqa: F811
        register_backend("cached", CachedBackend)
    except Exception:
        pass

    try:
        from nerve.radar.pycore_backend import PyCoreBackend  # noqa: F811
        register_backend("pycore", PyCoreBackend)
    except Exception:
        pass


def available_backends() -> list[str]:
    """Return the names of all registered backends.

    Triggers auto-discovery on first call.
    """
    _autodiscover()
    return list(_REGISTRY.keys())


def get_backend(name: str | None = None) -> type[RadarBackend]:
    """Get a registered :class:`RadarBackend` class.

    Args:
        name: Backend identifier.  If ``None``, the first registered
              backend is returned (cached, after auto-discovery).

    Returns:
        The :class:`RadarBackend` subclass (not an instance).

    Raises:
        ImportError: If the requested backend's dependencies are missing.
        ValueError: If *name* is unknown or no backends are registered.
    """
    _autodiscover()

    if name is not None:
        if name not in _REGISTRY:
            avail = ", ".join(_REGISTRY) if _REGISTRY else "<none>"
            raise ValueError(
                f"Unknown radar backend '{name}'. "
                f"Registered backends: {avail}. "
                f"You can also subclass nerve.radar.RadarBackend directly "
                f"and call register_backend()."
            )
        return _REGISTRY[name]

    if not _REGISTRY:
        raise ValueError(
            "No radar backends are registered. Install a backend package "
            "or subclass nerve.radar.RadarBackend and call "
            "register_backend()."
        )
    return next(iter(_REGISTRY.values()))


def open_recording(
    recording_path,
    *,
    capture_number: int = 0,
    radar_index: int = 0,
    name: str | None = None,
) -> RadarBackend:
    """Open a radar recording, with automatic backend fallback.

    If *name* is provided, only that backend is tried (no fallback).
    Otherwise each registered backend is tried in registration order
    (cached first, then any source backends), and the first one whose
    ``from_recording`` succeeds is returned.  ``FileNotFoundError`` is
    treated as "this backend cannot open this recording, try the
    next one"; any other exception propagates immediately.

    Args:
        recording_path: Path to the recording directory (e.g.
            ``<session>/ti_radar``).
        capture_number: Capture set index (default 0).
        radar_index: Radar device index (default 0).
        name: Optional backend identifier.  If given, no fallback.

    Returns:
        An initialized :class:`RadarBackend` instance.

    Raises:
        ValueError: If ``name`` is given but unknown.
        FileNotFoundError: If no backend could open the recording.
        ImportError: Propagated from the chosen backend if its
            dependencies are missing and no other backend can serve.
    """
    _autodiscover()

    if name is not None:
        backend_cls = get_backend(name)
        return backend_cls.from_recording(
            recording_path,
            capture_number=capture_number,
            radar_index=radar_index,
        )

    if not _REGISTRY:
        raise ValueError(
            "No radar backends are registered. Install a backend package "
            "or subclass nerve.radar.RadarBackend and call "
            "register_backend()."
        )

    # Always try the cached backend first (when registered), regardless of
    # registration order: a cache hit is cheap and produces results
    # identical to the source backend.  All other backends keep their
    # registration order as the fallback chain.
    ordered = [
        (n, c) for n, c in _REGISTRY.items() if n == "cached"
    ] + [
        (n, c) for n, c in _REGISTRY.items() if n != "cached"
    ]

    last_error: Exception | None = None
    tried: list[str] = []
    for backend_name, backend_cls in ordered:
        tried.append(backend_name)
        try:
            return backend_cls.from_recording(
                recording_path,
                capture_number=capture_number,
                radar_index=radar_index,
            )
        except FileNotFoundError as e:
            last_error = e
            continue

    raise FileNotFoundError(
        f"No registered radar backend could open {recording_path!r}. "
        f"Tried: {tried}. Last error: {last_error}"
    )
