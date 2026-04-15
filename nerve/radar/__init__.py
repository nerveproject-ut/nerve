"""
Abstract radar interface with pluggable backends.

Backends are registered via :func:`register_backend` and retrieved with
:func:`get_backend`.  When no backends have been explicitly registered the
module performs a one-time auto-discovery of any bundled backends whose
dependencies are available (e.g. an optional bundled backend adapter).

Usage::

    from nerve.radar import get_backend

    Backend = get_backend()                          # first registered backend
    Backend = get_backend("my_backend")              # by name
    radar   = Backend.from_recording("/path/to/ti_radar")

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
    """Try to import bundled backends whose dependencies are available."""
    global _AUTODISCOVERED
    if _AUTODISCOVERED:
        return
    _AUTODISCOVERED = True

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
              backend is returned (useful when only one is installed).

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
