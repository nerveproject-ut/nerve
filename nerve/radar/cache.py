"""
Build precomputed radar caches from any registered :class:`RadarBackend`.

A radar cache is a single HDF5 file containing:

* per-frame point clouds (variable-N, fixed-D),
* per-frame Range-Doppler maps (fixed shape, optional),
* session-level scalars (``num_frames``, ``frame_period_s``).

Once written, the cache lets a downstream user run dataset generation
without needing the original radar DSP library, by registering
:class:`nerve.radar.cached_backend.CachedBackend` (which is auto-discovered
by :mod:`nerve.radar`).

Typical usage::

    from nerve.radar.cache import build_cache

    build_cache("/data/sessions/2023-12-15_15-02-22/ti_radar",
                backend_name="pycore",
                include_range_doppler=True)

The cache file is written atomically as ``radar_cache.h5`` inside the
recording directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np

from nerve.radar.cached_backend import CACHE_FILENAME


FORMAT_VERSION = 1


def build_cache(
    recording_path,
    *,
    backend_name: Optional[str] = None,
    include_range_doppler: bool = True,
    force: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Build a ``radar_cache.h5`` inside the given recording directory.

    Args:
        recording_path: Path to the recording directory (e.g.
            ``<session>/ti_radar``).
        backend_name: Backend identifier to use as the data source.  If
            ``None``, the first registered non-cached backend is used.
        include_range_doppler: If ``True``, also cache Range-Doppler
            maps.  Set to ``False`` to skip them and produce a much
            smaller cache (point cloud only) when the FFT is not needed
            downstream.
        force: Overwrite any existing cache at the target path.
        progress_callback: Optional callable taking ``(current, total)``
            for progress reporting.

    Returns:
        Path to the generated cache file.

    Raises:
        FileExistsError: If a cache already exists and ``force`` is
            ``False``.
        NotADirectoryError: If ``recording_path`` does not point to a
            directory.
        RuntimeError: If no source backend is available.
        ValueError: If ``backend_name`` is unknown or refers to the
            cached backend itself.
    """
    # Imported lazily to avoid an import cycle with nerve.radar.__init__.
    from nerve.radar import _REGISTRY, _autodiscover
    from nerve.radar.cached_backend import CachedBackend

    recording_path = Path(recording_path)
    if not recording_path.is_dir():
        raise NotADirectoryError(
            f"Radar recording directory does not exist: {recording_path}"
        )

    cache_path = recording_path / CACHE_FILENAME
    if cache_path.exists() and not force:
        raise FileExistsError(
            f"{cache_path} already exists. Pass force=True (or --force on "
            f"the CLI) to overwrite."
        )

    _autodiscover()
    candidate_names = [
        n for n, c in _REGISTRY.items() if c is not CachedBackend
    ]

    if backend_name is not None:
        if backend_name not in _REGISTRY:
            raise ValueError(
                f"Backend '{backend_name}' is not registered. "
                f"Available backends: {list(_REGISTRY)}."
            )
        if _REGISTRY[backend_name] is CachedBackend:
            raise ValueError(
                "Cannot build a cache from the cached backend itself. "
                "Choose a source backend that can read raw radar data "
                f"(e.g. one of {candidate_names})."
            )
        chosen_name = backend_name
    else:
        if not candidate_names:
            raise RuntimeError(
                "No source radar backend is available. Install one (e.g. "
                "pycore via the proprietary DSP library) or pass "
                "backend_name explicitly."
            )
        chosen_name = candidate_names[0]

    backend_cls = _REGISTRY[chosen_name]
    backend = backend_cls.from_recording(str(recording_path))

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        num_frames = int(backend.get_num_frames())
        frame_period = float(backend.get_frame_period())

        if num_frames <= 0:
            raise RuntimeError(
                f"Source backend reported {num_frames} frames for "
                f"{recording_path}. Refusing to build an empty cache."
            )

        first_pc = np.asarray(backend.get_point_cloud(0))
        pc_dim = (
            int(first_pc.shape[1])
            if first_pc.ndim == 2 and first_pc.shape[1] > 0
            else 5
        )

        rd_shape: Optional[tuple] = None
        first_rd = None
        if include_range_doppler:
            first_rd = np.asarray(backend.get_range_doppler(0))
            if first_rd.size > 0 and first_rd.ndim == 2:
                rd_shape = tuple(first_rd.shape)
            else:
                # Source backend doesn't expose Range-Doppler for this
                # recording; gracefully skip without failing.
                first_rd = None

        with h5py.File(str(tmp_path), "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["num_frames"] = int(num_frames)
            f.attrs["frame_period_s"] = float(frame_period)
            f.attrs["source_backend"] = str(chosen_name)
            f.attrs["point_cloud_dim"] = int(pc_dim)

            pc_group = f.create_group("point_clouds")

            if rd_shape is not None:
                rd_dataset = f.create_dataset(
                    "range_doppler",
                    shape=(num_frames,) + rd_shape,
                    dtype=np.float32,
                    chunks=(1,) + rd_shape,
                    compression="gzip",
                    compression_opts=4,
                )
            else:
                rd_dataset = None

            _write_frame_pc(pc_group, 0, first_pc, pc_dim)
            if rd_dataset is not None and first_rd is not None:
                rd_dataset[0] = np.asarray(first_rd, dtype=np.float32)

            if progress_callback is not None:
                progress_callback(1, num_frames)

            for i in range(1, num_frames):
                pc = np.asarray(backend.get_point_cloud(i))
                _write_frame_pc(pc_group, i, pc, pc_dim)

                if rd_dataset is not None:
                    rd = np.asarray(backend.get_range_doppler(i))
                    if rd.shape == rd_shape:
                        rd_dataset[i] = rd.astype(np.float32, copy=False)
                    else:
                        # FMCW recordings normally have a fixed FFT shape;
                        # pad with zeros if the source ever changes shape
                        # mid-recording rather than aborting.
                        rd_dataset[i] = 0.0

                if progress_callback is not None:
                    progress_callback(i + 1, num_frames)
    finally:
        try:
            backend.close()
        except Exception:
            pass

    tmp_path.replace(cache_path)
    return cache_path


def _write_frame_pc(
    group: h5py.Group,
    idx: int,
    pc: np.ndarray,
    pc_dim: int,
) -> None:
    """Write a single per-frame point cloud dataset."""
    key = str(idx).zfill(10)

    if pc.ndim != 2 or pc.size == 0:
        group.create_dataset(
            key, shape=(0, pc_dim), dtype=np.float32
        )
        return

    # Defensive: align with declared point_cloud_dim if upstream gives a
    # variable D (shouldn't happen with a sane backend, but be safe).
    if pc.shape[1] != pc_dim:
        if pc.shape[1] < pc_dim:
            pad = np.zeros(
                (pc.shape[0], pc_dim - pc.shape[1]),
                dtype=np.float32,
            )
            pc = np.concatenate(
                [pc.astype(np.float32, copy=False), pad], axis=1
            )
        else:
            pc = pc[:, :pc_dim].astype(np.float32, copy=False)

    group.create_dataset(
        key,
        data=pc.astype(np.float32, copy=False),
        compression="gzip",
        compression_opts=4,
    )
