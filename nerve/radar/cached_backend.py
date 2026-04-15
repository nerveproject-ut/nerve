"""
Cached radar backend that reads precomputed point clouds and Range-Doppler
maps from an HDF5 file produced by :func:`nerve.radar.cache.build_cache`.

This backend lets users generate datasets without needing the proprietary
DSP library normally required by an active radar backend (e.g. PyCore).
The cache is produced once on a machine that does have the DSP library,
then shipped alongside the raw recording (or distributed independently).

Cache layout (``radar_cache.h5`` inside ``<session>/ti_radar/``)::

    /                                   (file root, attrs below)
        attrs:
            format_version   : int   (e.g. 1)
            num_frames       : int
            frame_period_s   : float
            point_cloud_dim  : int   (D in (N, D), usually 5)
            source_backend   : str   (e.g. "pycore", traceability)
    /point_clouds/
        <frame_idx_zfill_10> : float32 (N, D), one dataset per frame
    /range_doppler           : float32 (num_frames, R, D), gzip, optional

The Range-Doppler dataset is optional: a cache built with ``--no-fft``
omits it, in which case :meth:`get_range_doppler` returns an empty array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from nerve.radar.interface import RadarBackend


CACHE_FILENAME = "radar_cache.h5"
SUPPORTED_FORMAT_VERSIONS = frozenset({1})


class CachedBackend(RadarBackend):
    """RadarBackend that reads from a precomputed ``radar_cache.h5``.

    Use :meth:`from_recording` to open an existing cache.  The instance
    holds the underlying HDF5 file open until :meth:`close` is called
    (or the context manager exits), so do not keep many instances alive
    in parallel without closing them.
    """

    def __init__(self) -> None:
        self._h5: Optional[h5py.File] = None
        self._num_frames: int = 0
        self._frame_period: float = 0.0
        self._point_cloud_group: Optional[h5py.Group] = None
        self._range_doppler_dataset: Optional[h5py.Dataset] = None
        self._point_cloud_dim: int = 5
        self._cache_path: Optional[Path] = None

    @classmethod
    def from_recording(
        cls,
        recording_path,
        capture_number: int = 0,
        radar_index: int = 0,
    ) -> "CachedBackend":
        """Open a cache located at ``<recording_path>/radar_cache.h5``.

        ``capture_number`` and ``radar_index`` are accepted for interface
        compatibility but are currently ignored: a cache file always
        corresponds to a single ``(capture, radar_index)`` pair, fixed at
        precompute time.

        Raises:
            FileNotFoundError: If the cache file does not exist. Callers
                that want a fallback to a different backend should catch
                this exception (or use :func:`nerve.radar.open_recording`).
            ValueError: If the cache file has an unsupported
                ``format_version``.
        """
        recording_path = Path(recording_path)
        cache_path = recording_path / CACHE_FILENAME

        if not cache_path.is_file():
            raise FileNotFoundError(
                f"No radar cache found at {cache_path}.\n"
                f"Either generate one with "
                f"`nerve precompute-radar-cache <session>` (requires a "
                f"backend that can read the raw radar recording, e.g. "
                f"pycore), or have the dataset authors ship one alongside "
                f"the session."
            )

        instance = cls()
        instance._cache_path = cache_path
        instance._h5 = h5py.File(str(cache_path), "r")

        try:
            attrs = instance._h5.attrs
            version = int(attrs.get("format_version", 0))
            if version not in SUPPORTED_FORMAT_VERSIONS:
                raise ValueError(
                    f"Unsupported radar cache format_version={version} at "
                    f"{cache_path}. Supported versions: "
                    f"{sorted(SUPPORTED_FORMAT_VERSIONS)}. Re-run "
                    f"`nerve precompute-radar-cache` with the current "
                    f"toolkit to refresh the cache."
                )

            instance._num_frames = int(attrs.get("num_frames", 0))
            instance._frame_period = float(attrs.get("frame_period_s", 0.1))
            instance._point_cloud_dim = int(attrs.get("point_cloud_dim", 5))

            if "point_clouds" not in instance._h5:
                raise ValueError(
                    f"Radar cache {cache_path} is malformed: missing "
                    f"required group 'point_clouds'."
                )
            instance._point_cloud_group = instance._h5["point_clouds"]

            if "range_doppler" in instance._h5:
                instance._range_doppler_dataset = instance._h5["range_doppler"]
            else:
                instance._range_doppler_dataset = None
        except Exception:
            instance._h5.close()
            instance._h5 = None
            raise

        return instance

    def get_num_frames(self) -> int:
        return self._num_frames

    def get_frame_period(self) -> float:
        return self._frame_period

    def get_point_cloud(self, frame_idx: int) -> np.ndarray:
        if self._point_cloud_group is None:
            raise RuntimeError(
                "CachedBackend is not initialized. Use from_recording()."
            )
        if frame_idx < 0 or frame_idx >= self._num_frames:
            return np.empty((0, self._point_cloud_dim), dtype=np.float32)

        key = str(frame_idx).zfill(10)
        if key not in self._point_cloud_group:
            return np.empty((0, self._point_cloud_dim), dtype=np.float32)
        return self._point_cloud_group[key][...]

    def get_range_doppler(self, frame_idx: int) -> np.ndarray:
        if self._range_doppler_dataset is None:
            return np.empty((0, 0), dtype=np.float32)
        if frame_idx < 0 or frame_idx >= self._num_frames:
            return np.empty((0, 0), dtype=np.float32)
        return self._range_doppler_dataset[frame_idx]

    def get_raw_adc(self, frame_idx: int) -> np.ndarray:
        # Raw ADC is intentionally not cached: it is large and never
        # consumed by the dataset generation pipeline.
        return np.empty((0,), dtype=np.float32)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._point_cloud_group = None
        self._range_doppler_dataset = None
