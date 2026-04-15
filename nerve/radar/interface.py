"""
Abstract base class for radar backends.

To add support for a new radar system, subclass :class:`RadarBackend` and
implement all abstract methods.  Register your backend with
:func:`nerve.radar.register_backend` so that it can be retrieved via
:func:`nerve.radar.get_backend`.  See the radar walkthrough notebook for
conceptual background on the FMCW DSP pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class RadarBackend(ABC):
    """Abstract interface for reading radar recordings."""

    @classmethod
    @abstractmethod
    def from_recording(
        cls,
        recording_path: str | Path,
        capture_number: int = 0,
        radar_index: int = 0,
    ) -> "RadarBackend":
        """Open a radar recording directory.

        Args:
            recording_path: Path to the ``ti_radar/`` directory inside a session.
            capture_number: Capture set index (usually 0).
            radar_index: Radar device index when multiple radars are present.

        Returns:
            An initialized backend instance.
        """
        ...

    @abstractmethod
    def get_num_frames(self) -> int:
        """Return the total number of radar frames in the recording."""
        ...

    @abstractmethod
    def get_point_cloud(self, frame_idx: int) -> np.ndarray:
        """Extract the radar point cloud for a given frame.

        Returns:
            Array of shape ``(N, D)`` where N is the number of detected points
            and D contains at least ``[x, y, z, velocity, snr]``.
        """
        ...

    @abstractmethod
    def get_range_doppler(self, frame_idx: int) -> np.ndarray:
        """Extract the Range-Doppler map for a given frame.

        Returns:
            2D array of shape ``(range_bins, doppler_bins)`` with magnitude values.
        """
        ...

    @abstractmethod
    def get_raw_adc(self, frame_idx: int) -> np.ndarray:
        """Return the raw ADC data cube for a given frame.

        Returns:
            Array of shape ``(range_samples, tx_antennas, rx_antennas, chirps, 2)``
            where the last dimension is [I, Q].
        """
        ...

    @abstractmethod
    def get_frame_period(self) -> float:
        """Return the frame period in seconds.

        This is the time interval between consecutive radar frames as
        configured in the recording.  Used by the dataset generation
        pipeline for temporal alignment between sensors.

        Returns:
            Frame period in seconds (e.g. 0.1 for 10 fps).
        """
        ...

    def close(self) -> None:
        """Release resources. Override if the backend holds open files."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
