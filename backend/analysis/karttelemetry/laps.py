"""Lap detection scaffolding.

Current implementation is a placeholder: it looks for prominent peaks in the
yaw-rate signal and groups consecutive peaks that look like corners.

The real algorithm will come later once we have ground-truth sessions:
likely a combination of track-shape fingerprint matching (via accumulated
heading from gyro + accel), audio engine-note cue at the start/finish line,
and periodicity detection in gyro magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class LapCandidate:
    """A candidate lap identified by a sequence of corner peaks."""

    start_idx: int
    end_idx: int
    start_ts_ns: int
    end_ts_ns: int
    peak_indices: tuple[int, ...]


def detect_laps_from_gyro(
    imu: np.ndarray,
    candidate_turn_axis: Literal["x", "y", "z"] = "z",
    *,
    min_samples: int = 500,
    peak_abs_threshold: float = 2.0,
    min_corner_spacing_samples: int = 50,
    corners_per_lap: int = 4,
) -> list[LapCandidate]:
    """Return a list of lap candidates based on gyro peaks on one axis.

    Placeholder implementation: finds peaks on |omega_axis| with
    scipy.signal.find_peaks, then slices every `corners_per_lap` peaks into a
    lap. Returns [] if data is too short or scipy is unavailable.
    """
    if imu is None or imu.size < min_samples:
        return []

    axis_field = {"x": "gx", "y": "gy", "z": "gz"}[candidate_turn_axis]
    try:
        from scipy.signal import find_peaks
    except Exception:  # noqa: BLE001
        return []

    signal = np.abs(imu[axis_field])
    peaks, _ = find_peaks(
        signal,
        height=peak_abs_threshold,
        distance=min_corner_spacing_samples,
    )
    if peaks.size < corners_per_lap:
        return []

    laps: list[LapCandidate] = []
    ts = imu["ts_ns"]
    for i in range(0, peaks.size - corners_per_lap + 1, corners_per_lap):
        group = peaks[i : i + corners_per_lap]
        s_idx = int(group[0])
        e_idx = int(group[-1])
        laps.append(
            LapCandidate(
                start_idx=s_idx,
                end_idx=e_idx,
                start_ts_ns=int(ts[s_idx]),
                end_ts_ns=int(ts[e_idx]),
                peak_indices=tuple(int(x) for x in group),
            )
        )
    return laps
