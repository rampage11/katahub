"""Loaders for session artifacts: manifest, IMU, HR, audio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Структура одной IMU записи, 88 байт, little-endian; соответствует Swift
# IMUSampleBinary: 1×UInt64 + 10×Double, без хвостового padding
# (проверено MemoryLayout: size == stride == 88).
IMU_DTYPE: np.dtype = np.dtype(
    [
        ("ts_ns", "<u8"),
        ("ax", "<f8"),
        ("ay", "<f8"),
        ("az", "<f8"),
        ("gx", "<f8"),
        ("gy", "<f8"),
        ("gz", "<f8"),
        ("qw", "<f8"),
        ("qx", "<f8"),
        ("qy", "<f8"),
        ("qz", "<f8"),
    ]
)
assert IMU_DTYPE.itemsize == 88, "IMU dtype must be exactly 88 bytes"

HR_DTYPE: np.dtype = np.dtype(
    [
        ("ts_ns", "<u8"),
        ("bpm", "<f8"),
    ]
)
assert HR_DTYPE.itemsize == 16, "HR dtype must be exactly 16 bytes"


def load_manifest(session_dir: Path) -> dict[str, Any]:
    """Load and parse manifest.json from a session directory."""
    session_dir = Path(session_dir)
    manifest_path = session_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_imu(path: Path) -> np.ndarray:
    """Load a packed IMU binary file into a structured ndarray.

    Returns an array with named fields: ts_ns, ax, ay, az, gx, gy, gz,
    qw, qx, qy, qz.
    """
    path = Path(path)
    return np.fromfile(path, dtype=IMU_DTYPE)


def load_hr(path: Path) -> np.ndarray:
    """Load heart rate binary into a structured ndarray with (ts_ns, bpm)."""
    path = Path(path)
    return np.fromfile(path, dtype=HR_DTYPE)


def load_audio(path: Path) -> tuple[int, np.ndarray]:
    """Load audio using soundfile. Returns (samplerate, data)."""
    import soundfile as sf  # ленивая загрузка: soundfile тянет libsndfile

    path = Path(path)
    data, samplerate = sf.read(str(path), always_2d=False)
    return int(samplerate), np.asarray(data)


def load_timing(path: Path) -> dict[str, Any]:
    """Load timing.json artifact with lap/flag events from RaceMann."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_session(session_dir: Path, *, load_audio_data: bool = True) -> dict[str, Any]:
    """Load everything available in a session directory.

    Returns a dict with keys: 'session_dir', 'manifest', 'iphone_imu',
    'watch_imu', 'heart_rate', 'audio', 'timing' (any may be None).
    If `load_audio_data` is False, audio is skipped even if present.

    Файлы определяются по имени на диске, не по manifest — на iOS бывает,
    что watch-файлы прилетают пост-stop и не попадают в manifest.sources.
    """
    session_dir = Path(session_dir)
    manifest = load_manifest(session_dir)

    out: dict[str, Any] = {
        "session_dir": session_dir,
        "manifest": manifest,
        "iphone_imu": None,
        "watch_imu": None,
        "heart_rate": None,
        "audio": None,
        "timing": None,
    }

    # Фиксированные имена файлов → (ключ в out, функция-загрузчик)
    file_map = {
        "iphone_imu.bin": ("iphone_imu", load_imu),
        "watch_imu.bin": ("watch_imu", load_imu),
        "heart_rate.bin": ("heart_rate", load_hr),
    }

    for filename, (key, fn) in file_map.items():
        file_path = session_dir / filename
        if not file_path.exists():
            continue
        try:
            out[key] = fn(file_path)
        except Exception as e:  # noqa: BLE001
            out[key] = None
            out.setdefault("_errors", []).append(f"{key}: {e}")

    audio_path = session_dir / "iphone_audio.wav"
    if load_audio_data and audio_path.exists():
        try:
            out["audio"] = load_audio(audio_path)
        except Exception as e:  # noqa: BLE001
            out["audio"] = None
            out.setdefault("_errors", []).append(f"audio: {e}")

    timing_path = session_dir / "timing.json"
    if timing_path.exists():
        try:
            out["timing"] = load_timing(timing_path)
        except Exception as e:  # noqa: BLE001
            out["timing"] = None
            out.setdefault("_errors", []).append(f"timing: {e}")

    return out
