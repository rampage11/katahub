"""Matplotlib plotting primitives for a loaded session."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - only triggers if matplotlib missing at import
    plt = None  # type: ignore[assignment]
    Axes = Any  # type: ignore[assignment,misc]
    Figure = Any  # type: ignore[assignment,misc]


def _seconds_from_ts(ts_ns: np.ndarray) -> np.ndarray:
    """Convert ns timestamps into seconds from the first sample."""
    if ts_ns.size == 0:
        return np.empty(0, dtype=np.float64)
    t0 = ts_ns[0]
    return (ts_ns.astype(np.int64) - np.int64(t0)) / 1e9


def plot_g_force(imu: np.ndarray, ax: Axes | None = None, *, source: str = "iPhone") -> Axes:
    """Plot the magnitude of user acceleration (g) over time."""
    if ax is None:
        _, ax = plt.subplots()

    if imu is None or imu.size == 0:
        ax.set_title(f"G-force — {source} (no data)")
        ax.set_xlabel("t, s")
        ax.set_ylabel("|a|, g")
        return ax

    t = _seconds_from_ts(imu["ts_ns"])
    mag = np.sqrt(imu["ax"] ** 2 + imu["ay"] ** 2 + imu["az"] ** 2)
    ax.plot(t, mag, lw=0.7)

    i_max = int(np.argmax(mag))
    i_min = int(np.argmin(mag))
    ax.scatter([t[i_max]], [mag[i_max]], color="red", s=20, zorder=3)
    ax.scatter([t[i_min]], [mag[i_min]], color="green", s=20, zorder=3)
    ax.annotate(
        f"max {mag[i_max]:.2f}g",
        xy=(t[i_max], mag[i_max]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )
    ax.annotate(
        f"min {mag[i_min]:.2f}g",
        xy=(t[i_min], mag[i_min]),
        xytext=(5, -10),
        textcoords="offset points",
        fontsize=8,
    )

    ax.set_title(f"G-force magnitude — {source}")
    ax.set_xlabel("t, s")
    ax.set_ylabel("|a|, g")
    ax.grid(True, alpha=0.3)
    return ax


def plot_rotation(imu: np.ndarray, ax: Axes | None = None, *, source: str = "iPhone") -> Axes:
    """Plot magnitude of angular velocity (rad/s) over time."""
    if ax is None:
        _, ax = plt.subplots()

    if imu is None or imu.size == 0:
        ax.set_title(f"Rotation — {source} (no data)")
        ax.set_xlabel("t, s")
        ax.set_ylabel("|omega|, rad/s")
        return ax

    t = _seconds_from_ts(imu["ts_ns"])
    mag = np.sqrt(imu["gx"] ** 2 + imu["gy"] ** 2 + imu["gz"] ** 2)
    ax.plot(t, mag, lw=0.7, color="C1")

    i_max = int(np.argmax(mag))
    ax.scatter([t[i_max]], [mag[i_max]], color="red", s=20, zorder=3)
    ax.annotate(
        f"max {mag[i_max]:.2f} rad/s",
        xy=(t[i_max], mag[i_max]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

    ax.set_title(f"Rotation rate — {source}")
    ax.set_xlabel("t, s")
    ax.set_ylabel("|omega|, rad/s")
    ax.grid(True, alpha=0.3)
    return ax


def plot_heart_rate(hr: np.ndarray, ax: Axes | None = None) -> Axes:
    """Plot BPM over time. HR timestamps use epoch ns, we show seconds from first."""
    if ax is None:
        _, ax = plt.subplots()

    if hr is None or hr.size == 0:
        ax.set_title("Heart rate (no data)")
        ax.set_xlabel("t, s")
        ax.set_ylabel("bpm")
        return ax

    t = _seconds_from_ts(hr["ts_ns"])
    ax.plot(t, hr["bpm"], marker="o", ms=3, lw=0.8, color="C3")
    ax.set_title("Heart rate")
    ax.set_xlabel("t, s")
    ax.set_ylabel("bpm")
    ax.grid(True, alpha=0.3)
    return ax


def plot_audio_spectrogram(
    samplerate: int,
    data: np.ndarray,
    ax: Axes | None = None,
    *,
    f_max: float = 2000.0,
) -> Axes:
    """Plot a log-magnitude spectrogram up to f_max Hz."""
    if ax is None:
        _, ax = plt.subplots()

    if data is None or len(data) == 0 or samplerate <= 0:
        ax.set_title("Audio spectrogram (no data)")
        ax.set_xlabel("t, s")
        ax.set_ylabel("f, Hz")
        return ax

    # Если несколько каналов, усредняем в моно
    if data.ndim > 1:
        signal = data.mean(axis=1)
    else:
        signal = data

    # Параметры подобраны под речевой/моторный диапазон до 2 кГц;
    # NFFT и hop выбраны так, чтобы разрешение было ~22 Гц при 44.1 кГц.
    nfft = 2048
    noverlap = nfft // 2
    Pxx, freqs, bins, im = ax.specgram(
        signal.astype(np.float64),
        NFFT=nfft,
        Fs=samplerate,
        noverlap=noverlap,
        scale="dB",
        cmap="viridis",
    )
    ax.set_ylim(0, f_max)
    ax.set_title(f"Audio spectrogram (0-{int(f_max)} Hz)")
    ax.set_xlabel("t, s")
    ax.set_ylabel("f, Hz")
    return ax


def summary_figure(session: dict[str, Any]) -> Figure:
    """Build a 2x3 figure summarising a loaded session.

    Row 1: iPhone G-force, iPhone rotation, Audio spectrogram
    Row 2: Watch G-force,  Watch rotation,  Heart rate
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    iphone_imu = session.get("iphone_imu")
    watch_imu = session.get("watch_imu")
    hr = session.get("heart_rate")
    audio = session.get("audio")

    plot_g_force(iphone_imu, ax=axes[0, 0], source="iPhone")
    plot_rotation(iphone_imu, ax=axes[0, 1], source="iPhone")
    if audio is not None:
        sr, data = audio
        plot_audio_spectrogram(sr, data, ax=axes[0, 2])
    else:
        axes[0, 2].set_title("Audio spectrogram (no data)")
        axes[0, 2].set_xlabel("t, s")
        axes[0, 2].set_ylabel("f, Hz")

    plot_g_force(watch_imu, ax=axes[1, 0], source="Watch")
    plot_rotation(watch_imu, ax=axes[1, 1], source="Watch")
    plot_heart_rate(hr, ax=axes[1, 2])

    manifest = session.get("manifest") or {}
    sid = manifest.get("id", "?")
    device = manifest.get("deviceModel") or manifest.get("deviceName") or ""
    fig.suptitle(f"Session {sid}  {device}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
