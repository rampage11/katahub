"""Интерактивный overlay траекторий сессии на карту трека.

Usage:
    python -m karttelemetry.overlay <session_dir> <track_image.png>

Открывает matplotlib-окно:
- Фон: карта трека
- Поверх: IMU-траектории, по кругу на цвет (lap boundaries из timing.json;
  если timing.json нет — рисуется одна общая кривая)
- Слайдеры: scale (пикселей на метр), rotation (°), offset_x/offset_y (px),
  speed_assumed (м/с — для yaw-based shape)
- Кнопка "Сохранить": пишет alignment.json в session_dir;
  при повторном запуске эти значения подхватываются как начальные.

Траектория — yaw-based shape (постоянная скорость, направление из
attitude.yaw). Это принципиальный выбор для MVP: двойной интеграл
userAccel без GPS/wheel speed накапливает слишком большую ошибку
за 30+ секунд круга.

Для корректной геометрии нужно знать "forward" direction относительно
startup iPhone, но на треке pilot ставит телефон под комбез — ориентация
неизвестна. Поэтому offset_x/y + rotation позволяют вручную подогнать.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.image import imread

from . import loader


def quat_to_yaw(w, x, y, z):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _unwrap_yaw(imu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_seconds_from_zero, unwrapped_yaw_rad)."""
    t_ns = imu["ts_ns"].astype(np.int64)
    t = (t_ns - t_ns[0]) / 1e9
    yaw = quat_to_yaw(imu["qw"], imu["qx"], imu["qy"], imu["qz"])
    return t, np.unwrap(yaw)


def trajectory_in_meters(
    imu: np.ndarray,
    speed_mps: float,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Yaw-based shape в метрах, система координат — начало в (0,0)."""
    if end_idx is None:
        end_idx = len(imu)
    sub = imu[start_idx:end_idx]
    if len(sub) < 2:
        return np.array([0.0]), np.array([0.0])
    t_ns = sub["ts_ns"].astype(np.int64)
    t = (t_ns - t_ns[0]) / 1e9
    dt = np.diff(t, prepend=t[0])
    yaw = quat_to_yaw(sub["qw"], sub["qx"], sub["qy"], sub["qz"])
    dx = speed_mps * np.cos(yaw) * dt
    dy = speed_mps * np.sin(yaw) * dt
    return np.cumsum(dx), np.cumsum(dy)


def split_laps_by_timing(imu: np.ndarray, timing: dict | None) -> list[slice]:
    """Границы кругов по timing.json. Fallback — одна слайс на всё."""
    if not timing or not timing.get("laps"):
        return [slice(0, len(imu))]

    # timing.laps[i].wallTime — ISO datetime. imu['ts_ns'] — ns от boot.
    # На iPhone timestamp — mach_absolute_time в ns от boot (не wall clock),
    # поэтому напрямую сопоставить wallTime и ts_ns сложно. Для MVP берём
    # пропорции: нормализуем позиции lap boundaries по общей длительности
    # сессии между manifest.startedAt..endedAt vs ts_ns диапазоном.
    #
    # Это грубо, но работает: timing.wallTime у laps размечает когда
    # круги прошли относительно начала сессии, и ts_ns относительно
    # первого семпла то же самое (просто в разных часах).
    import datetime as _dt

    def parse_iso(s: str) -> _dt.datetime:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(s)

    laps = timing["laps"]
    # Берём первый wallTime как 0, остальные — delta в секундах.
    t0 = parse_iso(laps[0]["wallTime"])
    deltas = [(parse_iso(l["wallTime"]) - t0).total_seconds() for l in laps]

    # Нормализация imu time
    t_ns = imu["ts_ns"].astype(np.int64)
    imu_t = (t_ns - t_ns[0]) / 1e9

    # Lap i: от previous boundary до deltas[i].
    # Первую "pre-lap" часть сессии (от 0 до deltas[0]) трактуем как outlap.
    bounds_s = [0.0] + deltas
    slices: list[slice] = []
    for i in range(len(bounds_s) - 1):
        start_s = bounds_s[i]
        end_s = bounds_s[i + 1]
        start_idx = int(np.searchsorted(imu_t, start_s))
        end_idx = int(np.searchsorted(imu_t, end_s))
        if end_idx > start_idx + 1:
            slices.append(slice(start_idx, end_idx))
    return slices if slices else [slice(0, len(imu))]


def load_alignment(session_dir: Path) -> dict[str, float]:
    path = session_dir / "alignment.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_alignment(session_dir: Path, params: dict[str, float]) -> None:
    path = session_dir / "alignment.json"
    path.write_text(json.dumps(params, indent=2, sort_keys=True))


def apply_affine(x_m: np.ndarray, y_m: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """m → px: scale (px/m), rotate (deg), translate (px)."""
    s = params.get("scale_px_per_m", 10.0)
    rot = np.radians(params.get("rotation_deg", 0.0))
    ox = params.get("offset_x_px", 0.0)
    oy = params.get("offset_y_px", 0.0)
    c, si = np.cos(rot), np.sin(rot)
    xr = (c * x_m - si * y_m) * s + ox
    yr = (si * x_m + c * y_m) * s + oy
    return xr, yr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="karttelemetry.overlay")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("track_image", type=Path)
    args = parser.parse_args(argv)

    if not args.session_dir.is_dir():
        print(f"error: not a directory: {args.session_dir}", file=sys.stderr)
        return 2
    if not args.track_image.is_file():
        print(f"error: image not found: {args.track_image}", file=sys.stderr)
        return 2

    session = loader.load_session(args.session_dir, load_audio_data=False)
    imu = session["iphone_imu"]
    timing = session.get("timing")

    if imu is None:
        print("error: iphone_imu.bin отсутствует", file=sys.stderr)
        return 2

    img = imread(str(args.track_image))
    img_h, img_w = img.shape[:2]

    # Начальные параметры: пробуем из alignment.json; иначе разумные дефолты.
    initial = load_alignment(args.session_dir)
    params = {
        "scale_px_per_m": initial.get("scale_px_per_m", img_w / 200.0),  # 200м длина трека по умолчанию
        "rotation_deg": initial.get("rotation_deg", 0.0),
        "offset_x_px": initial.get("offset_x_px", img_w / 2),
        "offset_y_px": initial.get("offset_y_px", img_h / 2),
        "speed_mps": initial.get("speed_mps", 11.0),  # ~40 km/h типичная indoor скорость
    }

    lap_slices = split_laps_by_timing(imu, timing)
    print(f"сегментов для отрисовки: {len(lap_slices)}")

    # ---- UI ----
    fig = plt.figure(figsize=(14, 9))
    ax_map = fig.add_axes([0.05, 0.25, 0.9, 0.7])
    ax_map.imshow(img, origin="upper")
    ax_map.set_xlim(0, img_w)
    ax_map.set_ylim(img_h, 0)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"{args.session_dir.name}  ({len(lap_slices)} laps/segments)")

    cmap = plt.get_cmap("viridis")
    lines = []
    for i, sl in enumerate(lap_slices):
        x_m, y_m = trajectory_in_meters(imu, params["speed_mps"], sl.start, sl.stop)
        xr, yr = apply_affine(x_m, y_m, params)
        color = cmap(i / max(1, len(lap_slices) - 1))
        (line,) = ax_map.plot(xr, yr, lw=1.5, color=color, label=f"lap {i+1}")
        lines.append((line, sl))
    if len(lap_slices) <= 12:
        ax_map.legend(loc="upper right", fontsize=8)

    # Слайдеры
    axcolor = "lightgoldenrodyellow"
    slider_defs = [
        ("scale_px_per_m", 0.1, 50.0, params["scale_px_per_m"]),
        ("rotation_deg", -180.0, 180.0, params["rotation_deg"]),
        ("offset_x_px", 0.0, float(img_w), params["offset_x_px"]),
        ("offset_y_px", 0.0, float(img_h), params["offset_y_px"]),
        ("speed_mps", 1.0, 30.0, params["speed_mps"]),
    ]
    sliders = {}
    for i, (key, vmin, vmax, vinit) in enumerate(slider_defs):
        y = 0.18 - i * 0.03
        ax_sl = fig.add_axes([0.12, y, 0.75, 0.02], facecolor=axcolor)
        s = Slider(ax_sl, key, vmin, vmax, valinit=vinit)
        sliders[key] = s

    def update(_val: float) -> None:
        for key, s in sliders.items():
            params[key] = s.val
        for line, sl in lines:
            x_m, y_m = trajectory_in_meters(imu, params["speed_mps"], sl.start, sl.stop)
            xr, yr = apply_affine(x_m, y_m, params)
            line.set_data(xr, yr)
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update)

    ax_save = fig.add_axes([0.88, 0.02, 0.1, 0.04])
    btn_save = Button(ax_save, "Сохранить")

    def on_save(_event: Any) -> None:
        save_alignment(args.session_dir, dict(params))
        print(f"сохранено: {args.session_dir / 'alignment.json'}")
        btn_save.label.set_text("Сохранено!")
        fig.canvas.draw_idle()

    btn_save.on_clicked(on_save)

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
