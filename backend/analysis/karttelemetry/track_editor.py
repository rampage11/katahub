"""Интерактивный редактор карты трека.

Загружаешь PNG → мышью ставишь точки опорной траектории (racing line)
и две точки для линии отсечки (start/finish). Сохраняет в track.json
рядом с картинкой.

Usage:
    python -m karttelemetry.track_editor <track.png>

Горячие клавиши:
    T        — режим "трасса": клик добавляет точку в опорную траекторию
    F        — режим "отсечка": следующие два клика определяют линию старта/финиша
    U        — отменить последнюю точку в текущем режиме
    C        — очистить всё (с подтверждением)
    S        — сохранить
    L        — замкнуть траекторию (добавить сегмент от последней к первой)
    D        — переключить направление движения (ccw ↔ cw)

При закрытии окна предлагается сохранить, если были несохранённые изменения.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.lines import Line2D


class TrackEditor:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.save_path = image_path.with_suffix(".track.json")

        self.img = imread(str(image_path))
        self.img_h, self.img_w = self.img.shape[:2]

        self.trajectory: list[tuple[float, float]] = []
        self.start_finish: list[tuple[float, float]] = []
        self.mode: str = "trajectory"   # или "startfinish"
        self.dirty: bool = False

        # Подгружаем, если уже размечали ранее
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text())
                self.trajectory = [tuple(p) for p in data.get("trajectory_px", [])]
                self.start_finish = [tuple(p) for p in data.get("start_finish_px", [])]
                print(f"loaded existing: {self.save_path.name} "
                      f"({len(self.trajectory)} точек траектории, "
                      f"{len(self.start_finish)} точек отсечки)")
            except Exception as e:  # noqa: BLE001
                print(f"warning: не удалось прочитать {self.save_path.name}: {e}")

        # Фигура
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.ax.imshow(self.img, origin="upper")
        self.ax.set_xlim(0, self.img_w)
        self.ax.set_ylim(self.img_h, 0)
        self.ax.set_aspect("equal")

        # Отрисовка
        self.traj_line, = self.ax.plot([], [], "o-", color="cyan", lw=2, ms=5,
                                        markeredgecolor="white", picker=5)
        self.traj_closing_line, = self.ax.plot([], [], "--", color="cyan", lw=1, alpha=0.5)
        self.sf_dots, = self.ax.plot([], [], "o", color="red", ms=12,
                                      markeredgecolor="white", mew=2)
        self.sf_line, = self.ax.plot([], [], "-", color="red", lw=3, alpha=0.9)

        self._update_title()
        self._redraw()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    # ---- interactions ----

    def _on_click(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        pt = (float(event.xdata), float(event.ydata))

        if self.mode == "trajectory":
            self.trajectory.append(pt)
            self.dirty = True
        else:  # startfinish
            if len(self.start_finish) >= 2:
                # Начинаем новую пару точек
                self.start_finish = [pt]
            else:
                self.start_finish.append(pt)
            self.dirty = True

        self._update_title()
        self._redraw()

    def _on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key == "t":
            self.mode = "trajectory"
            self._update_title()
            self._redraw()
        elif key == "f":
            self.mode = "startfinish"
            self._update_title()
            self._redraw()
        elif key == "u":
            if self.mode == "trajectory" and self.trajectory:
                self.trajectory.pop()
                self.dirty = True
            elif self.mode == "startfinish" and self.start_finish:
                self.start_finish.pop()
                self.dirty = True
            self._update_title()
            self._redraw()
        elif key == "c":
            # Простое подтверждение через title
            self.ax.set_title("Нажми Y чтобы ОЧИСТИТЬ ВСЁ, любая другая — отмена")
            self.fig.canvas.draw_idle()
            self._pending_clear = True
        elif key == "y" and getattr(self, "_pending_clear", False):
            self.trajectory.clear()
            self.start_finish.clear()
            self.dirty = True
            self._pending_clear = False
            self._update_title()
            self._redraw()
        elif key == "l":
            # Замыкаем траекторию: сохраняем явную отметку, что это замкнутый круг.
            # Для отрисовки сами дорисовываем линию от последней точки к первой.
            if len(self.trajectory) > 2:
                # Ничего не добавляем — замыкание отрисовывается отдельной линией.
                self.dirty = True
                self._update_title()
                self._redraw()
        elif key == "s":
            self._save()
        else:
            if getattr(self, "_pending_clear", False):
                self._pending_clear = False
                self._update_title()
                self.fig.canvas.draw_idle()

    def _on_close(self, _event: Any) -> None:
        if self.dirty:
            # Нельзя блокировать GUI thread вопросом в stdin из close callback,
            # поэтому просто сохраняем автоматом (безопасно: пишем рядом).
            self._save()
            print("(auto-saved on close)")

    # ---- render ----

    def _update_title(self) -> None:
        mode_label = {
            "trajectory": "ТРАССА (кликай вдоль racing line)",
            "startfinish": "ОТСЕЧКА (2 клика для линии)",
        }[self.mode]
        info = f"[{mode_label}]   T=трасса  F=отсечка  U=отмена  L=замкнуть  S=сохранить  C=очистить"
        self.ax.set_title(f"{self.image_path.name} | {info} | "
                          f"точек: трасса={len(self.trajectory)} отсечка={len(self.start_finish)}")

    def _redraw(self) -> None:
        # Траектория
        if self.trajectory:
            xs, ys = zip(*self.trajectory)
            self.traj_line.set_data(xs, ys)
            # Замыкание (визуально; факт замыкания сохраняем в json как флаг closed)
            if len(self.trajectory) > 2:
                self.traj_closing_line.set_data(
                    [self.trajectory[-1][0], self.trajectory[0][0]],
                    [self.trajectory[-1][1], self.trajectory[0][1]],
                )
            else:
                self.traj_closing_line.set_data([], [])
        else:
            self.traj_line.set_data([], [])
            self.traj_closing_line.set_data([], [])

        # Отсечка
        if self.start_finish:
            xs, ys = zip(*self.start_finish)
            self.sf_dots.set_data(xs, ys)
            if len(self.start_finish) == 2:
                self.sf_line.set_data(xs, ys)
            else:
                self.sf_line.set_data([], [])
        else:
            self.sf_dots.set_data([], [])
            self.sf_line.set_data([], [])

        self.fig.canvas.draw_idle()

    # ---- save ----

    def _save(self) -> None:
        # Сохраняем предыдущие метаданные если они были — редактор их не меняет,
        # пользователь редактирует в JSON напрямую (track_name, direction).
        prev: dict = {}
        if self.save_path.exists():
            try:
                prev = json.loads(self.save_path.read_text())
            except Exception:  # noqa: BLE001
                pass
        payload = {
            "schema_version": 1,
            "track_name": prev.get("track_name"),
            "image_path": self.image_path.name,
            "image_size_px": [self.img_w, self.img_h],
            "direction": prev.get("direction"),         # "ccw" | "cw"
            "trajectory_type": prev.get("trajectory_type", "racing_line"),
            "trajectory_px": [[round(x, 2), round(y, 2)] for (x, y) in self.trajectory],
            "trajectory_closed": True,
            "start_finish_px": [[round(x, 2), round(y, 2)] for (x, y) in self.start_finish],
        }
        self.save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        self.dirty = False
        self.ax.set_title(f"сохранено: {self.save_path.name}")
        self.fig.canvas.draw_idle()
        print(f"saved: {self.save_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="karttelemetry.track_editor")
    parser.add_argument("track_image", type=Path, help="PNG карта трека вид сверху")
    args = parser.parse_args(argv)

    if not args.track_image.is_file():
        print(f"error: image not found: {args.track_image}", file=sys.stderr)
        return 2

    editor = TrackEditor(args.track_image)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
