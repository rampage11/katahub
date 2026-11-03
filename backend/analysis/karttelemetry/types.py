"""Core-типы pipeline.

Используются laps/, corners/, comparison/ модулями после того как у нас появятся
реальные сессии и начнётся разработка детектора кругов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class LapCoordinate:
    """Нормализованная кривая прогресса одного круга.

    `s` — безразмерный прогресс 0..1 вдоль круга, не метры. Это принципиально:
    пока нет надёжного источника дистанции (GPS/VO/wheel speed indoor нет),
    сравниваем круги по относительной фазе, не по абсолютной геометрии трека.
    Когда появится track map (из визуального odometry или вручную размеченная),
    маппинг s→distance_m добавится без переделки downstream API.
    """

    lap_id: int
    t_ns: np.ndarray            # абсолютные таймстемпы в нс каждого семпла круга
    s: np.ndarray               # нормализованный прогресс 0..1 той же длины что t_ns
    distance_m: np.ndarray | None = None  # появится позже когда известна геометрия


@dataclass
class CornerSegment:
    """Один поворот внутри круга, в LapCoordinate-пространстве."""

    corner_id: int
    direction: Literal["left", "right"]

    # Все позиции — в нормализованном прогрессе 0..1 того круга к которому относится.
    entry_s: float
    turn_in_s: float
    apex_s: float
    exit_s: float

    braking_start_s: float | None = None
    braking_end_s: float | None = None


@dataclass
class CornerMetrics:
    """Количественные метрики одного прохождения поворота."""

    corner_id: int

    braking_duration_s: float | None = None
    braking_peak_g: float | None = None
    braking_jerk: float | None = None

    rotation_peak_rads: float | None = None
    rotation_duration_s: float | None = None

    lateral_g_peak: float | None = None
    lateral_g_mean: float | None = None

    exit_accel: float | None = None

    confidence: float = 1.0


@dataclass
class LapDelta:
    """Отрезочная разница lap A vs reference lap, в секундах.

    Positive = lost time, negative = gained. Confidence отражает насколько
    надёжно aligned оба круга в данном сегменте (DTW cost, length ratio и т.п.).
    """

    corner_id: int
    delta_s: float
    confidence: float


@dataclass
class ClockSync:
    """Модель рассинхронизации часов между источниками: t_aligned = a*t + b.

    За 20-минутную сессию дрейф 50ppm накапливает ~60мс — заметно для corner
    analysis. Храним per-source чтобы при merge iPhone/Watch/Audio/RaceMann
    можно было точно совместить таймлайны.
    """

    source: str                 # "iphone" | "watch" | "audio" | "racemann"
    offset_ns: int = 0
    drift_ppm: float = 0.0

    def apply(self, t_ns: np.ndarray) -> np.ndarray:
        a = 1.0 + self.drift_ppm * 1e-6
        return (t_ns.astype(np.int64) * a).astype(np.int64) + self.offset_ns
