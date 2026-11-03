"""Загрузка и валидация шаблона трассы (v2).

Шаблон трассы — это НЕ часть pipeline'а, он строится отдельно (admin op,
`build_template.py`) из хорошей сессии + аннотации, и хранится в
`tracks/<track>/` как три файла:

- `tmpl.npy`      — wz-профиль одного круга @100Hz, shape=(N,), ~33.17s
- `track.npz`     — x, y, v, wz @50Hz, shape=(N/2,). Используется для карты
                    и для сравнения геометрии (bounds, кусок кумулятивного
                    пути).
- `features.npy`  — DTW features `T` для шаблона, shape=(N/2, 2). Колонки:
                    [wz/1.2, 0.5·z(|a_h|)]. Нужны для `dtw_closed(T, F_lap[::2])`.

В run-time pipeline загружает трассу (по умолчанию MIKS) один раз, не
пересчитывает `T`. Если `features.npy` отсутствует — pipeline напечатает
warning и упадёт с явной ошибкой (чтобы руки написали `build_template.py`).

Константы ниже вынесены из reference `corners.py`:
- CORN  — окна поворотов в template-секундах (центр ± полвысоты)
- CUTS_S — границы 7 секторов (между собой + начало/конец круга — 8 точек,
          6 внутренних cuts)

Все метрики (slip, hf band, brake len etc.) считаются в CORN-окнах; все
sector-времена — в CUTS_S-окнах. Если шаблон будет пересобран на новой
сессии, CORN/CUTS_S надо обновить вместе с ним (admin op также
валидирует согласованность).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------- Константы MIKS ----------
# Окна поворотов (template-секунды, из handoff2 reference/corners.py).
# Полный круг MIKS = 33.17 s, 11 поворотов (T1..T11).
CORN_MIKS: dict[str, tuple[float, float]] = {
    "T1":  (0.64, 2.64),
    "T2":  (2.84, 4.88),
    "T3":  (6.10, 7.34),
    "T4":  (8.54, 10.54),
    "T5":  (13.72, 15.36),
    "T6":  (15.98, 18.64),
    "T7":  (19.62, 22.26),
    "T8":  (23.02, 24.84),
    "T9":  (25.08, 28.52),
    "T10": (28.92, 30.54),
    "T11": (31.68, 33.16),
}

# Сектор-разделители (template-секунды). 6 внутренних → 7 секторов S1..S7.
CUTS_S_MIKS: list[float] = [2.76, 8.38, 15.82, 19.42, 24.98, 28.78]

# Соответствие сектор → список поворотов внутри него (для UI).
# Выведено из CUTS_S + CORN серединок поворотов (см. скрипт в history).
SECTOR_CORNERS_MIKS: list[list[str]] = [
    ["T1"],
    ["T2", "T3"],
    ["T4", "T5"],
    ["T6"],
    ["T7", "T8"],
    ["T9"],
    ["T10", "T11"],
]

# Фаз-классификатор (v2): три порога постоянны, нижний (brake cut) — на
# сессию через Otsu, clamp [-0.55, -0.40] g.
PHASE_CUT_MIN = -0.55
PHASE_CUT_MAX = -0.40
PHASE_LIFT_UPPER_G = -0.12   # верхняя граница "сброс/лёгкий трейл"
PHASE_ACCEL_LOWER_G = 0.20   # нижняя граница "газ"

# Названия фаз в UI (Russian; renamed per v2 spec):
#   brake  → тормоз
#   lift   → сброс / лёгкий трейл  (prev "сброс газа")
#   accel  → газ
#   cruise → ровно                 (prev "качение")
PHASE_LABELS_RU: dict[str, str] = {
    "brake":  "тормоз",
    "lift":   "сброс / лёгкий трейл",
    "accel":  "газ",
    "cruise": "ровно",
}


@dataclass
class TrackTemplate:
    """Данные шаблона трассы. Все массивы — numpy."""

    name: str
    tmpl_wz_100: np.ndarray     # (N_100,) wz @100Hz
    x_50: np.ndarray            # (N_50,) координаты круга @50Hz
    y_50: np.ndarray
    v_50: np.ndarray            # (N_50,) скорость @50Hz, м/с
    wz_50: np.ndarray           # (N_50,) wz @50Hz
    tt_50: np.ndarray | None    # (N_50,) cumulative time @50Hz (optional)
    features_50: np.ndarray     # (N_50, 2) DTW features [wz/1.2, 0.5·z(|a_h|)]
    corn: dict[str, tuple[float, float]]
    cuts_s: list[float]
    sector_corners: list[list[str]]
    fs_tmpl: float = 100.0
    fs_track: float = 50.0

    @property
    def lap_duration_s(self) -> float:
        return len(self.tmpl_wz_100) / self.fs_tmpl


def _resolve_track_dir(name: str) -> Path:
    """Ищет директорию трассы. Приоритет: $TRACKS_DIR env, затем repo-local.

    Repo-local используется в dev; prod backend ставит TRACKS_DIR=/app/tracks
    (см. Dockerfile).
    """
    import os
    env = os.environ.get("TRACKS_DIR")
    if env:
        p = Path(env) / name
        if p.exists():
            return p
    # Repo-local: analysis/tracks/<name>/
    here = Path(__file__).resolve().parent.parent
    local = here / "tracks" / name
    if local.exists():
        return local
    # Backend-local: analysis/../tracks/<name>/ (backend layout)
    backend = here.parent / "tracks" / name
    if backend.exists():
        return backend
    raise FileNotFoundError(
        f"track '{name}' not found; tried $TRACKS_DIR={env!r}, {local}, {backend}"
    )


def load_track_template(name: str = "miks") -> TrackTemplate:
    """Полная загрузка шаблона трассы.

    Если `features.npy` отсутствует — выбрасывает FileNotFoundError с
    инструкцией собрать его через `build_template.py`.
    """
    d = _resolve_track_dir(name)
    tmpl_path = d / "tmpl.npy"
    track_path = d / "track.npz"
    feat_path = d / "features.npy"

    if not tmpl_path.exists() or not track_path.exists():
        raise FileNotFoundError(
            f"track {name}: missing tmpl.npy or track.npz in {d}"
        )

    tmpl = np.load(tmpl_path).astype(np.float64)
    track = np.load(track_path)
    x = track["x"].astype(np.float64)
    y = track["y"].astype(np.float64)
    v = track["v"].astype(np.float64)
    wz_50 = track["wz"].astype(np.float64)
    tt_50 = track["tt"].astype(np.float64) if "tt" in track.files else None

    if not feat_path.exists():
        raise FileNotFoundError(
            f"track {name}: missing features.npy in {d}. "
            f"Rebuild with `python3 analysis/build_template.py --track {name} "
            f"--dir <good_session>` and commit the result."
        )
    features = np.load(feat_path).astype(np.float64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(
            f"track {name}: features.npy must be (N, 2), got {features.shape}"
        )
    if features.shape[0] != len(wz_50):
        raise ValueError(
            f"track {name}: features len {features.shape[0]} != track.npz "
            f"len {len(wz_50)}"
        )

    # Константы CORN/CUTS_S захардкожены под MIKS; для других трасс потом
    # придётся вынести в metadata файл внутри track dir.
    if name != "miks":
        # Пока только MIKS; остальные трассы — для будущего.
        raise NotImplementedError(f"CORN/CUTS_S hardcoded for MIKS, got '{name}'")

    return TrackTemplate(
        name=name,
        tmpl_wz_100=tmpl,
        x_50=x, y_50=y, v_50=v, wz_50=wz_50, tt_50=tt_50,
        features_50=features,
        corn=dict(CORN_MIKS),
        cuts_s=list(CUTS_S_MIKS),
        sector_corners=[list(s) for s in SECTOR_CORNERS_MIKS],
    )


def compute_features_from_signals(wz_f: np.ndarray, a_h: np.ndarray,
                                   ds: int = 2) -> np.ndarray:
    """Build DTW features matrix [wz/1.2, 0.5·z(|a_h|)] after decimation by ds.

    Используется и admin-op (`build_template.py`), и run-time pipeline (для
    фичей per-lap). Нормировка a_h — z-score шаблона; при feature extraction
    на per-lap window берётся z-score ОТНОСИТЕЛЬНО шаблона, чтобы было
    согласованно (см. laps_compare.compute_dtw_path).
    """
    wz_ds = wz_f[::ds]
    ah_ds = a_h[::ds]
    mu = float(ah_ds.mean())
    sd = max(float(ah_ds.std()), 1e-6)
    return np.c_[wz_ds / 1.2, (ah_ds - mu) / sd * 0.5]


__all__ = [
    "TrackTemplate",
    "load_track_template",
    "compute_features_from_signals",
    "CORN_MIKS",
    "CUTS_S_MIKS",
    "SECTOR_CORNERS_MIKS",
    "PHASE_CUT_MIN",
    "PHASE_CUT_MAX",
    "PHASE_LIFT_UPPER_G",
    "PHASE_ACCEL_LOWER_G",
    "PHASE_LABELS_RU",
]
