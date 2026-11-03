"""Per-corner метрики по v2 spec §7.

Порт reference handoff2_reference/corners.py:lap_metrics + add_frame.

Внешнее API:
- `add_frame(ax, ay, wz, yaw, segment_slice)` — находит θ (ориентацию
  датчика пилота) один раз на сессию. Возвращает dict с a_lat, a_long, hf.
  Используется в full_pipeline, когда у нас уже есть wz_f, ax, ay, yaw
  всей сессии и хороший кусок с кругами как reference для θ.

- `lap_metrics(w, a, b, path, ds=2, corn)` — по DTW-пути (template_50 ↔
  lap_100) для каждого Tn считает:
    * vmin  — минимум |a_lat|·g/|wz| где |wz|>1 (км/ч)
    * alat_pk  — пик |a_lat| в окне, g
    * wz_pk    — пик |wz|, рад/с
    * amin     — min a_long в окне [entry-1.5s, exit]
    * brake_on — секунд от входа до начала торможения (может быть
                 отрицательным — если тормоз начался раньше геом. окна)
    * brake_len — секунд торможения в окне (под per-session cut)
    * lift_len — секунд в фазе "сброс/лёгкий трейл"
    * slip     — секунд где `gauss(|wz|-|a_lat|·g/v, 60 ms) > 0.28` >= 80 ms
    * hf       — средний wz-power в полосе 2-8 Hz в окне
    * dur      — длительность окна (с)

Используется и в session-pipeline (per-lap метрики), и в recommender
(которой нужны и best, и median значения).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
from scipy.signal import butter, filtfilt


G = 9.81
_FS = 100.0


def _butter_lowpass(x: np.ndarray, cutoff_hz: float, fs: float = _FS,
                     order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def _butter_bandpass(x: np.ndarray, lo_hz: float, hi_hz: float, fs: float = _FS,
                      order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo_hz / nyq, hi_hz / nyq], btype="band")
    return filtfilt(b, a, x)


def add_frame(ax: np.ndarray, ay: np.ndarray, wz_f: np.ndarray,
               yaw: np.ndarray, segment: slice, fs: float = _FS,
               step_deg: int = 2) -> dict[str, Any]:
    """Находит ориентацию датчика θ один раз на сессию.

    `segment` — срез (slice) по индексам всей записи, покрывающий чистый
    кусок с кругами (reference берёт `slice(st[1], st[-1])` — все круги
    кроме первого прогревочного). На этом окне считается
    `corr(a_lat, wz)` на сетке 2° (0..360°), выбирается max.

    Reference также НЕ применяет здесь physical-sign check (reference делает
    его вне add_frame через sess2b/corners); в v2 физический знак проверяем
    отдельно в full_pipeline (per spec §5: средний a_long до/после apex).

    Возвращает dict:
      theta: float (rad)
      corr:  float (корреляция на найденном θ, ожидаем > 0.9)
      alat:  np.ndarray (поперечная, g — исходные `ax`,`ay` в g)
      along: np.ndarray (продольная, g)
      hf:    np.ndarray ((wz²) в полосе 2-8 Гц — высокочастотная активность
             руки; "wz-power" per sample)
    """
    axf = _butter_lowpass(ax, 1.5, fs=fs)
    ayf = _butter_lowpass(ay, 1.5, fs=fs)
    best_th, best_c = 0.0, -2.0
    for deg in range(0, 360, step_deg):
        th = np.radians(deg)
        psi = yaw + th
        alat = -axf * np.sin(psi) + ayf * np.cos(psi)
        if alat[segment].std() < 1e-6 or wz_f[segment].std() < 1e-6:
            continue
        c = float(np.corrcoef(alat[segment], wz_f[segment])[0, 1])
        if c > best_c:
            best_c = c
            best_th = th
    psi = yaw + best_th
    alat = -axf * np.sin(psi) + ayf * np.cos(psi)
    along = axf * np.cos(psi) + ayf * np.sin(psi)
    # 2-8 Hz мощность на raw wz (не filtered — иначе высокочастотный хвост
    # обрезан). Ref: brake2.py / corners.py.
    hf = _butter_bandpass(wz_f, 2.0, 8.0, fs=fs) ** 2  # grossly, но ref так
    return {
        "theta": best_th,
        "corr": best_c,
        "alat": alat,
        "along": along,
        "hf": hf,
    }


def _lap_idx_map(path: list[tuple[int, int]] | np.ndarray, a: int,
                  ds: int) -> Any:
    """Возвращает callable template_time_s → absolute lap sample index.

    `path` — DTW путь (template_50 ↔ lap_50), т.е. j — индекс в decimated
    lap feature. Convert в полный sample index: j*ds + a.
    """
    p = np.asarray(path, dtype=np.int64)
    tj = p[:, 1] * ds
    ti = p[:, 0]   # template @50Hz

    def lap_idx(t_tmpl_s: float) -> int:
        return a + int(np.interp(t_tmpl_s * 50.0, ti, tj))
    return lap_idx


def lap_metrics(w: dict[str, np.ndarray], a: int, b: int,
                 path: list[tuple[int, int]] | np.ndarray,
                 corn: dict[str, tuple[float, float]],
                 ds: int = 2,
                 cut_g: float = -0.47,
                 fs: float = _FS) -> dict[str, dict[str, float]]:
    """Для каждого Tn (template-окно `(t0, t1)` в секундах) считаем метрики.

    `w` — dict с полями `wz_f`, `alat`, `along`, `hf` (все в g / rad/s / g).
    `a`, `b` — абсолютные индексы в w: начало/конец круга @100Hz.
    `path` — DTW путь, где i — template @50Hz, j — lap @50Hz.
    `cut_g` — session-wide brake cut (per-session Otsu).

    Returns `{corner_name: metrics_dict}`.
    """
    lap_idx = _lap_idx_map(path, a, ds=ds)
    wz = w["wz_f"]
    alat = w["alat"]
    along = gf(w["along"], 8)  # σ=80 мс @100Hz
    hf = w["hf"]

    # v for slip detection (км/ч pipeline будет использовать 3.6; тут в m/s).
    v_lap = np.abs(alat) * G / np.maximum(np.abs(wz), 0.3)

    out: dict[str, dict[str, float]] = {}
    pre_samples = int(1.5 * fs)
    min_slip_samples = int(0.08 * fs)   # 80 мс

    for name, (t0, t1) in corn.items():
        i0 = lap_idx(t0)
        i1 = lap_idx(t1)
        if i1 <= i0 or i0 < a or i1 > b:
            continue
        i0b = max(a, i0 - pre_samples)   # 1.5 s до геом. входа
        seg = slice(i0, i1)
        seg_brake = slice(i0b, i1)
        wz_seg = wz[seg]
        alat_seg = alat[seg]
        along_brake = along[seg_brake]

        # peaks
        wz_pk = float(np.abs(wz_seg).max()) if wz_seg.size else float("nan")
        alat_pk = float(np.abs(alat_seg).max()) if alat_seg.size else float("nan")

        # vmin: |a_lat|·g/|wz| where |wz|>1
        v_seg = v_lap[seg]
        mask = np.abs(wz_seg) > 1.0
        if mask.any():
            vmin_ms = float(v_seg[mask].min())
            vmin_kmh = vmin_ms * 3.6
        else:
            vmin_kmh = float("nan")

        # Brake metrics based on cut_g
        amin = float(along_brake.min()) if along_brake.size else float("nan")
        br_mask = along_brake < cut_g
        brake_len = float(br_mask.sum() / fs)
        # brake_on: когда он начал тормозить, относительно геом. входа i0
        if br_mask.any():
            first = int(np.argmax(br_mask))   # первый True в seg_brake
            brake_on = (first + i0b - i0) / fs
        else:
            brake_on = float("nan")

        # lift_len: сколько сэмплов между lift_upper и cut_g (сброс / лёгкий трейл)
        # Пороги фаз согласно §6: ТОРМОЗ < cut, иначе если a_long < -0.12 → lift.
        from .track_template import PHASE_LIFT_UPPER_G
        lift_mask = (along_brake >= cut_g) & (along_brake < PHASE_LIFT_UPPER_G)
        lift_len = float(lift_mask.sum() / fs)

        # slip: excess yaw rate vs expected from v and a_lat
        # expected wz = a_lat·g/v  (|wz| ≈ |a_lat|·g/v в scrub-free мод-е)
        vv = gf(v_lap, int(0.3 * fs))  # σ=0.3s (ref: gf(v,30) @ fs=100)
        exc = np.abs(wz_seg) - np.abs(alat_seg) * G / np.maximum(vv[seg], 3.0)
        exc_sm = gf(exc, int(0.06 * fs))  # σ=60 мс
        slip_mask = exc_sm > 0.28
        # фрагменты длины >= min_slip_samples
        slip = 0.0
        i = 0
        n = len(slip_mask)
        while i < n:
            if not slip_mask[i]:
                i += 1
                continue
            j = i
            while j < n and slip_mask[j]:
                j += 1
            if j - i >= min_slip_samples:
                slip += (j - i) / fs
            i = j

        # HF band mean
        hf_val = float(hf[seg].mean()) if seg.stop > seg.start else float("nan")

        out[name] = {
            "vmin_kmh": vmin_kmh,
            "alat_pk": alat_pk,
            "wz_pk": wz_pk,
            "amin": amin,
            "brake_len": brake_len,
            "brake_on": brake_on,
            "lift_len": lift_len,
            "slip": slip,
            "hf": hf_val,
            "dur": (i1 - i0) / fs,
        }
    return out


def otsu_threshold(values: np.ndarray) -> float:
    """1D Otsu — классический поиск порога как в OpenCV. Для нашего случая
    (a_long minima per lap × corner), N обычно = laps × corners (~400).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    # Строим гистограмму
    nb = 256
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return float((lo + hi) / 2)
    hist, edges = np.histogram(v, bins=nb, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return float((lo + hi) / 2)
    probs = hist / total
    centers = (edges[:-1] + edges[1:]) / 2
    best_var = -1.0
    best_t = centers[0]
    cum = np.cumsum(probs)
    cum_mu = np.cumsum(probs * centers)
    mu_t = cum_mu[-1]
    for i in range(1, nb):
        w0 = cum[i - 1]
        w1 = 1 - w0
        if w0 < 1e-9 or w1 < 1e-9:
            continue
        mu0 = cum_mu[i - 1] / w0
        mu1 = (mu_t - cum_mu[i - 1]) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = centers[i]
    return float(best_t)


__all__ = [
    "add_frame",
    "lap_metrics",
    "otsu_threshold",
    "G",
]
