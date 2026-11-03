"""Полный IMU → анализ сессии pipeline (v2 spec).

Отличия от v1:
- Шаблон трассы загружается из `tracks/<name>/` (tmpl.npy + track.npz +
  features.npy), НЕ строится из сессии. Rebuild — отдельный admin op.
- Три gate'а: session (RMS(wz_f) и |∫wz|), direction (знак ∑wz·dt),
  timing match (±0.3 s). Lap gate (NCC + DTW residual) per-lap.
- Lap boundaries: NCC с tmpl.npy → find_peaks → refine ±2 s корреляцией
  с окном "последние 5 с + первые 5 с" шаблона.
- Соответствие IMU↔RaceMann по минимальному median error сдвига.
- DTW: `dtw_closed(T, F_lap[::2], band=100)` на двух фичах — reference.
- Сектор-границы по CUTS_S через DTW path + локальная refine ±0.6 s.
- Ориентация θ один раз на сессию (reference: add_frame).
- Фаз-порог: per-session Otsu на amin[lap,corner], clamp [-0.55, -0.40].
- Per-corner метрики §7 + rule-based recommender §10.

Hacks которые удалены:
- Хардкод `-0.25 g` для торможения (теперь session-wide Otsu cut).
- IMU-fallback для отсутствующего timing (теперь lap gate).
- Loose assertion `< 1.0 s`.

Выход `analyze_session(watch_imu_path, timing_path, track_name)` — JSON
совместимый dict для web-рендера.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation as R

from . import loader
from .corner_metrics import add_frame, lap_metrics, otsu_threshold, G
from .recommender import (
    generate_recommendations,
    compute_potential_s,
    compute_trend_s,
)
from .track_template import (
    TrackTemplate,
    load_track_template,
    PHASE_CUT_MIN,
    PHASE_CUT_MAX,
    PHASE_LIFT_UPPER_G,
    PHASE_ACCEL_LOWER_G,
    PHASE_LABELS_RU,
)


_FS = 100.0


# ------------------------------------------------------------------
#  §0. Загрузка IMU и приведение в world frame
# ------------------------------------------------------------------

def _load_and_rotate(watch_imu_path: Path, conj: bool = False) -> dict[str, np.ndarray]:
    """Reference load2.py: imu_v1 bin → world-frame dict of arrays.

    Поля: t (с), ax ay az (g), wx wy wz (world rad/s), yaw (unwrap).
    `conj=True` поворачивает кватернионы в обратную сторону (используется
    когда direction gate выявил отрицательный ∑wz·dt).

    Pure numpy — pandas сознательно не используется чтобы не тянуть новую
    зависимость в контейнер (Dockerfile ставит только numpy+scipy).
    """
    arr = loader.load_imu(watch_imu_path)
    if len(arr) == 0:
        raise ValueError(f"empty IMU file: {watch_imu_path}")
    t_ns = arr["ts_ns"].astype(np.int64)
    # ts_ns: iOS отправляет ns, но исторически в некоторых сборках мс. Reference
    # heuristic: если первый ts < 1e13, значит это мс.
    t0 = t_ns[0]
    if t0 < 1e13:
        t = t_ns.astype(float) / 1e3
    else:
        t = t_ns.astype(float) / 1e9

    quat = np.c_[arr["qx"], arr["qy"], arr["qz"], arr["qw"]].astype(float)
    gyro = np.c_[arr["gx"], arr["gy"], arr["gz"]].astype(float)
    accel = np.c_[arr["ax"], arr["ay"], arr["az"]].astype(float)

    rot = R.from_quat(quat)
    if conj:
        rot = rot.inv()

    w = rot.apply(gyro)
    a = rot.apply(accel)
    yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])

    return {
        "t": t,
        "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2],
        "wx": w[:, 0], "wy": w[:, 1], "wz": w[:, 2],
        "yaw": yaw,
    }


def _resample_uniform(d: dict[str, np.ndarray], fs: float = _FS) -> dict[str, np.ndarray]:
    """Resample на равномерную сетку fs Гц (линейная интерполяция)."""
    t = d["t"]
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    out: dict[str, np.ndarray] = {"t": tu}
    for col in ("wz", "ax", "ay", "az", "yaw", "wx", "wy"):
        out[col] = np.interp(tu, t, d[col])
    return out


def _butter_lowpass(x: np.ndarray, cutoff: float, fs: float = _FS,
                     order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


# ------------------------------------------------------------------
#  §2. Session gates
# ------------------------------------------------------------------

def _session_gate(wz_f: np.ndarray, fs: float = _FS) -> tuple[bool, dict]:
    rms = float(np.sqrt(np.mean(wz_f ** 2)))
    turns = float(abs(wz_f.sum() / fs) / (2 * np.pi))
    ok = rms > 0.6 and turns >= 1
    return ok, {"rms_wz": rms, "turns": turns}


def _direction_sign(wz_f: np.ndarray, fs: float = _FS) -> int:
    """Return +1 if ∑wz·dt > 0 else -1. На MIKS ожидаем +1 (ccw)."""
    return 1 if wz_f.sum() / fs > 0 else -1


# ------------------------------------------------------------------
#  §3. Lap boundaries by NCC с tmpl.npy
# ------------------------------------------------------------------

def _ncc(sig: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation sig × tmpl (reference sess2.py)."""
    n = len(tmpl)
    if n < 10 or len(sig) <= n:
        return np.zeros(max(0, len(sig) - n))
    tm = (tmpl - tmpl.mean()) / max(tmpl.std(), 1e-9)
    cs = np.cumsum(np.insert(sig, 0, 0))
    cs2 = np.cumsum(np.insert(sig ** 2, 0, 0))
    L = len(sig) - n
    num = np.correlate(sig, tm, "valid")[:L]
    mu = (cs[n:n + L] - cs[:L]) / n
    var = (cs2[n:n + L] - cs2[:L]) / n - mu ** 2
    denom = n * np.sqrt(np.maximum(var, 1e-9))
    return num / denom


def _detect_lap_starts(sig: np.ndarray, tmpl: np.ndarray,
                        fs: float = _FS) -> tuple[np.ndarray, np.ndarray]:
    """find_peaks NCC с templ, then refine ±2 s корреляцией с tail+head.

    Returns (indices_in_sig, ncc_scores_at_peaks).
    """
    c = _ncc(sig, tmpl)
    if len(c) == 0:
        return np.array([], dtype=np.int64), np.array([])
    pk, _ = find_peaks(c, distance=int(28 * fs), height=0.5)
    if len(pk) == 0:
        return np.array([], dtype=np.int64), np.array([])
    pre = post = int(5 * fs)
    win = np.concatenate([tmpl[-pre:], tmpl[:post]])
    refined = []
    for p in pk:
        best = (-np.inf, p)
        for s in range(-int(2 * fs), int(2 * fs) + 1):
            a = p + s - pre
            b = p + s + post
            if a < 0 or b > len(sig):
                continue
            seg = sig[a:b]
            if seg.std() < 1e-9:
                continue
            r = float(np.corrcoef(seg, win)[0, 1])
            if r > best[0]:
                best = (r, p + s)
        refined.append(best[1])
    refined_arr = np.array(refined, dtype=np.int64)
    scores_at_pk = c[pk]
    return refined_arr, scores_at_pk


# ------------------------------------------------------------------
#  §3b. Match IMU ↔ RaceMann sequences
# ------------------------------------------------------------------

def _match_imu_to_racemann(dur_imu: np.ndarray, dur_rm: np.ndarray
                             ) -> tuple[int, float, int]:
    """Reference sess2.py: find offset k such that dur[i] ≈ rm[i+k].

    `dur_imu` — IMU lap durations (сек, из diff(start_indices)/fs).
    `dur_rm`  — RaceMann lastLapMs/1000 (сек).

    Returns (offset_k, median_error_s, n_matched).
    """
    best = (0, float("inf"), 0)
    if len(dur_imu) == 0 or len(dur_rm) == 0:
        return best
    for k in range(-5, 6):
        if k >= 0:
            n = min(len(dur_imu), len(dur_rm) - k)
        else:
            n = min(len(dur_imu) + k, len(dur_rm))
        if n <= 0:
            continue
        a_start = max(0, -k)
        b_start = max(0, k)
        a = dur_imu[a_start:a_start + n]
        b = dur_rm[b_start:b_start + n]
        e = np.abs(a - b)
        med = float(np.median(e))
        if med < best[1]:
            best = (k, med, n)
    return best


# ------------------------------------------------------------------
#  §4. DTW closed path (reference dtw.py:dtw_closed)
# ------------------------------------------------------------------

def _dtw_closed(A: np.ndarray, B: np.ndarray, band: int = 100) -> np.ndarray:
    """DTW с Sakoe-Chiba corridor; оба конца закреплены. Returns path (K, 2).

    Port of reference dtw.py:dtw_closed, with cost = sum((B-A)^2) elementwise.
    """
    N, M = len(A), len(B)
    INF = 1e18
    D = np.full((N + 1, M + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        c = int(round(i * M / N))
        lo = max(1, c - band)
        hi = min(M, c + band)
        diff = B[lo - 1:hi] - A[i - 1]
        cost = np.sum(diff * diff, axis=1)
        prev = D[i - 1]
        row = D[i]
        cand = np.minimum(prev[lo - 1:hi], prev[lo:hi + 1])
        for j in range(lo, hi + 1):
            v = cand[j - lo]
            left = row[j - 1]
            if left < v:
                v = left
            row[j] = cost[j - lo] + v
    i, j = N, M
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        opts = (
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1),
        )
        _, i, j = min(opts, key=lambda x: x[0])
        path.append((i - 1, j - 1))
    return np.array(path[::-1], dtype=np.int64)


def _dtw_residual(path: np.ndarray, F_lap_50: np.ndarray,
                   wzT_50: np.ndarray) -> float:
    """RMS residual между wz-колонкой пути (lap × 1.2) и template wz."""
    wz_w = np.interp(path[:, 0], np.arange(len(wzT_50)), wzT_50)
    wz_lap = F_lap_50[path[:, 1], 0] * 1.2  # feature → wz
    return float(np.sqrt(np.mean((wz_lap - wz_w) ** 2)))


# ------------------------------------------------------------------
#  §4b. Sector cuts via DTW path + local correlation refine
# ------------------------------------------------------------------

def _sector_cuts_for_lap(path: np.ndarray, a: int, b: int,
                          cuts_s: list[float],
                          wz_f: np.ndarray, wz_t_100: np.ndarray,
                          ds: int = 2, fs: float = _FS) -> list[int]:
    """Для одного круга: cuts в абсолютных индексах sig.

    Reference sess2b.py: DTW путь → ci = cut*50 (template @50Hz) →
    j = путь[argmax(path[:,0]>=ci), 1]*ds + a. Затем refine ±0.6 s
    локальной корреляцией wz_f ±W с wz_template ±W.
    """
    cuts = [a]
    W = 200  # ±2 s @100Hz
    search = int(0.6 * fs)
    tmpl_len = len(wz_t_100)
    for c in cuts_s:
        ci = int(round(c * 50))
        idxs = np.where(path[:, 0] >= ci)[0]
        if len(idxs) == 0:
            # cut за концом path: аппроксимируем концом окна
            cuts.append(b)
            continue
        j = int(path[idxs[0], 1]) * ds + a
        tmpl_center = ci * 2  # в tmpl @100Hz: 50Hz*2
        best = (-np.inf, j)
        for s in range(-search, search + 1):
            e = j + s
            if e - W < 0 or e + W > len(wz_f):
                continue
            a_t = tmpl_center - W
            b_t = tmpl_center + W
            if a_t < 0 or b_t > tmpl_len:
                continue
            seg = wz_f[e - W:e + W]
            tseg = wz_t_100[a_t:b_t]
            if seg.std() < 1e-9 or tseg.std() < 1e-9:
                continue
            r = float(np.corrcoef(seg, tseg)[0, 1])
            if r > best[0]:
                best = (r, e)
        cuts.append(best[1])
    cuts.append(b)
    # нормализация: сектор-дуры неотрицательны
    for k in range(1, len(cuts)):
        if cuts[k] < cuts[k - 1]:
            cuts[k] = cuts[k - 1]
    return cuts


# ------------------------------------------------------------------
#  §5b. Physical sign check для θ (знак a_long)
# ------------------------------------------------------------------

def _flip_theta_if_needed(wz_f: np.ndarray, along: np.ndarray, lap_bounds: list[int],
                           fs: float = _FS) -> bool:
    """Проверка знака продольного ускорения вокруг апексов.

    §5 spec: mean a_long [0.8 s до apex] < 0 < mean a_long [0.8 s после apex].
    Если нет (большинство apex'ов нарушают), θ += 180° (flip along+alat).

    Returns True if flip needed.
    """
    wz_abs = np.abs(wz_f)
    peaks = []
    for la, lb in zip(lap_bounds[:-1], lap_bounds[1:]):
        if lb - la < int(5 * fs):
            continue
        # Локальные пики |wz| в круге
        for i in range(la + int(1 * fs), lb - int(1 * fs), int(0.5 * fs)):
            if wz_abs[i] >= 1.0 and wz_abs[i] >= wz_abs[max(la, i - 50):min(lb, i + 50)].max() * 0.98:
                peaks.append(i)
    if not peaks:
        return False
    w = int(0.8 * fs)
    delta_sum = 0.0
    for p in peaks:
        if p - w < 0 or p + w >= len(along):
            continue
        delta_sum += along[p:p + w].mean() - along[p - w:p].mean()
    # delta_sum > 0: до apex тормоз (отриц.), после — газ (полож.) → OK
    return delta_sum < 0


# ------------------------------------------------------------------
#  §3c. Timing load + dedupe
# ------------------------------------------------------------------

def _parse_timing(timing_json: dict, kart: str | None = None
                   ) -> tuple[list[dict], str | None]:
    """Возвращает (list of lap records, kart_number).

    Dedupe по wallTime (per spec §2). Drop records lapMs > 60000 ms.
    """
    if not timing_json:
        return [], None
    if kart is None:
        kart = timing_json.get("kartNumber") or None

    # Пробуем allPasses (ground truth), если нет — laps[].
    passes = timing_json.get("allPasses") or []
    laps: list[dict] = []
    if passes and kart:
        for p in passes:
            if str(p.get("kartNumber")) != str(kart):
                continue
            lap_ms = p.get("lastLapMs")
            if not lap_ms or lap_ms > 60_000:
                continue
            laps.append({
                "lapNumber": p.get("lapNumber"),
                "lapMs": int(lap_ms),
                "wallTime": p.get("wallTime"),
                "position": p.get("position"),
            })
    else:
        raw = timing_json.get("laps") or []
        if isinstance(raw, str):
            raw = json.loads(raw)
        for l in raw:
            if l.get("lapMs", 0) > 60_000:
                continue
            laps.append({
                "lapNumber": l.get("lapNumber"),
                "lapMs": int(l["lapMs"]),
                "wallTime": l.get("wallTime"),
                "position": l.get("position"),
            })

    # Dedupe по wallTime (per spec §2)
    seen = set()
    unique = []
    for l in laps:
        key = l["wallTime"] or f"{l.get('lapNumber')}:{l['lapMs']}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(l)
    unique.sort(key=lambda l: (l.get("lapNumber") or 0, l.get("wallTime") or ""))
    return unique, kart


# ------------------------------------------------------------------
#  §8. Hybrid dot mapping (lap→template progress)
# ------------------------------------------------------------------

def _hybrid_mapping(path: np.ndarray, wz_lap: np.ndarray, v_lap: np.ndarray,
                     N_lap: int, ds: int = 2, fs: float = _FS,
                     min_gap_s: float = 0.5, min_wz: float = 0.8
                     ) -> np.ndarray:
    """Reference maps2.py:hybrid — зlgdы DTW на границах поворотов (zero
    crossings wz с реальной дугой рядом), внутри — по пути v·dt.

    Returns m[j] = template index for each lap sample j (0..N_lap-1).
    `path` в decimated координатах (i_tmpl_50, j_lap_50).
    """
    p = np.asarray(path)
    # Базовая проекция j → i: DTW путь в 50Hz, переводим в 100Hz через ds.
    m0 = np.interp(np.arange(N_lap), p[:, 1] * ds, p[:, 0]) * ds
    half = int(fs / 2)
    zc = np.where(np.diff(np.sign(wz_lap)) != 0)[0]
    zc = [int(z) for z in zc
          if np.abs(wz_lap[max(0, z - half):z + half]).max() > min_wz]
    anchors = [0]
    for z in zc:
        if z - anchors[-1] > min_gap_s * fs and z < N_lap - 1 - min_gap_s * fs:
            anchors.append(z)
    anchors.append(N_lap - 1)
    s = np.cumsum(v_lap) / fs
    m = np.empty(N_lap)
    for a_, b_ in zip(anchors[:-1], anchors[1:]):
        span = max(s[b_] - s[a_], 1e-6)
        frac = (s[a_:b_ + 1] - s[a_]) / span
        m[a_:b_ + 1] = m0[a_] + frac * (m0[b_] - m0[a_])
    return m


# ------------------------------------------------------------------
#  §6. Phases (classify a_long per-sample with per-session cut)
# ------------------------------------------------------------------

def _classify_phases(along_g_sm: np.ndarray, cut_g: float,
                      min_segment_s: float = 0.15, fs: float = _FS) -> list[str]:
    """Фазовая классификация. `cut_g` в g, отрицательное; `along_g_sm` в g.

    Поглощение сегментов <min_segment_s.
    """
    ph: list[str] = []
    for a in along_g_sm:
        if a < cut_g:
            ph.append("brake")
        elif a < PHASE_LIFT_UPPER_G:
            ph.append("lift")
        elif a > PHASE_ACCEL_LOWER_G:
            ph.append("accel")
        else:
            ph.append("cruise")
    # Абсорбция: короткие сегменты поглощаются соседом слева
    min_len = int(min_segment_s * fs)
    i = 0
    n = len(ph)
    while i < n:
        j = i
        while j < n and ph[j] == ph[i]:
            j += 1
        if (j - i) < min_len and i > 0 and j < n:
            prev = ph[i - 1]
            for k in range(i, j):
                ph[k] = prev
        i = j
    return ph


# ------------------------------------------------------------------
#  §9. Grip ratio (reference comparison)
# ------------------------------------------------------------------

def _grip_ratio(metrics_by_lap: dict[int, dict[str, dict[str, float]]],
                 valid_laps: list[int],
                 reference_metrics: dict[str, dict[str, float]] | None
                 ) -> float | None:
    """Соотношение median(alat_pk across corners) / тот же показатель
    reference. ≈ 1 = тот же hold; <1 — слабее держак.
    """
    if not reference_metrics:
        return None
    our = []
    for lap in valid_laps:
        if lap not in metrics_by_lap:
            continue
        for c, m in metrics_by_lap[lap].items():
            if np.isfinite(m.get("alat_pk", np.nan)):
                our.append(m["alat_pk"])
    if not our:
        return None
    ref = [m["alat_pk"] for m in reference_metrics.values()
           if np.isfinite(m.get("alat_pk", np.nan))]
    if not ref:
        return None
    return float(np.median(our)) / max(float(np.median(ref)), 1e-9)


# ------------------------------------------------------------------
#  Главная функция
# ------------------------------------------------------------------

VISUAL_TOKENS: dict[str, Any] = {
    "phases": {
        "brake":  {"color": "#E24B4A", "width": 9, "label": PHASE_LABELS_RU["brake"]},
        "lift":   {"color": "#EF9F27", "width": 7, "label": PHASE_LABELS_RU["lift"]},
        "accel":  {"color": "#1D9E75", "width": 6, "label": PHASE_LABELS_RU["accel"]},
        "cruise": {"color": "#B4B2A9", "width": 4, "label": PHASE_LABELS_RU["cruise"]},
    },
    "pavement": {"color": "#D3D1C7", "width": 22, "opacity": 0.55},
    "sectors":  ["#888780", "#7F77DD", "#D85A30", "#EF9F27", "#1D9E75",
                 "#22D3EE", "#A78BFA"],
    "slip":     {"color": "#534AB7", "base_radius": 6, "scale": 22},
}


def analyze_session(watch_imu_path: str | Path,
                     timing_path: str | Path | None = None,
                     track_name: str = "miks",
                     reference_metrics: dict[str, dict[str, float]] | None = None,
                     ) -> dict[str, Any]:
    """Полный pipeline. See module docstring.

    `reference_metrics` — опциональный словарь из другой сессии (другой карт/
    день) для grip-ratio и правил сравнения в recommender §9/§10.
    """
    watch_imu_path = Path(watch_imu_path)
    notes: list[str] = []

    # 1. load + resample
    tpl: TrackTemplate = load_track_template(track_name)
    notes.append(
        f"track='{track_name}', tmpl_len={len(tpl.tmpl_wz_100)}"
        f" ({tpl.lap_duration_s:.2f}s), N_50={len(tpl.wz_50)}"
    )

    df = _load_and_rotate(watch_imu_path, conj=False)
    w = _resample_uniform(df, fs=_FS)
    fs = _FS
    # Фильтры §0: wz_f, ah_f
    w["wz_f"] = _butter_lowpass(w["wz"], 1.5, fs=fs)
    ah = np.hypot(w["ax"], w["ay"])
    w["ah_f"] = _butter_lowpass(ah, 1.5, fs=fs)

    notes.append(f"IMU: {len(w['t'])} samples @{fs:.0f}Hz, duration {len(w['t'])/fs:.1f}s")

    # 2. Session gate
    sg_ok, sg_info = _session_gate(w["wz_f"], fs=fs)
    notes.append(
        f"session gate: RMS(wz_f)={sg_info['rms_wz']:.2f} "
        f"turns={sg_info['turns']:.2f} → {'OK' if sg_ok else 'FAIL'}"
    )
    if not sg_ok:
        return {
            "status": "no_driving",
            "notes": notes,
            "gates": {"session": sg_info},
            "visual": VISUAL_TOKENS,
        }

    # 2b. Direction sign check
    dir_sign = _direction_sign(w["wz_f"], fs=fs)
    if dir_sign < 0:
        notes.append("direction sign negative → re-rotating with conj quaternion")
        df = _load_and_rotate(watch_imu_path, conj=True)
        w = _resample_uniform(df, fs=_FS)
        w["wz_f"] = _butter_lowpass(w["wz"], 1.5, fs=fs)
        ah = np.hypot(w["ax"], w["ay"])
        w["ah_f"] = _butter_lowpass(ah, 1.5, fs=fs)
        dir_sign = _direction_sign(w["wz_f"], fs=fs)
    notes.append(f"direction sign after fix: {dir_sign:+d}")

    # 3. Lap starts via NCC(tmpl.npy) + find_peaks + refine
    sig = w["wz_f"]
    tmpl = tpl.tmpl_wz_100
    starts, ncc_scores = _detect_lap_starts(sig, tmpl, fs=fs)
    notes.append(
        f"IMU laps: {len(starts)}, "
        f"NCC median={float(np.median(ncc_scores)):.2f}"
        if len(starts) else "IMU laps: 0"
    )
    if len(starts) < 2:
        return {
            "status": "no_laps",
            "notes": notes,
            "gates": {"session": sg_info, "direction": dir_sign},
            "visual": VISUAL_TOKENS,
        }

    # 4. Timing load + match
    timing_json = {}
    if timing_path and Path(timing_path).exists():
        try:
            timing_json = loader.load_timing(timing_path)
        except Exception as e:  # noqa: BLE001
            notes.append(f"timing load failed: {e}")
    rm_laps, kart = _parse_timing(timing_json)
    notes.append(f"RaceMann: {len(rm_laps)} laps, kart={kart}")

    # IMU lap durations (seconds)
    dur_imu = np.diff(starts) / fs
    dur_rm = np.array([l["lapMs"] / 1000.0 for l in rm_laps])

    timing_match_ok = False
    offset_k = 0
    rm_errors_med = float("inf")
    if len(dur_rm) >= 1 and len(dur_imu) >= 1:
        offset_k, rm_errors_med, n_matched = _match_imu_to_racemann(dur_imu, dur_rm)
        count_ok = abs(len(dur_imu) - len(dur_rm)) <= 1
        dur_ok = rm_errors_med <= 0.3
        timing_match_ok = count_ok and dur_ok
        notes.append(
            f"timing match: offset_k={offset_k}, med_err={rm_errors_med:.2f}s "
            f"({n_matched} laps) → {'OK' if timing_match_ok else 'MISMATCH'}"
        )
    else:
        notes.append("timing match: no RaceMann laps — IMU-only mode")

    # Присвоение lap_number: если match OK — используем RaceMann; иначе 1..N
    # индекс. Длина окна [start, start + lapMs/1000·fs] — если match OK,
    # иначе [start, next_start].
    lap_bounds: list[int] = []  # starts in sig
    lap_ends: list[int] = []
    lap_numbers: list[int] = []
    lap_times_s: list[float] = []
    lap_wall_time: list[str | None] = []
    lap_positions: list[int | None] = []
    lap_sources: list[str] = []
    for i, s in enumerate(starts):
        lap_bounds.append(int(s))
        if timing_match_ok:
            # Mapping i → rm index
            rm_idx_ = (i - max(0, -offset_k)) + max(0, offset_k)
            if 0 <= rm_idx_ < len(rm_laps):
                lap = rm_laps[rm_idx_]
                length_samples = int(round(lap["lapMs"] / 1000.0 * fs))
                end = s + length_samples
                if end > len(sig):
                    end = int(starts[i + 1]) if i + 1 < len(starts) else len(sig)
                lap_ends.append(end)
                lap_numbers.append(lap.get("lapNumber") or (i + 1))
                lap_times_s.append(lap["lapMs"] / 1000.0)
                lap_wall_time.append(lap.get("wallTime"))
                lap_positions.append(lap.get("position"))
                lap_sources.append("timing")
                continue
        # Fallback: IMU boundary-to-boundary
        if i + 1 < len(starts):
            end = int(starts[i + 1])
        else:
            end = len(sig)
        lap_ends.append(end)
        lap_numbers.append(i + 1)
        lap_times_s.append((end - s) / fs)
        lap_wall_time.append(None)
        lap_positions.append(None)
        lap_sources.append("imu")

    # Drop last lap if ends out of range (shouldn't happen but safe).
    while lap_ends and lap_ends[-1] > len(sig):
        lap_bounds.pop(); lap_ends.pop(); lap_numbers.pop()
        lap_times_s.pop(); lap_wall_time.pop(); lap_positions.pop(); lap_sources.pop()

    N_laps = len(lap_bounds)
    notes.append(f"laps assembled: {N_laps}")

    # 5. Orientation θ (one θ for whole session) + a_lat/a_long
    # Выбираем segment: от первого IMU lap до последнего.
    seg_slice = slice(lap_bounds[0], lap_ends[-1])
    frame = add_frame(
        ax=w["ax"], ay=w["ay"],
        wz_f=w["wz_f"], yaw=w["yaw"],
        segment=seg_slice, fs=fs, step_deg=2,
    )
    notes.append(
        f"θ = {np.degrees(frame['theta']):.0f}°, "
        f"corr(a_lat, wz_f) = {frame['corr']:.2f}"
    )
    along_sm = gaussian_filter1d(frame["along"], int(0.08 * fs))  # σ=80 ms
    # Physical sign check (§5): если нарушено, θ += 180°
    if _flip_theta_if_needed(w["wz_f"], along_sm, lap_bounds + [lap_ends[-1]],
                              fs=fs):
        notes.append("sign check: a_long inverted → θ += 180°")
        frame["theta"] += np.pi
        # Пересчёт alat/along одномерно (они линейны по cos/sin ψ+θ, flip sign)
        frame["alat"] = -frame["alat"]
        frame["along"] = -frame["along"]
        along_sm = -along_sm
    w["alat"] = frame["alat"]
    w["along"] = frame["along"]
    w["hf"] = frame["hf"]
    w["along_sm"] = along_sm

    # 6. DTW per lap, lap gate, sector cuts
    # Build template features (прекомпилирован в tracks/.../features.npy).
    T = tpl.features_50
    wzT_50 = tpl.wz_50
    wzT_100 = tpl.tmpl_wz_100
    ds = 2

    paths: dict[int, np.ndarray] = {}       # by lap_number
    residuals: dict[int, float] = {}
    ncc_by_lap: dict[int, float] = {}
    sector_cuts: dict[int, list[int]] = {}
    sector_times_by_lap: dict[int, dict[str, float]] = {}
    valid_lap_mask: list[bool] = []

    SEC_NAMES = [f"S{k+1}" for k in range(len(tpl.cuts_s) + 1)]  # 7 секторов

    for i, (a, b) in enumerate(zip(lap_bounds, lap_ends)):
        if b - a < int(5 * fs):
            valid_lap_mask.append(False)
            continue
        if b > len(sig):
            valid_lap_mask.append(False)
            continue
        # Build per-lap features @50Hz (ah normalized by template)
        wz_lap = w["wz_f"][a:b]
        ah_lap = w["ah_f"][a:b]
        # Normalize ah relative to template-features scale: ref dtw.py:features
        # использует (ah - ah.mean()) / ah.std() * 0.5 на том окне. Для совм-ти
        # используем тот же стандарт — ah per-lap, не per-template.
        mu = float(ah_lap.mean())
        sd = max(float(ah_lap.std()), 1e-6)
        F_lap = np.c_[wz_lap / 1.2, (ah_lap - mu) / sd * 0.5]
        F_lap_50 = F_lap[::ds]
        try:
            path = _dtw_closed(T, F_lap_50, band=100)
        except Exception:  # noqa: BLE001
            valid_lap_mask.append(False)
            continue
        resid = _dtw_residual(path, F_lap_50, wzT_50)
        lap_num = lap_numbers[i]
        paths[lap_num] = path
        residuals[lap_num] = resid
        ncc_v = float(ncc_scores[i]) if i < len(ncc_scores) else 0.0
        ncc_by_lap[lap_num] = ncc_v
        lap_valid = (ncc_v >= 0.6) and (resid <= 0.5)
        valid_lap_mask.append(lap_valid)

        if lap_valid:
            cuts = _sector_cuts_for_lap(path, a, b, tpl.cuts_s,
                                        w["wz_f"], wzT_100,
                                        ds=ds, fs=fs)
            sector_cuts[lap_num] = cuts
            secs = {SEC_NAMES[k]: (cuts[k + 1] - cuts[k]) / fs
                    for k in range(len(SEC_NAMES))}
            sector_times_by_lap[lap_num] = secs

    n_valid = sum(valid_lap_mask)
    n_invalid = N_laps - n_valid
    notes.append(
        f"lap gate: {n_valid} valid, {n_invalid} invalid "
        f"(NCC<0.6 or DTW_resid>0.5)"
    )

    # 7. Per-corner metrics с per-session Otsu cut
    #    Шаг A: amin[lap, corner] для всех ВАЛИДНЫХ кругов, фиксированный
    #    tentative cut = -0.47 g (reference) — просто чтобы собрать amin'ы.
    #    Шаг B: Otsu по amin → session cut.
    #    Шаг C: пересчёт всех метрик с session cut.

    valid_laps = [lap_numbers[i] for i, v in enumerate(valid_lap_mask) if v]
    frame_w = {
        "wz_f": w["wz_f"], "alat": w["alat"],
        "along": w["along"], "hf": w["hf"],
    }

    # Шаг A: tentative pass — амины
    amins = []
    for i, (a, b) in enumerate(zip(lap_bounds, lap_ends)):
        if not valid_lap_mask[i]:
            continue
        lap_num = lap_numbers[i]
        mt = lap_metrics(frame_w, a, b, paths[lap_num], tpl.corn,
                          ds=ds, cut_g=-0.47, fs=fs)
        for name, cm in mt.items():
            if np.isfinite(cm["amin"]):
                amins.append(cm["amin"])
    # Шаг B: Otsu + clamp
    if len(amins) >= 10:
        raw_cut = otsu_threshold(np.array(amins))
        phase_cut = float(np.clip(raw_cut, PHASE_CUT_MIN, PHASE_CUT_MAX))
    else:
        raw_cut = -0.47
        phase_cut = -0.47
    notes.append(
        f"phase cut: Otsu raw={raw_cut:.3f}, clamped={phase_cut:.3f} "
        f"(from {len(amins)} amins across {len(valid_laps)} valid laps)"
    )

    # Шаг C: все метрики с session cut
    metrics_by_lap: dict[int, dict[str, dict[str, float]]] = {}
    for i, (a, b) in enumerate(zip(lap_bounds, lap_ends)):
        if not valid_lap_mask[i]:
            continue
        lap_num = lap_numbers[i]
        metrics_by_lap[lap_num] = lap_metrics(
            frame_w, a, b, paths[lap_num], tpl.corn,
            ds=ds, cut_g=phase_cut, fs=fs,
        )

    # 8. Best lap selection (среди валидных по чистому времени)
    valid_times = [(lap_numbers[i], lap_times_s[i])
                   for i, v in enumerate(valid_lap_mask) if v]
    if valid_times:
        best_lap_num = min(valid_times, key=lambda x: x[1])[0]
    else:
        # Fallback: best among ALL
        best_lap_num = min(zip(lap_numbers, lap_times_s), key=lambda x: x[1])[0]
    best_idx = lap_numbers.index(best_lap_num)
    best_time_s = lap_times_s[best_idx]
    notes.append(f"best lap: #{best_lap_num} @ {best_time_s:.3f}s")

    # 9. Delta map (best-vs-session median across valid laps on template indices)
    delta_map: dict[str, float] = {}
    if best_lap_num in metrics_by_lap:
        # Для каждого Tn: Δ = median(session - best) sector-time в окне ±
        # окна поворота. Простая эмиксия: разница медианы sector_time (среди
        # валидных) и best lap sector time. Удобно: SECTOR ↔ corners mapping
        # есть в tpl.sector_corners.
        best_secs = sector_times_by_lap.get(best_lap_num, {})
        median_secs = {}
        for s_name in SEC_NAMES:
            vals = [sec_tms[s_name] for sec_tms in sector_times_by_lap.values()
                    if s_name in sec_tms]
            if vals:
                median_secs[s_name] = float(np.median(vals))
        # По corner → sector → delta
        for s_i, corners_in in enumerate(tpl.sector_corners):
            s_name = SEC_NAMES[s_i]
            if s_name not in best_secs or s_name not in median_secs:
                continue
            d = median_secs[s_name] - best_secs[s_name]
            share = d / max(len(corners_in), 1)
            for c in corners_in:
                delta_map[c] = share

    # 10. Grip ratio + car-share decomposition
    grip_ratio = _grip_ratio(metrics_by_lap, valid_laps, reference_metrics)
    delta_vs_reference = None
    if reference_metrics and grip_ratio is not None and best_lap_num in metrics_by_lap:
        # Σ времени в поворотах = Σ CORN длин
        total_corner_time = sum(
            metrics_by_lap[best_lap_num][c].get("dur", 0.0)
            for c in metrics_by_lap[best_lap_num]
        )
        car_share_s = (1.0 - grip_ratio) * total_corner_time * 0.5
        delta_vs_reference = {
            "grip_ratio": round(grip_ratio, 3),
            "car_share_s": round(car_share_s, 3),
        }

    # 11. Recommendations (session median) + reference-aware
    hf_vals = []
    for m in metrics_by_lap.values():
        for cm in m.values():
            if np.isfinite(cm.get("hf", np.nan)):
                hf_vals.append(cm["hf"])
    hf_session_median = float(np.median(hf_vals)) if hf_vals else None

    recommendations = generate_recommendations(
        metrics_by_lap=metrics_by_lap,
        best_lap_num=best_lap_num,
        delta_map=delta_map,
        sector_times_by_lap=sector_times_by_lap,
        reference_metrics=reference_metrics,
        hf_session_median=hf_session_median,
        valid_laps=valid_laps,
    )

    # 12. Session summary / potential / trend
    valid_lap_times = [lap_times_s[i] for i, v in enumerate(valid_lap_mask) if v]
    potential_s = compute_potential_s(sector_times_by_lap, valid_laps)
    trend_s = compute_trend_s(valid_lap_times, window=10)
    if valid_lap_times:
        summary = {
            "count": len(lap_times_s),
            "valid_count": len(valid_lap_times),
            "invalid_count": n_invalid,
            "best_s": round(float(min(valid_lap_times)), 3),
            "median_s": round(float(np.median(valid_lap_times)), 3),
            "repeatable_s": round(
                float(np.median(sorted(valid_lap_times)[: max(1, len(valid_lap_times) // 3)])), 3
            ),
        }
    else:
        summary = {"count": len(lap_times_s), "valid_count": 0,
                   "invalid_count": n_invalid, "best_s": None,
                   "median_s": None, "repeatable_s": None}

    # 13. Trajectory for best lap (template-geometry projection via DTW path)
    #    Строим phases по best lap (с session cut), проецируем на template
    #    через hybrid_mapping, и подаём на track.npz x,y (resampled @100Hz).
    #    Это даёт SVG рендер того же стиля что v1.

    best_a = lap_bounds[best_idx]
    best_b = lap_ends[best_idx]
    best_wz = w["wz_f"][best_a:best_b]
    best_along_g = along_sm[best_a:best_b]
    best_alat_g = w["alat"][best_a:best_b]
    best_phases_lap = _classify_phases(best_along_g, cut_g=phase_cut, fs=fs)

    # Трaectория: берём x,y из track.npz @50Hz, upsample до 100Hz
    x_t = np.interp(np.arange(len(tpl.x_50) * 2) / 2,
                    np.arange(len(tpl.x_50)), tpl.x_50)
    y_t = np.interp(np.arange(len(tpl.y_50) * 2) / 2,
                    np.arange(len(tpl.y_50)), tpl.y_50)
    v_t = np.interp(np.arange(len(tpl.v_50) * 2) / 2,
                    np.arange(len(tpl.v_50)), tpl.v_50)
    N_tmpl_100 = len(x_t)

    # Phases на template timeline через hybrid mapping для best lap
    best_path = paths.get(best_lap_num)
    if best_path is None:
        # best не прошёл gate — используем линейное отображение
        best_phases_tmpl = ["cruise"] * N_tmpl_100
    else:
        v_lap_proj = np.abs(best_alat_g) * G / np.maximum(np.abs(best_wz), 0.3)
        v_lap_proj = np.clip(v_lap_proj, 3.0, 20.0)
        hybrid_m = _hybrid_mapping(best_path, best_wz, v_lap_proj,
                                    N_lap=best_b - best_a, ds=ds, fs=fs)
        # j (lap @100Hz) → m[j]*2 (template @100Hz) — hybrid в 50Hz-space
        best_phases_tmpl = ["cruise"] * N_tmpl_100
        for j in range(len(best_phases_lap)):
            if j >= len(hybrid_m):
                break
            # hybrid возвращает template @50Hz в decimated виде; домножить ×ds
            idx_t = int(round(hybrid_m[j] * 1))  # m0 уже умножен в hybrid × ds
            if 0 <= idx_t < N_tmpl_100:
                best_phases_tmpl[idx_t] = best_phases_lap[j]
        # Backfill holes
        last = "cruise"
        for i in range(N_tmpl_100):
            if best_phases_tmpl[i] == "cruise" and i > 0:
                best_phases_tmpl[i] = last if best_phases_tmpl[i] == "cruise" else best_phases_tmpl[i]
            last = best_phases_tmpl[i]

    # Phase counts (best lap, lap-time units)
    phase_counts = {k: best_phases_lap.count(k)
                    for k in ("brake", "lift", "accel", "cruise")}
    phase_counts_total = sum(phase_counts.values()) or 1
    phase_pct = {k: round(v / phase_counts_total * 100, 1)
                 for k, v in phase_counts.items()}

    # Decimated trajectory + per-lap stats
    def _decimate(xs, ys, vs, phases):
        n = len(xs)
        target = min(800, n)
        stride = max(1, n // target)
        out = []
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        for i in range(0, n, stride):
            out.append({
                "s": round(float(cum[i]), 2),
                "x": round(float(xs[i]), 2),
                "y": round(float(ys[i]), 2),
                "phase": phases[i],
                "speed_kmh": round(float(vs[i] * 3.6), 1),
            })
        return out

    track_pts = _decimate(x_t, y_t, v_t, best_phases_tmpl)
    bounds = {
        "min_x": round(float(x_t.min()), 2),
        "max_x": round(float(x_t.max()), 2),
        "min_y": round(float(y_t.min()), 2),
        "max_y": round(float(y_t.max()), 2),
    }

    # 14. Per-lap time list for UI
    per_lap_times = []
    for i in range(N_laps):
        per_lap_times.append({
            "number": lap_numbers[i],
            "time_s": round(lap_times_s[i], 3),
            "source": lap_sources[i],
            "ncc": round(ncc_by_lap.get(lap_numbers[i], 0.0), 3),
            "dtw_residual": round(residuals.get(lap_numbers[i], float("nan")), 3)
                            if np.isfinite(residuals.get(lap_numbers[i], float("nan"))) else None,
            "valid": bool(valid_lap_mask[i]),
            "position": lap_positions[i],
        })

    # 15. Corner summary table for UI (per-corner median across valid laps,
    #     plus best-lap values + reference if present)
    corner_table = []
    for c_name in tpl.corn:
        row = {"name": c_name}
        if c_name in metrics_by_lap.get(best_lap_num, {}):
            bm = metrics_by_lap[best_lap_num][c_name]
            row["best"] = {k: (round(float(v), 3) if np.isfinite(v) else None)
                           for k, v in bm.items()}
        med = {}
        for key in ("vmin_kmh", "alat_pk", "amin", "slip", "brake_len",
                    "lift_len", "hf"):
            vals = [m[c_name][key] for m in metrics_by_lap.values()
                    if c_name in m and np.isfinite(m[c_name].get(key, np.nan))]
            if vals:
                med[key] = round(float(np.median(vals)), 3)
        row["median"] = med
        if reference_metrics and c_name in reference_metrics:
            row["reference"] = {
                k: (round(float(v), 3) if np.isfinite(v) else None)
                for k, v in reference_metrics[c_name].items()
            }
        row["delta_s"] = round(delta_map.get(c_name, 0.0), 3)
        corner_table.append(row)

    # 16. Sector table for UI (per sector best/median/sigma)
    sector_table = []
    for s_name in SEC_NAMES:
        vals = [sec_tms[s_name] for sec_tms in sector_times_by_lap.values()
                if s_name in sec_tms and np.isfinite(sec_tms[s_name])]
        if not vals:
            continue
        sector_table.append({
            "name": s_name,
            "corners": tpl.sector_corners[int(s_name[1:]) - 1],
            "best_s": round(float(min(vals)), 3),
            "median_s": round(float(np.median(vals)), 3),
            "sigma_s": round(float(np.std(vals)), 3),
            "samples": len(vals),
        })

    # 17. Assemble output
    return {
        "status": "ok",
        "track": {
            "name": track_name,
            "trajectory": track_pts,
            "lap_length_m": round(float(tpl.v_50.sum() / 50.0), 1)
                            if tpl.v_50 is not None else None,
        },
        "best_lap": {
            "number": int(best_lap_num),
            "time_s": round(best_time_s, 3),
            "phase_pct": phase_pct,
            "slip_count": sum(
                1 for s in [metrics_by_lap.get(best_lap_num, {}).get(c, {}).get("slip", 0)
                            for c in tpl.corn]
                if s and s > 0.1
            ),
        },
        "all_laps_summary": summary,
        "per_lap_times": per_lap_times,
        "corner_table": corner_table,
        "sector_table": sector_table,
        "sector_times_by_lap": {str(k): v for k, v in sector_times_by_lap.items()},
        "delta_map": {k: round(v, 3) for k, v in delta_map.items()},
        "phase_cut": round(phase_cut, 3),
        "phase_cut_raw": round(raw_cut, 3),
        "invalid_laps_count": n_invalid,
        "invalid_laps": [lap_numbers[i] for i, v in enumerate(valid_lap_mask) if not v],
        "recommendations": recommendations,
        "grip_ratio": round(grip_ratio, 3) if grip_ratio is not None else None,
        "delta_vs_reference": delta_vs_reference,
        "potential_s": round(potential_s, 3) if potential_s is not None else None,
        "trend_s": round(trend_s, 3) if trend_s is not None else None,
        "gates": {
            "session": {"rms_wz": round(sg_info["rms_wz"], 3),
                        "turns": round(sg_info["turns"], 3), "ok": sg_ok},
            "direction_sign": int(dir_sign),
            "timing_match": {
                "ok": timing_match_ok,
                "offset_k": int(offset_k),
                "median_error_s": round(rm_errors_med, 3)
                                  if np.isfinite(rm_errors_med) else None,
            },
        },
        "theta_deg": round(float(np.degrees(frame["theta"]) % 360), 1),
        "theta_corr": round(float(frame["corr"]), 3),
        "track_bounds": bounds,
        "visual": VISUAL_TOKENS,
        "kart_number": kart,
        "notes": notes,
        "corner_metrics_by_lap": {
            str(k): {
                cn: {mk: (round(float(mv), 3) if np.isfinite(mv) else None)
                     for mk, mv in cm.items()}
                for cn, cm in v.items()
            }
            for k, v in metrics_by_lap.items()
        },
    }


__all__ = ["analyze_session", "VISUAL_TOKENS"]
