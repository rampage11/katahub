"""Детектор кругов из IMU — Алгоритм 2 из v2 spec: coarse-by-heading + NCC + fine refinement.

Валидируется на MIKS-данных: Watch IMU 8 минут непрерывной записи,
timing.json как ground truth (36 кругов карт 72). Цели:
- Обнаружить все границы с точностью ±100мс vs RaceMann
- Работать без timing.json (на practice-сессиях без хронометража)

Pipeline:
1. heading ψ(t) = unwrap(yaw from quaternion или ∫wz·dt)
2. Grubosearch: κ(ψ₀) = var(лoc len(l_k)) для фазы ψ₀ по 1° → минимум даёт начальные t_k
3. NCC: шаблон = медианный круг по grubo-границам, корреляция с полным wz
4. Refine: ±1с окно вокруг NCC-пика, local correlation конец+начало шаблона

Используется offline для валидации или на устройстве для live-режима
(ещё не portuguese, задача v3).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def quaternion_yaw(qw, qx, qy, qz):
    """ZYX order yaw из quaternion. Возвращает radians, не unwrapped."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def lowpass(x, fs, cutoff, order=2):
    """Zero-phase Butterworth НЧ."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def unwrap_heading(yaw):
    """Разворот yaw с ±π разрывами в непрерывный ψ(t)."""
    return np.unwrap(yaw)


def detect_laps_from_heading(psi, t, direction_sign=-1.0):
    """Grubo-детекция по 2π периодичности heading.

    direction_sign: -1 для ccw (yaw падает), +1 для cw. На MIKS трасса ccw.
    Перебираем фазу ψ₀ по 1° и выбираем ту что даёт минимальную дисперсию
    длительностей кругов. Возвращает список индексов t[].
    """
    # Нормализуем так чтобы кумулятивный ψ был монотонен в желаемом направлении.
    psi_norm = direction_sign * psi
    psi_rel = psi_norm - psi_norm[0]

    # Общее число кругов ≈ накопленный ψ / 2π. Берём 2 запаса на дрейф.
    total = psi_rel[-1] / (2 * np.pi)
    if total < 1.5:
        return []
    n_laps_approx = int(np.floor(total))
    if n_laps_approx < 2:
        return []

    best_boundaries = []
    best_var = np.inf
    for deg in range(0, 360):
        phase = np.radians(deg)
        # Все пересечения psi_rel = phase + k*2π
        k_targets = phase + 2 * np.pi * np.arange(1, n_laps_approx + 2)
        boundaries = []
        for target in k_targets:
            cross = np.where(np.diff(np.sign(psi_rel - target)))[0]
            if len(cross) > 0:
                boundaries.append(int(cross[0]))
        if len(boundaries) < 3:
            continue
        times = np.diff(t[boundaries])
        var = np.var(times)
        if var < best_var:
            best_var = var
            best_boundaries = boundaries
    return best_boundaries


def build_template(wz, t, boundaries, template_len_s=33.0, fs=100.0):
    """Шаблон круга как медианный wz по всем grubo-кругам.

    Нормализуем длительности к template_len_s, затем медиана поэлементно.
    """
    n = int(template_len_s * fs)
    template_mat = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s < n * 0.7 or e - s > n * 1.5:
            continue
        # resample под фиксированную длину n
        xs = np.linspace(0, 1, e - s)
        xs_target = np.linspace(0, 1, n)
        row = np.interp(xs_target, xs, wz[s:e])
        template_mat.append(row)
    if not template_mat:
        return None
    return np.median(np.array(template_mat), axis=0)


def ncc_refine(wz, t, template, boundaries, fs=100.0):
    """Normalized cross-correlation для уточнения границ.

    Для каждой пары смежных grubo-границ находим пик NCC шаблона c wz в окне
    ±1 сек. Возвращает уточнённые индексы границ.
    """
    tn = len(template)
    refined = [boundaries[0]]
    for b in boundaries[1:]:
        win = int(fs * 1.0)
        lo = max(0, b - tn - win)
        hi = min(len(wz) - tn, b - tn + win)
        if hi <= lo:
            refined.append(b)
            continue
        scores = []
        for off in range(lo, hi):
            seg = wz[off:off + tn]
            if len(seg) != tn:
                continue
            # Normalized correlation
            seg_c = seg - seg.mean()
            t_c = template - template.mean()
            denom = (np.linalg.norm(seg_c) * np.linalg.norm(t_c))
            if denom < 1e-9:
                scores.append(-np.inf)
                continue
            scores.append(np.dot(seg_c, t_c) / denom)
        if not scores:
            refined.append(b)
            continue
        best_off = lo + int(np.argmax(scores))
        refined.append(best_off + tn)
    return refined


def detect_laps(samples, direction_sign=-1.0, fs=100.0):
    """Основная функция: samples — ndarray dtype IMU_DTYPE.

    Возвращает dict {
        'boundaries_idx': list[int],  # индексы в samples
        'boundaries_s': list[float],  # секунды от начала записи
        'lap_times_s': list[float],
        'coarse_boundaries_idx': list[int],
        'template': ndarray | None,
    }
    """
    ts = samples['ts_ns'].astype(np.int64)
    t = (ts - ts[0]) / 1e9

    # Watch IMU имеет quaternion. Используем его если доступно, иначе ∫wz.
    if 'qw' in samples.dtype.names:
        yaw = quaternion_yaw(samples['qw'], samples['qx'], samples['qy'], samples['qz'])
        psi = unwrap_heading(yaw)
    else:
        wz = samples['gz']
        # численное интегрирование с dt
        dt = np.diff(t, prepend=t[0])
        psi = np.cumsum(wz * dt)

    # НЧ фильтруем yaw rate для шаблона/NCC
    wz = samples['gz']
    wz_f = lowpass(wz, fs=fs, cutoff=1.5, order=2)

    coarse = detect_laps_from_heading(psi, t, direction_sign=direction_sign)
    if len(coarse) < 3:
        return {
            'boundaries_idx': coarse,
            'boundaries_s': [t[i] for i in coarse],
            'lap_times_s': [],
            'coarse_boundaries_idx': coarse,
            'template': None,
        }

    # Шаблон из грубых границ
    template = build_template(wz_f, t, coarse, template_len_s=np.median(np.diff(t[coarse])), fs=fs)
    if template is None:
        refined = coarse
    else:
        refined = ncc_refine(wz_f, t, template, coarse, fs=fs)

    lap_times = [t[refined[i + 1]] - t[refined[i]] for i in range(len(refined) - 1)]
    return {
        'boundaries_idx': refined,
        'boundaries_s': [t[i] for i in refined],
        'lap_times_s': lap_times,
        'coarse_boundaries_idx': coarse,
        'template': template,
    }


def validate_against_timing(detected_s, timing_laps_s):
    """Сравнивает detected boundaries с ground truth от RaceMann.

    timing_laps_s = список времен прохождения отсечки (wall sec from
    первый круг). Возвращает ошибки |detected - truth| попарно.
    """
    errors = []
    for ts in detected_s:
        if not timing_laps_s:
            break
        best = min(timing_laps_s, key=lambda x: abs(x - ts))
        errors.append(ts - best)
    return errors
