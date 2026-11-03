"""Standalone-пакет по кругу 31:
  - lap31.png (статика — трасса с фазами и корнерами)
  - lap31.html (анимация точки карта по кругу, в один файл с embedded SVG)
  - lap31_imu.csv (сырые IMU-сэмплы круга 31)
  - lap31_phases.csv (per-sample phase и ключевые метрики)
  - lap31_meta.json (θ, inverted, closure, корнеры, slip-события, фазы по времени)

Запуск:
  python3 lap31_standalone.py --out /tmp/lap31_report
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from karttelemetry import loader, full_pipeline as fp


def compute_session_sf_offset(samples=None, timing=None, kart_number='72',
                               direction_sign=1.0, fs=100.0):
    """Один раз на сессию: медианный offset (сек) от IMU boundary start
    до RaceMann физ. S/F. RaceMann wallTime имеет ±1с квантование —
    усредняется медианой по всем кругам. Возвращает (offset_s, dbg_list).

    samples/timing по умолчанию берутся из fallback-путей (MIKS сессия) —
    для стандaлон-запуска из `main()`. Все внешние вызовы (katahub backend)
    передают данные явно.
    """
    if samples is None:
        wimu = Path('/tmp/all_sessions/sessions/2026-10-05T19-13-41_2D5D4AAF/watch_imu.bin')
        samples = np.fromfile(wimu, dtype=loader.IMU_DTYPE)
    if timing is None:
        tj = Path('/tmp/all_sessions/sessions/2026-10-05T19-21-26_35951FED/timing.json')
        timing = json.loads(tj.read_text())
    kart_str = str(kart_number)
    from karttelemetry import lap_detector as ld
    det = ld.detect_laps(samples, direction_sign=direction_sign, fs=fs)
    boundaries = det['boundaries_idx']
    ts_ns_arr = samples['ts_ns'].astype(np.int64)
    pair_mid_wall = [(ts_ns_arr[boundaries[i]] + ts_ns_arr[boundaries[i + 1]]) / 2
                     for i in range(len(boundaries) - 1)]
    import datetime as dt
    our = [p for p in timing['allPasses'] if str(p.get('kartNumber')) == kart_str and p.get('lapNumber')]
    # Dedupe по lap_number
    seen_laps = set()
    offsets = []
    for p in our:
        if p['lapNumber'] in seen_laps:
            continue
        seen_laps.add(p['lapNumber'])
        try:
            rm_end = int(dt.datetime.fromisoformat(p['wallTime'].replace('Z', '+00:00')).timestamp() * 1e9)
            rm_start = rm_end - p['lastLapMs'] * 1_000_000
            rm_mid = (rm_start + rm_end) / 2
            bi = int(np.argmin([abs(mw - rm_mid) for mw in pair_mid_wall]))
            # Фильтр: круг должен попасть в IMU window. Lap_dur ≈ 33с →
            # mid_diff < half-lap (15с) гарантирует что IMU-пара соответствует
            # этому кругу, а не ближайшему в окне (при вне-окна круге diff
            # сотни секунд).
            mid_diff = abs(rm_mid - pair_mid_wall[bi]) / 1e9
            if mid_diff > 15.0:
                continue
            offs = (rm_start - ts_ns_arr[boundaries[bi]]) / 1e9
            offsets.append(offs)
        except Exception:
            pass
    if not offsets:
        return 0.0, []
    offs_med = float(np.median(offsets))
    offs_std = float(np.std(offsets))
    print(f'[session] S/F offset median={offs_med:+.2f}s, std={offs_std:.2f}s (n={len(offsets)} laps in IMU window)')
    return offs_med, offsets


def compute_session_phase_cut(samples, kart_number='72', fs=100.0) -> float:
    """Session-wide Otsu brake cut (v2 §6).

    Пробегает все IMU-кругки (NCC-detected boundaries), собирает `amin`
    (минимум сглаженного a_long в окне [entry-1.5s, exit] по CORN_MIKS),
    запускает Otsu, clamp [-0.55, -0.40] g.

    Используется laps_compare / process_lap когда нужен единый порог
    на сессию без вызова full pipeline.
    """
    try:
        from karttelemetry.full_pipeline import analyze_session
    except Exception:
        return -0.47
    # Не стартуем full pipeline (слишком тяжело для compare path).
    # Простая оценка — Otsu по медианным amin'ам 10 корнеров через detect_laps.
    # Падаем на fallback -0.47 если что.
    from karttelemetry import lap_detector as ld
    from karttelemetry.track_template import CORN_MIKS, PHASE_CUT_MIN, PHASE_CUT_MAX
    from karttelemetry.corner_metrics import otsu_threshold
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, filtfilt
    try:
        det = ld.detect_laps(samples, direction_sign=1.0, fs=fs)
        bounds = det['boundaries_idx']
        if len(bounds) < 3:
            return -0.47
    except Exception:
        return -0.47
    # Собираем a_long смотря все валидные круги через те же маневры что в
    # process_lap, но дешевле — на всей записи.
    def rotate(vx, vy, vz, qw, qx, qy, qz):
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wzq = qw * qx, qw * qy, qw * qz
        rx = (1 - 2 * (yy + zz)) * vx + 2 * (xy - wzq) * vy + 2 * (xz + wy) * vz
        ry = 2 * (xy + wzq) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz - wx) * vz
        rz = 2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (1 - 2 * (xx + yy)) * vz
        return rx, ry, rz
    ax_w, ay_w, _ = rotate(samples['ax'], samples['ay'], samples['az'],
                           samples['qw'], samples['qx'], samples['qy'], samples['qz'])
    _, _, wz_w = rotate(samples['gx'], samples['gy'], samples['gz'],
                         samples['qw'], samples['qx'], samples['qy'], samples['qz'])
    yaw = np.arctan2(2 * (samples['qw'] * samples['qz'] + samples['qx'] * samples['qy']),
                      1 - 2 * (samples['qy'] * samples['qy'] + samples['qz'] * samples['qz']))
    psi = np.unwrap(yaw)
    b2, a2 = butter(2, 1.5 / (fs / 2))
    axf = filtfilt(b2, a2, ax_w)
    ayf = filtfilt(b2, a2, ay_w)
    wz_f = filtfilt(b2, a2, wz_w)
    best_th, best_c = 0.0, -2.0
    seg = slice(bounds[1], bounds[-1])
    for deg in range(0, 360, 2):
        th = np.radians(deg)
        alat = -axf * np.sin(psi + th) + ayf * np.cos(psi + th)
        if alat[seg].std() < 1e-6 or wz_f[seg].std() < 1e-6:
            continue
        c = float(np.corrcoef(alat[seg], wz_f[seg])[0, 1])
        if c > best_c:
            best_c = c
            best_th = th
    along_full = axf * np.cos(psi + best_th) + ayf * np.sin(psi + best_th)
    along_sm = gaussian_filter1d(along_full, int(0.08 * fs))

    # Берём amin'ы для каждого круга × окно поворота (CORN_MIKS в template
    # времени; используем простую линейную проекцию 0..1 → 0..dur).
    amins = []
    for lap_idx in range(len(bounds) - 1):
        a = bounds[lap_idx]
        b = bounds[lap_idx + 1]
        dur = (b - a) / fs
        if dur < 25 or dur > 45:
            continue
        for name, (t0, t1) in CORN_MIKS.items():
            frac0 = t0 / 33.17
            frac1 = t1 / 33.17
            i0 = a + int(frac0 * (b - a))
            i1 = a + int(frac1 * (b - a))
            i0b = max(a, i0 - int(1.5 * fs))
            if i1 <= i0b or i1 > len(along_sm):
                continue
            amins.append(float(along_sm[i0b:i1].min()))
    if len(amins) < 10:
        return -0.47
    raw = otsu_threshold(np.array(amins))
    return float(np.clip(raw, PHASE_CUT_MIN, PHASE_CUT_MAX))


def process_lap(lap_number: int = 31, samples=None, timing=None,
                 kart_number='72', sf_offset_override: float | None = None,
                 cut_g: float | None = None):
    """Единая точка обработки круга. samples/timing — опционально; если
    не переданы, используются hardcoded fallback пути (для стандaлон-запуска).

    `cut_g` — session-wide brake cut (v2 §6); по умолчанию считается через
    `compute_session_phase_cut(samples)`. ХАРДКОД `-0.25` УДАЛЁН.
    """
    if samples is None:
        wimu = Path('/tmp/all_sessions/sessions/2026-10-05T19-13-41_2D5D4AAF/watch_imu.bin')
        assert wimu.exists(), f'missing {wimu}'
        samples = np.fromfile(wimu, dtype=loader.IMU_DTYPE)
    if timing is None:
        tj = Path('/tmp/all_sessions/sessions/2026-10-05T19-21-26_35951FED/timing.json')
        assert tj.exists(), f'missing {tj}'
        timing = json.loads(tj.read_text())

    kart_str = str(kart_number)
    our = [p for p in timing['allPasses'] if str(p.get('kartNumber')) == kart_str and p.get('lapNumber')]
    by_num = {p['lapNumber']: p for p in our}

    # IMU-based lap boundaries (NCC+DTW) — RaceMann wallTime даёт ±1с ошибку.
    from karttelemetry import lap_detector as ld
    det = ld.detect_laps(samples, direction_sign=1.0, fs=100.0)
    boundaries = det['boundaries_idx']
    if len(boundaries) < 2:
        raise RuntimeError('IMU lap detection failed')
    ts_ns_arr = samples['ts_ns'].astype(np.int64)
    fs_nominal = 100.0

    if lap_number in by_num:
        # ---- Timing-based режим: RaceMann ground truth ----
        lap_rec = by_num[lap_number]
        import datetime as dt
        pair_mid_wall = [(ts_ns_arr[boundaries[i]] + ts_ns_arr[boundaries[i + 1]]) / 2
                         for i in range(len(boundaries) - 1)]
        rm_end = int(dt.datetime.fromisoformat(lap_rec['wallTime'].replace('Z', '+00:00')).timestamp() * 1e9)
        rm_start = rm_end - lap_rec['lastLapMs'] * 1_000_000
        rm_mid = (rm_start + rm_end) / 2
        best_pair_i = int(np.argmin([abs(mw - rm_mid) for mw in pair_mid_wall]))
        i0_raw = boundaries[best_pair_i]
        i1_raw = boundaries[best_pair_i + 1]

        # S/F offset: один на сессию.
        if sf_offset_override is not None:
            sf_offset_s = sf_offset_override
        else:
            sf_offset_s = (rm_start - ts_ns_arr[i0_raw]) / 1e9
        off = int(round(sf_offset_s * fs_nominal))
        i0 = max(0, i0_raw + off)
        imu_next = i1_raw + off
        rm_end_ns = ts_ns_arr[i0] + lap_rec['lastLapMs'] * 1_000_000
        i1_rm = int(np.searchsorted(ts_ns_arr, rm_end_ns, side='right'))
        if abs((imu_next - i0) - (i1_rm - i0)) < int(0.5 * fs_nominal):
            i1 = i1_rm
        else:
            i1 = imu_next
    else:
        # ---- IMU-fallback: lap_number как 1-based индекс в boundaries ----
        # Применяется когда timing.json пустой (TimingClient не поймал гонку)
        # или круг отсутствует в timing. Границы берутся напрямую от NCC+DTW.
        n_pairs = len(boundaries) - 1
        if lap_number < 1 or lap_number > n_pairs:
            raise RuntimeError(f'lap {lap_number} out of IMU-detected range 1..{n_pairs} (timing пустой)')
        i0 = int(boundaries[lap_number - 1])
        i1 = int(boundaries[lap_number])
        # Синтетический lap_rec: lastLapMs = реальная длина окна в мс.
        rm_dur_ms = int((ts_ns_arr[i1 - 1] - ts_ns_arr[i0]) / 1e6) if i1 > i0 + 1 else 33000
        lap_rec = {'lapNumber': lap_number, 'lastLapMs': rm_dur_ms, 'wallTime': ''}
        sf_offset_s = 0.0
        off = 0
    i1 = min(i1, len(samples))
    L = samples[i0:i1]
    fs = len(L) / ((ts_ns_arr[i1 - 1] - ts_ns_arr[i0]) / 1e9) if i1 > i0 + 1 else fs_nominal
    imu_dur_s = (ts_ns_arr[i1 - 1] - ts_ns_arr[i0]) / 1e9 if i1 > i0 + 1 else 0.0
    rm_dur_s = lap_rec['lastLapMs'] / 1000.0
    print(f'Lap {lap_number}: RaceMann {rm_dur_s:.3f}s, window {imu_dur_s:.3f}s, '
          f'S/F shift {off:+d} samples ({sf_offset_s:+.2f}s)')
    if abs(imu_dur_s - rm_dur_s) > 0.5:
        print(f'  WARN: large window/timing mismatch')

    # Rotation body→world
    def rotate(vx, vy, vz, qw, qx, qy, qz):
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wzq = qw * qx, qw * qy, qw * qz
        rx = (1 - 2 * (yy + zz)) * vx + 2 * (xy - wzq) * vy + 2 * (xz + wy) * vz
        ry = 2 * (xy + wzq) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz - wx) * vz
        rz = 2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (1 - 2 * (xx + yy)) * vz
        return rx, ry, rz

    ax_w, ay_w, _ = rotate(L['ax'], L['ay'], L['az'], L['qw'], L['qx'], L['qy'], L['qz'])
    _, _, wz_w = rotate(L['gx'], L['gy'], L['gz'], L['qw'], L['qx'], L['qy'], L['qz'])
    yaw = np.arctan2(2 * (L['qw'] * L['qz'] + L['qx'] * L['qy']),
                     1 - 2 * (L['qy'] * L['qy'] + L['qz'] * L['qz']))
    psi = np.unwrap(yaw)
    t = np.arange(len(L)) / fs

    # θ grid search по 1° по всему кругу (max SIGNED corr)
    best_th, best_corr = 0.0, -2.0
    for deg in range(360):
        th = np.deg2rad(deg)
        alat = -ax_w * np.sin(psi + th) + ay_w * np.cos(psi + th)
        c = float(np.corrcoef(alat, wz_w)[0, 1])
        if c > best_corr:
            best_corr = c
            best_th = th
    th = best_th
    print(f'θ = {np.degrees(th):.0f}°  corr = {best_corr:+.3f}')

    # a_long/a_lat
    c = np.cos(psi + th)
    s = np.sin(psi + th)
    a_long = ax_w * c + ay_w * s
    a_lat = -ax_w * s + ay_w * c

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import butter, filtfilt
    b2, a2 = butter(2, 1.5 / (fs / 2))
    wz_f = filtfilt(b2, a2, wz_w)
    a_lat_f = filtfilt(b2, a2, a_lat)
    ah_f = filtfilt(b2, a2, np.hypot(ax_w, ay_w))
    a_long_sm = gaussian_filter1d(a_long, 8)

    # Physical sign check (на всякий)
    wz_abs = np.abs(wz_f)
    peaks = []
    i = int(2 * fs)
    while i < len(wz_abs) - int(2 * fs):
        if wz_abs[i] >= 1.0:
            end = min(i + int(4 * fs), len(wz_abs))
            mi = i + int(np.argmax(wz_abs[i:end]))
            peaks.append(mi)
            i = mi + int(2 * fs)
        else:
            i += 1
    w1s = int(1.5 * fs)
    delta_sum = 0.0
    for p in peaks:
        if p - w1s < 0 or p + w1s >= len(a_long_sm):
            continue
        delta_sum += (a_long_sm[p:p + w1s].mean() - a_long_sm[p - w1s:p].mean())
    sign_ok = delta_sum > 0
    print(f'Sign check: {len(peaks)} peaks, meanΔ = {delta_sum / max(len(peaks), 1):+.3f}g '
          f'→ {"OK" if sign_ok else "FLIP"}')
    if not sign_ok:
        th += math.pi
        c = np.cos(psi + th)
        s = np.sin(psi + th)
        a_long = ax_w * c + ay_w * s
        a_lat = -ax_w * s + ay_w * c
        a_long_sm = gaussian_filter1d(a_long, 8)
        a_lat_f = filtfilt(b2, a2, a_lat)

    # Трaectory: v из |a_lat|/|wz| в поворотах
    mask_v = np.abs(wz_f) > 1.0
    v = np.where(mask_v, np.abs(a_lat_f) * 9.81 / np.maximum(np.abs(wz_f), 0.1), np.nan)
    v = np.clip(v, 4, 20)
    idx = np.arange(len(v))
    good = ~np.isnan(v)
    if good.sum() > 1:
        v = np.interp(idx, idx[good], v[good])
    v = gaussian_filter1d(v, 15, mode='wrap')

    # Heading закрыть до 2π
    psi_i = np.cumsum(wz_f) / fs
    total = psi_i[-1] - psi_i[0]
    target = round(total / (2 * np.pi)) * 2 * np.pi
    if abs(target) > 1e-6:
        psi_i = psi_i - (total - target) * idx / max(len(psi_i) - 1, 1)

    x = np.cumsum(v * np.cos(psi_i)) / fs
    y = np.cumsum(v * np.sin(psi_i)) / fs
    gap = float(np.hypot(x[-1], y[-1]))
    x = x - x[-1] * idx / max(len(x) - 1, 1)
    y = y - y[-1] * idx / max(len(y) - 1, 1)
    print(f'Trajectory: len={v.sum()/fs:.1f}m, closure={gap:.1f}m')

    # Phases + corners + slips.
    # v2 §6: brake < cut_g (session-wide Otsu, clamp [-0.55,-0.40]),
    # lift [cut_g, -0.12g], accel > +0.20g, иначе cruise.
    # Исключение: при |a_lat| > 0.6g "lift" НЕ рисуем (утечка θ и скраб шин),
    # оставляем "cruise".
    if cut_g is None:
        try:
            cut_g = compute_session_phase_cut(samples, kart_number=kart_number,
                                               fs=fs_nominal)
        except Exception:
            cut_g = -0.47
    print(f'phase cut g = {cut_g:.3f}')

    def cls(a, alat_g, _cut=cut_g):
        """a в g (a_long_sm уже gaussian-отфильтрован), alat_g в g."""
        if a < _cut: return 'brake'
        if a < -0.12:
            # Исключение "lift" при высокой боковой — утечка θ или скраб шин
            if abs(alat_g) > 0.6:
                return 'cruise'
            return 'lift'
        if a > 0.20:  return 'accel'
        return 'cruise'
    # a_lat_f в m/s² (×G в orient-блоке) — делим на G ОДИН раз, передаём в g.
    phases = [cls(a_long_sm[i], a_lat_f[i] / 9.81) for i in range(len(a_long_sm))]
    # Поглощение сегментов <150мс
    min_len = int(0.15 * fs)
    runs = []
    i0 = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[i0]:
            runs.append([i0, i, phases[i0]])
            i0 = i
    cleaned = runs[:]
    for i, r in enumerate(cleaned):
        if r[1] - r[0] < min_len and 0 < i < len(cleaned) - 1:
            cleaned[i][2] = cleaned[i - 1][2]
    phases_final = [''] * len(phases)
    for s_, e_, p in cleaned:
        for k in range(s_, e_):
            phases_final[k] = p

    # Corners на этом круге: сегменты const sign(wz_f) с peak>0.8, dur>0.8s
    sign = np.sign(wz_f)
    sign[np.abs(wz_f) < 0.15] = 0
    corners = []
    i = 0
    while i < len(wz_f):
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j < len(wz_f) and sign[j] == sign[i]:
            j += 1
        if j - i > int(0.8 * fs) and np.abs(wz_f[i:j]).max() > 0.8:
            apex = i + int(np.argmax(np.abs(wz_f[i:j])))
            corners.append({
                'start_t': i / fs, 'end_t': j / fs, 'apex_t': apex / fs,
                'apex_x': float(x[apex]), 'apex_y': float(y[apex]),
                'apex_speed_kmh': float(v[apex] * 3.6),
                'direction': int(sign[i]),
            })
        i = j
    for idx_, c_ in enumerate(corners, 1):
        c_['name'] = f'T{idx_}'
    print(f'Corners: {len(corners)}')

    # Slips (с реальной v)
    wz_exp = a_lat_f * 9.81 / np.maximum(v, 3.0)
    excess = (np.abs(wz_f) - np.abs(wz_exp)) * (np.sign(wz_f) == np.sign(a_lat_f))
    excess = gaussian_filter1d(np.maximum(excess, 0), int(0.06 * fs))
    slip_events = []
    i = 0
    while i < len(excess):
        if excess[i] <= 0.28:
            i += 1
            continue
        j = i
        while j < len(excess) and excess[j] > 0.28:
            j += 1
        if j - i >= int(0.08 * fs):
            peak = i + int(np.argmax(excess[i:j]))
            slip_events.append({
                't': peak / fs,
                'x': float(x[peak]), 'y': float(y[peak]),
                'strength': float(excess[peak]),
            })
        i = j
    print(f'Slips: {len(slip_events)}')

    # НЕТ np.roll — окно уже сдвинуто при вырезке, index 0 = физ. S/F,
    # массивы непрерывны, шва нет. Все вычисления выше использовали эту
    # непрерывную L[].

    return {
        'fs': fs,
        'L': L,
        't': t,
        'x': x, 'y': y, 'v': v,
        'a_long': a_long, 'a_long_sm': a_long_sm,
        'a_lat': a_lat_f, 'wz': wz_f, 'ah': ah_f, 'psi': psi_i,
        'phases': phases_final,
        'corners': corners,
        'slips': slip_events,
        'theta_deg': float(np.degrees(th % (2 * math.pi))),
        'corr': best_corr,
        'closure_m': gap,
        'lap_len_m': float(v.sum() / fs),
        'raceman_time_s': lap_rec['lastLapMs'] / 1000.0,
        'lap_number': lap_number,
        'cut_g': float(cut_g),
    }


# ----- SVG RENDER (static + animated) -----

PHASE_COLOR = {'brake': '#E24B4A', 'lift': '#EF9F27', 'accel': '#1D9E75', 'cruise': '#B4B2A9'}
PHASE_WIDTH = {'brake': 9, 'lift': 7, 'accel': 6, 'cruise': 4}


def world_to_svg(x, y, bounds, size=(680, 560), pad=60):
    w = size[0] - 2 * pad
    h = size[1] - 2 * pad
    bx = bounds['maxx'] - bounds['minx']
    by = bounds['maxy'] - bounds['miny']
    scale = min(w / bx, h / by)
    sx = pad + (w - bx * scale) / 2 + (x - bounds['minx']) * scale
    sy = pad + h - ((y - bounds['miny']) * scale + (h - by * scale) / 2)
    return sx, sy


def render_static_svg(d, out_svg, out_png):
    bounds = {'minx': d['x'].min(), 'maxx': d['x'].max(),
              'miny': d['y'].min(), 'maxy': d['y'].max()}
    size = (680, 560); pad = 60
    pts = [world_to_svg(xi, yi, bounds, size, pad) for xi, yi in zip(d['x'], d['y'])]

    parts = [
        f'<svg viewBox="0 0 {size[0]} {size[1]}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#F6F5F0"/>',
        f'<text x="40" y="30" font-family="system-ui" font-size="14" fill="#2C2C2A" font-weight="500">Круг {d.get("lap_number", "?")} · {d["raceman_time_s"]:.3f} с · θ={d["theta_deg"]:.0f}° corr={d["corr"]:+.2f} · closure {d["closure_m"]:.1f}м</text>',
    ]

    # pavement
    dpav = 'M ' + ' L '.join(f'{sx:.1f} {sy:.1f}' for sx, sy in pts) + ' Z'
    parts.append(f'<path d="{dpav}" stroke="#D3D1C7" stroke-width="22" fill="none" stroke-linejoin="round" stroke-linecap="round" opacity="0.55"/>')

    # phase segments
    phases = d['phases']
    i0 = 0
    for i in range(1, len(phases) + 1):
        if i == len(phases) or phases[i] != phases[i0]:
            seg = pts[i0:min(i + 1, len(pts))]
            if len(seg) >= 2:
                dseg = 'M ' + ' L '.join(f'{sx:.1f} {sy:.1f}' for sx, sy in seg)
                col = PHASE_COLOR.get(phases[i0], '#B4B2A9')
                wid = PHASE_WIDTH.get(phases[i0], 4)
                parts.append(f'<path d="{dseg}" stroke="{col}" stroke-width="{wid}" fill="none" stroke-linejoin="round" stroke-linecap="round"/>')
            i0 = i

    # slip markers
    for sl in d['slips']:
        sx, sy = world_to_svg(sl['x'], sl['y'], bounds, size, pad)
        r = 6 + max(0, sl['strength'] - 0.28) * 22
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.1f}" fill="none" stroke="#534AB7" stroke-width="2"/>')
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.5" fill="#534AB7"/>')

    # corner labels + speeds
    for c_ in d['corners']:
        cx, cy = world_to_svg(c_['apex_x'], c_['apex_y'], bounds, size, pad)
        parts.append(f'<text x="{cx:.1f}" y="{(cy - 14):.1f}" font-family="system-ui" font-size="11" fill="#888780" text-anchor="middle">{c_["name"]}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{(cy + 24):.1f}" font-family="system-ui" font-size="13" font-weight="700" fill="#2C2C2A" text-anchor="middle">{int(c_["apex_speed_kmh"])}</text>')

    # start/finish tick
    sx0, sy0 = pts[0]
    if len(pts) > 5:
        tx = pts[5][0] - sx0; ty = pts[5][1] - sy0
        L = math.hypot(tx, ty) or 1
        nx = -ty / L * 16; ny = tx / L * 16
        parts.append(f'<line x1="{sx0 + nx:.1f}" y1="{sy0 + ny:.1f}" x2="{sx0 - nx:.1f}" y2="{sy0 - ny:.1f}" stroke="#2C2C2A" stroke-width="3"/>')
        parts.append(f'<text x="{sx0:.1f}" y="{(sy0 + 32):.1f}" font-family="system-ui" font-size="10" fill="#888780" text-anchor="middle">старт / финиш</text>')

    # legend
    lx, ly = 490, 70
    parts.append(f'<text x="{lx}" y="{ly}" font-family="system-ui" font-size="14" font-weight="500" fill="#2C2C2A">Легенда</text>')
    cut_label = d.get('cut_g', -0.25)
    for k, (ph, lbl) in enumerate([('brake', f'Тормоз (< {cut_label:.2f} g)'),
                                     ('lift', 'Сброс / лёгкий трейл'),
                                     ('accel', 'Газ'),
                                     ('cruise', 'Ровно')]):
        yy = ly + 24 * (k + 1)
        parts.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 28}" y2="{yy}" stroke="{PHASE_COLOR[ph]}" stroke-width="{PHASE_WIDTH[ph]}" stroke-linecap="round"/>')
        parts.append(f'<text x="{lx + 38}" y="{yy + 4}" font-family="system-ui" font-size="12" fill="#5F5E5A">{lbl}</text>')
    yy = ly + 24 * 5 + 10
    parts.append(f'<circle cx="{lx + 14}" cy="{yy}" r="9" fill="none" stroke="#534AB7" stroke-width="2"/>')
    parts.append(f'<circle cx="{lx + 14}" cy="{yy}" r="2.5" fill="#534AB7"/>')
    parts.append(f'<text x="{lx + 38}" y="{yy + 4}" font-family="system-ui" font-size="12" fill="#5F5E5A">Проскальзывание</text>')
    yy += 26
    parts.append(f'<text x="{lx}" y="{yy}" font-family="system-ui" font-size="11" fill="#888780">Цифры — скорость в апексе, км/ч</text>')
    yy += 18
    parts.append(f'<text x="{lx}" y="{yy}" font-family="system-ui" font-size="11" fill="#888780">Длина круга {d["lap_len_m"]:.0f}м</text>')

    parts.append('</svg>')
    svg = '\n'.join(parts)
    Path(out_svg).write_text(svg)

    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out_png), output_width=1360)
    print(f'wrote {out_svg} + {out_png}')
    return svg, bounds, pts


def render_animated_html(d, svg_static, bounds, pts, out_html):
    """Standalone HTML с анимацией. Данные встроены как JSON. Весь JS —
    простой, без f-string-экранирования шаблонных литералов (чтобы ничего
    не ломалось при генерации)."""
    import json as J
    data = {
        'pts': [[round(sx, 1), round(sy, 1)] for sx, sy in pts],
        'phases': d['phases'],
        'phase_color': PHASE_COLOR,
        'phase_width': PHASE_WIDTH,
        'fs': d['fs'],
        'corners': [{'name': c['name'], 'x': c['apex_x'], 'y': c['apex_y'],
                     'speed': c['apex_speed_kmh']} for c in d['corners']],
        'slips': [{'x': s['x'], 'y': s['y'], 'strength': s['strength']} for s in d['slips']],
        'bounds': bounds,
        'lap_time_s': d['raceman_time_s'],
        'theta_deg': d['theta_deg'],
    }
    data_json = J.dumps(data, ensure_ascii=False)

    # Собираем HTML БЕЗ Python f-string внутри JS-кода.
    head = (
        '<!doctype html>\n'
        f'<html lang="ru"><head><meta charset="utf-8"><title>Круг {d.get("lap_number", "?")} — анимация</title>\n'
        '<style>\n'
        'body { font-family: -apple-system, system-ui, sans-serif; background: #F6F5F0; color: #2C2C2A; margin: 0; padding: 20px; }\n'
        '.row { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }\n'
        '.sideinfo { font-size: 13px; line-height: 1.6; }\n'
        '.control { margin: 10px 0; }\n'
        'button { padding: 6px 12px; font-size: 13px; margin-right: 6px; }\n'
        '#status { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #5F5E5A; }\n'
        'h2 { margin: 0 0 6px; font-size: 15px; color: #5F5E5A; text-transform: uppercase; letter-spacing: .05em; }\n'
        '</style></head><body>\n'
    )
    header = (
        f'<h1 style="margin:0 0 16px">Круг {d.get("lap_number", "?")} · {d["raceman_time_s"]:.3f} с (RaceMann)</h1>\n'
        '<div class="control">\n'
        '<button id="btnPlay">▶ Play</button>\n'
        '<button id="btnPause">⏸ Pause</button>\n'
        '<button id="btnReset">⟲ Reset</button>\n'
        '<label>Speed: <select id="speedSel">'
        '<option value="0.25">0.25×</option>'
        '<option value="0.5">0.5×</option>'
        '<option value="1" selected>1× (real-time)</option>'
        '<option value="2">2×</option>'
        '<option value="5">5×</option>'
        '</select></label>\n'
        '<span id="status">—</span>\n'
        '</div>\n'
    )
    svg_block = (
        '<div class="row">\n'
        '<svg id="svg" viewBox="0 0 680 560" width="680" height="560">\n'
        '<rect width="100%" height="100%" fill="#F6F5F0"/>\n'
        '<path id="pavement" fill="none" stroke="#D3D1C7" stroke-width="22" '
        'stroke-linejoin="round" stroke-linecap="round" opacity="0.55"/>\n'
        '<g id="trailLayer"></g>\n'
        '<g id="cornerLayer"></g>\n'
        '<g id="slipLayer"></g>\n'
        '<line id="startFin" stroke="#2C2C2A" stroke-width="3"/>\n'
        '<circle id="kart" r="8" fill="#22d3ee" stroke="#2C2C2A" stroke-width="2" cx="340" cy="280"/>\n'
        '</svg>\n'
    )
    side = (
        '<div class="sideinfo">\n'
        '<h2>Данные</h2>\n'
        f'θ: <b>{d["theta_deg"]:.0f}°</b> corr <b>{d["corr"]:+.2f}</b><br>\n'
        f'Длина: <b>{d["lap_len_m"]:.0f} м</b> closure <b>{d["closure_m"]:.1f} м</b><br>\n'
        f'Корнеров: <b>{len(d["corners"])}</b><br>\n'
        f'Срывов: <b>{len(d["slips"])}</b><br>\n'
        f'Семплов: <b>{len(pts)}</b> @ 100Hz<br><br>\n'
        '<h2>Легенда</h2>\n'
        f'<div><span style="display:inline-block;width:20px;height:9px;background:#E24B4A;vertical-align:middle"></span> Тормоз (&lt;{d.get("cut_g", -0.25):.2f}g)</div>\n'
        '<div><span style="display:inline-block;width:20px;height:7px;background:#EF9F27;vertical-align:middle"></span> Сброс / лёгкий трейл</div>\n'
        '<div><span style="display:inline-block;width:20px;height:6px;background:#1D9E75;vertical-align:middle"></span> Газ</div>\n'
        '<div><span style="display:inline-block;width:20px;height:4px;background:#B4B2A9;vertical-align:middle"></span> Ровно</div>\n'
        '<div style="margin-top:6px"><svg width="20" height="12"><circle cx="10" cy="6" r="5" fill="none" stroke="#534AB7" stroke-width="2"/></svg> Проскальзывание задней</div>\n'
        '</div></div>\n'
    )

    # JS — raw string, никаких подстановок.
    js = r"""
<script>
(function() {
  var DATA = __DATA__;
  var SVG_NS = "http://www.w3.org/2000/svg";
  var size = [680, 560], pad = 60;
  var trailL = document.getElementById('trailLayer');
  var kart = document.getElementById('kart');
  var statusEl = document.getElementById('status');

  // Draw pavement
  var pavD = DATA.pts.map(function(p, i) {
    return (i === 0 ? 'M ' : 'L ') + p[0] + ' ' + p[1];
  }).join(' ') + ' Z';
  document.getElementById('pavement').setAttribute('d', pavD);

  // Start/finish tick
  var p0 = DATA.pts[0];
  var pN = DATA.pts[Math.min(5, DATA.pts.length - 1)];
  var tx = pN[0] - p0[0], ty = pN[1] - p0[1];
  var L = Math.hypot(tx, ty) || 1;
  var nx = -ty / L * 16, ny = tx / L * 16;
  var sf = document.getElementById('startFin');
  sf.setAttribute('x1', p0[0] + nx);
  sf.setAttribute('y1', p0[1] + ny);
  sf.setAttribute('x2', p0[0] - nx);
  sf.setAttribute('y2', p0[1] - ny);

  // Convert world coord → SVG. Reuses same formula as Python.
  function worldToSvg(x, y) {
    var w = DATA.bounds;
    var bx = w.maxx - w.minx, by = w.maxy - w.miny;
    var scale = Math.min((size[0] - 2 * pad) / bx, (size[1] - 2 * pad) / by);
    var sx = pad + (size[0] - 2 * pad - bx * scale) / 2 + (x - w.minx) * scale;
    var sy = pad + (size[1] - 2 * pad) - ((y - w.miny) * scale + (size[1] - 2 * pad - by * scale) / 2);
    return [sx, sy];
  }

  // Corners
  var cornerL = document.getElementById('cornerLayer');
  DATA.corners.forEach(function(c) {
    var p = worldToSvg(c.x, c.y);
    var t1 = document.createElementNS(SVG_NS, 'text');
    t1.setAttribute('x', p[0]); t1.setAttribute('y', p[1] - 14);
    t1.setAttribute('font-family', 'system-ui'); t1.setAttribute('font-size', '11');
    t1.setAttribute('fill', '#888780'); t1.setAttribute('text-anchor', 'middle');
    t1.textContent = c.name;
    cornerL.appendChild(t1);
    var t2 = document.createElementNS(SVG_NS, 'text');
    t2.setAttribute('x', p[0]); t2.setAttribute('y', p[1] + 24);
    t2.setAttribute('font-family', 'system-ui'); t2.setAttribute('font-size', '13');
    t2.setAttribute('font-weight', '700'); t2.setAttribute('fill', '#2C2C2A');
    t2.setAttribute('text-anchor', 'middle');
    t2.textContent = Math.round(c.speed);
    cornerL.appendChild(t2);
  });

  // Slips
  var slipL = document.getElementById('slipLayer');
  DATA.slips.forEach(function(s) {
    var p = worldToSvg(s.x, s.y);
    var r = 6 + Math.max(0, s.strength - 0.28) * 22;
    var c1 = document.createElementNS(SVG_NS, 'circle');
    c1.setAttribute('cx', p[0]); c1.setAttribute('cy', p[1]);
    c1.setAttribute('r', r); c1.setAttribute('fill', 'none');
    c1.setAttribute('stroke', '#534AB7'); c1.setAttribute('stroke-width', '2');
    slipL.appendChild(c1);
    var c2 = document.createElementNS(SVG_NS, 'circle');
    c2.setAttribute('cx', p[0]); c2.setAttribute('cy', p[1]);
    c2.setAttribute('r', '2.5'); c2.setAttribute('fill', '#534AB7');
    slipL.appendChild(c2);
  });

  var i = 0, playing = false, lastT = 0, drawn = 0;
  var totalSamples = DATA.pts.length;
  var dtMs = 1000 / DATA.fs;

  function stepTo(target) {
    if (target < 0) target = 0;
    if (target > totalSamples) target = totalSamples;
    if (target < drawn) {
      trailL.innerHTML = '';
      drawn = 0;
    }
    while (drawn < target) {
      var ph = DATA.phases[drawn];
      var end = drawn + 1;
      while (end < target && DATA.phases[end] === ph) end++;
      var hi = Math.min(end + 1, totalSamples);
      var dStr = '';
      for (var k = drawn; k < hi; k++) {
        var pt = DATA.pts[k];
        if (!pt) break;
        dStr += (k === drawn ? 'M ' : 'L ') + pt[0] + ' ' + pt[1] + ' ';
      }
      var col = DATA.phase_color[ph] || '#B4B2A9';
      var w = DATA.phase_width[ph] || 4;
      var path = document.createElementNS(SVG_NS, 'path');
      path.setAttribute('d', dStr);
      path.setAttribute('stroke', col);
      path.setAttribute('stroke-width', w);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      trailL.appendChild(path);
      drawn = end;
    }
    var p = DATA.pts[Math.max(0, Math.min(target, totalSamples - 1))];
    if (p && kart) {
      kart.setAttribute('cx', p[0]);
      kart.setAttribute('cy', p[1]);
    }
    var sec = target / DATA.fs;
    var cur_ph = DATA.phases[Math.max(0, Math.min(target, totalSamples - 1))] || '—';
    statusEl.textContent = 't=' + sec.toFixed(2) + 's · phase=' + cur_ph + ' · idx=' + target + '/' + totalSamples;
  }

  function tick(now) {
    if (!playing) return;
    var speed = parseFloat(document.getElementById('speedSel').value || '1');
    var dt = now - lastT;
    lastT = now;
    i += (dt / dtMs) * speed;
    if (i >= totalSamples) {
      playing = false;
      i = totalSamples;
    }
    stepTo(Math.floor(i));
    if (playing) requestAnimationFrame(tick);
  }

  document.getElementById('btnPlay').onclick = function() {
    if (!playing) {
      playing = true;
      lastT = performance.now();
      requestAnimationFrame(tick);
    }
  };
  document.getElementById('btnPause').onclick = function() { playing = false; };
  document.getElementById('btnReset').onclick = function() {
    playing = false; i = 0; drawn = 0;
    trailL.innerHTML = '';
    stepTo(0);
  };

  stepTo(0);
})();
</script>
</body></html>
"""
    js = js.replace('__DATA__', data_json)
    html = head + header + svg_block + side + js
    Path(out_html).write_text(html)
    print(f'wrote {out_html}  ({len(html)/1024:.0f} KB)')


def export_csv(d, out_imu, out_phases):
    import csv
    L = d['L']
    with open(out_imu, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'wall_ns', 'ax_g', 'ay_g', 'az_g', 'gx_rads', 'gy_rads', 'gz_rads',
                    'qw', 'qx', 'qy', 'qz'])
        for i in range(len(L)):
            w.writerow([round(d['t'][i], 3), int(L['ts_ns'][i]),
                        round(float(L['ax'][i]), 5), round(float(L['ay'][i]), 5), round(float(L['az'][i]), 5),
                        round(float(L['gx'][i]), 5), round(float(L['gy'][i]), 5), round(float(L['gz'][i]), 5),
                        round(float(L['qw'][i]), 6), round(float(L['qx'][i]), 6),
                        round(float(L['qy'][i]), 6), round(float(L['qz'][i]), 6)])
    print(f'wrote {out_imu} ({len(L)} rows)')

    with open(out_phases, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 'x_m', 'y_m', 'v_mps', 'v_kmh', 'a_long_g', 'a_long_smooth_g',
                    'a_lat_g', 'wz_rads', 'psi_rad', 'phase'])
        for i in range(len(L)):
            w.writerow([round(d['t'][i], 3), round(float(d['x'][i]), 3), round(float(d['y'][i]), 3),
                        round(float(d['v'][i]), 2), round(float(d['v'][i] * 3.6), 1),
                        round(float(d['a_long'][i]), 4), round(float(d['a_long_sm'][i]), 4),
                        round(float(d['a_lat'][i]), 4), round(float(d['wz'][i]), 4),
                        round(float(d['psi'][i]), 4), d['phases'][i]])
    print(f'wrote {out_phases}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='output dir (default /tmp/lap{N}_report)')
    ap.add_argument('--lap', type=int, default=31)
    args = ap.parse_args()
    N = args.lap
    out = Path(args.out) if args.out else Path(f'/tmp/lap{N}_report')
    out.mkdir(parents=True, exist_ok=True)

    d = process_lap(N)
    prefix = f'lap{N}'
    svg, bounds, pts = render_static_svg(d, out / f'{prefix}.svg', out / f'{prefix}.png')
    render_animated_html(d, svg, bounds, pts, out / f'{prefix}.html')
    export_csv(d, out / f'{prefix}_imu.csv', out / f'{prefix}_phases.csv')

    meta = {
        'lap_number': N,
        'raceman_time_s': d['raceman_time_s'],
        'theta_deg': d['theta_deg'],
        'corr': d['corr'],
        'closure_m': d['closure_m'],
        'lap_len_m': d['lap_len_m'],
        'phase_counts': {k: d['phases'].count(k) for k in ['brake', 'lift', 'accel', 'cruise']},
        'corners': d['corners'],
        'slips': d['slips'],
    }
    (out / f'{prefix}_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f'wrote {out / f"{prefix}_meta.json"}')
    print()
    print(f'=== {prefix} pack ready: {out} ===')


if __name__ == '__main__':
    main()
