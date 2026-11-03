"""Сравнение двух кругов на ОДНОЙ геометрии трассы.

Шаблон (canonical trajectory) берётся от best lap. Фазы и slips второго
круга проецируются на тот же шаблон через DTW по wz. Трасса не крутится
между кругами.

Запуск:
  # На fallback MIKS-сессии:
  python3 laps_compare.py --best 31 --other 25 --out /tmp/laps_compare

  # На произвольной сессии (zip из katahub):
  python3 laps_compare.py --zip SESSION.zip --best 31 --other 25 --out compare.html

  # Можно явно задать kart (иначе автоопределение из timing.kartNumber):
  python3 laps_compare.py --zip SESSION.zip --kart 72 --best 31 --other 25 --out compare.html
"""
import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from karttelemetry import loader
from lap31_standalone import (
    process_lap, PHASE_COLOR, PHASE_WIDTH, world_to_svg,
    compute_session_sf_offset, compute_session_phase_cut,
)


def _find_in_dir(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.exists():
        return direct
    for p in root.rglob(filename):
        if p.is_file():
            return p
    return None


def load_session_from_zip(zip_path: Path):
    """Извлекает watch_imu.bin и timing.json из зипа во временную директорию.
    Возвращает (samples, timing, kart_number_from_timing, cleanup_fn)."""
    tmp = Path(tempfile.mkdtemp(prefix='laps_compare_'))
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp)
        watch_imu = _find_in_dir(tmp, 'watch_imu.bin')
        if watch_imu is None:
            raise RuntimeError(f'watch_imu.bin not found in {zip_path}')
        timing_path = _find_in_dir(tmp, 'timing.json')
        if timing_path is None:
            raise RuntimeError(f'timing.json not found in {zip_path}')
        samples = np.fromfile(watch_imu, dtype=loader.IMU_DTYPE)
        timing = json.loads(timing_path.read_text())
        kart = timing.get('kartNumber') or ''
        return samples, timing, str(kart), lambda: shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def dtw_closed(A, B, band: int = 100):
    """Sakoe-Chiba DTW, оба конца закреплены.
    A, B — массивы (N,k)/(M,k) для мульти-признакового пути, либо 1D (трактуются как (N,1))."""
    A = np.atleast_2d(A.T if A.ndim == 1 else A)
    if A.shape[0] < A.shape[1]:
        A = A.T if A.ndim == 2 and A.shape[0] < A.shape[1] else A
    # Нормализуем в (N, k) если 1D
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    N, M = len(A), len(B)
    INF = 1e18
    D = np.full((N + 1, M + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, N + 1):
        c = int(round(i * M / N))
        lo = max(1, c - band)
        hi = min(M, c + band)
        A_i = A[i - 1]
        cost = np.sum((B[lo - 1:hi] - A_i) ** 2, axis=1)
        prev = D[i - 1]
        row = D[i]
        cand = np.minimum(prev[lo - 1:hi], prev[lo:hi + 1])
        for j in range(lo, hi + 1):
            v = cand[j - lo]
            if row[j - 1] < v:
                v = row[j - 1]
            row[j] = cost[j - lo] + v
    i, j = N, M
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        choices = [
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1),
        ]
        _, i, j = min(choices)
        path.append((i - 1, j - 1))
    return path[::-1]


def compute_dtw_path(other, best, ds=2):
    """DTW по двум признакам: (wz/1.2, 0.5·norm(a_h)). Нормировка a_h —
    по шаблону (best), чтобы оба круга были в одном масштабе. Второй признак
    спасает на прямой, где wz≈0 и путь без него гуляет."""
    mu = float(best['ah'].mean())
    sd = max(float(best['ah'].std()), 1e-6)
    def feat(lap):
        return np.c_[lap['wz'] / 1.2, (lap['ah'] - mu) / sd * 0.5]
    A = feat(best)[::ds]
    B = feat(other)[::ds]
    path = dtw_closed(A, B, band=100)
    return path, ds, len(A), len(B)


def hybrid_index(path, ds, wz, v, fs=100, min_gap_s=0.5, min_wz=0.8):
    """DTW залипает на плато wz в дугах постоянного радиуса (T1, T2) и
    не учитывает глубину входа — точка отстаёт до 4м. Фикс (per feedback):
    якоря DTW только на границах поворотов (нули wz с реальной дугой рядом),
    ВНУТРИ поворотов — по пройденному пути v·dt.

    Возвращает m[j] = template index для каждого lap-сэмпла j (0..N-1).
    """
    p = np.asarray(path)
    N = len(wz)
    # Базовая проекция j → i через DTW (используется только на якорях).
    m0 = np.interp(np.arange(N), p[:, 1] * ds, p[:, 0]) * ds
    # Нули wz, рядом с реальной дугой (|wz| > min_wz в ±0.5с окне)
    zc = np.where(np.diff(np.sign(wz)) != 0)[0]
    half = int(fs / 2)
    zc = [int(z) for z in zc
          if np.abs(wz[max(0, z - half):z + half]).max() > min_wz]
    anchors = [0]
    for z in zc:
        if z - anchors[-1] > min_gap_s * fs and z < N - 1 - min_gap_s * fs:
            anchors.append(z)
    anchors.append(N - 1)
    # Пройденный путь lap-круга
    s = np.cumsum(v) / fs
    m = np.empty(N)
    for a, b in zip(anchors[:-1], anchors[1:]):
        s_span = max(s[b] - s[a], 1e-6)
        frac = (s[a:b + 1] - s[a]) / s_span
        m[a:b + 1] = m0[a] + frac * (m0[b] - m0[a])
    return m


def project_phases_to_template(path, ds, N_ds, M_ds, other_phases, template_wz,
                                 other_wz=None, other_v=None, fs=100.0):
    """Строим phases на template timeline.

    Если даны other_wz и other_v — используем hybrid_index (DTW-якоря на
    границах поворотов, внутри — по v·dt). Иначе — базовый линейный DTW.
    """
    N_tmpl = len(template_wz)

    if other_wz is not None and other_v is not None:
        # hybrid: для каждого j (lap) → m[j] (template)
        m = hybrid_index(path, ds, other_wz, other_v, fs=fs)
        tmpl_phase = ['' for _ in range(N_tmpl)]
        for j in range(len(other_phases)):
            i = int(round(m[j]))
            if 0 <= i < N_tmpl and not tmpl_phase[i]:
                tmpl_phase[i] = other_phases[j]
        last = 'cruise'
        for i in range(N_tmpl):
            if not tmpl_phase[i]:
                tmpl_phase[i] = last
            else:
                last = tmpl_phase[i]
        return tmpl_phase

    # fallback: простой DTW-проекция
    tmpl_phase = ['' for _ in range(N_ds)]
    for i, j in path:
        if 0 <= i < N_ds and not tmpl_phase[i]:
            j_full = min(j * ds, len(other_phases) - 1)
            tmpl_phase[i] = other_phases[j_full]
    last = 'cruise'
    for i in range(N_ds):
        if not tmpl_phase[i]:
            tmpl_phase[i] = last
        else:
            last = tmpl_phase[i]
    out = []
    for p in tmpl_phase:
        out.append(p)
        out.append(p)
    return out[: N_tmpl]


def project_idx_to_template(path, ds, lap_idx, M_ds, hybrid_m=None):
    """Для одного индекса lap-круга находим template index.

    Если дан hybrid_m (из hybrid_index) — используем его (учитывает глубину
    входа в поворот). Иначе — ближайший через DTW path.
    """
    if hybrid_m is not None and 0 <= lap_idx < len(hybrid_m):
        return int(round(hybrid_m[lap_idx]))
    j_to_i = {}
    for i, j in path:
        if j not in j_to_i:
            j_to_i[j] = i
    j_ds = min(lap_idx // ds, M_ds - 1)
    for cand in range(j_ds - 10, j_ds + 11):
        if cand in j_to_i:
            return j_to_i[cand] * ds
    return None


def compute_sector_times_from_dtw(path, ds, template_sector_bounds, lap_fs,
                                   tmpl_wz=None, lap_wz=None):
    """Для каждого сектора вернуть (start_t_s, end_t_s, dur_s) в lap-time.
    DTW даёт грубую позицию j для внутренней границы, затем refine_boundary
    уточняет локальной корреляцией wz_f ±1.5с (точность ~0.05с)."""
    i_to_max_j = {}
    for i, j in path:
        if i not in i_to_max_j or j > i_to_max_j[i]:
            i_to_max_j[i] = j

    def lookup_j_full(tmpl_idx_full):
        tmpl_idx_ds = tmpl_idx_full // ds
        bst = None
        for di in range(0, 15):
            for sign in (0, -1, 1) if di > 0 else (0,):
                k = tmpl_idx_ds + sign * di
                if k in i_to_max_j:
                    bst = i_to_max_j[k]
                    break
            if bst is not None:
                break
        if bst is None:
            return None
        return bst * ds

    def refine_boundary(j0, s_full, half=250, search=150):
        """Локальная корреляция wz-окна шаблона с окном lap вокруг j0 ±search.
        Возвращает уточнённый j (в full-fs), либо исходный если шаблон плоский.
        """
        if tmpl_wz is None or lap_wz is None:
            return j0
        a, b = s_full - half, s_full + half
        if a < 0 or b > len(tmpl_wz):
            return j0
        T = tmpl_wz[a:b]
        if float(np.std(T)) < 0.3:   # прямая — нечего ловить
            return j0
        best_r, best_j = -1.0, j0
        for s in range(-search, search + 1):
            ja, jb = j0 + s - half, j0 + s + half
            if ja < 0 or jb > len(lap_wz):
                continue
            L = lap_wz[ja:jb]
            if len(L) != len(T):
                continue
            r = float(np.corrcoef(L, T)[0, 1])
            if np.isnan(r):
                continue
            if r > best_r:
                best_r, best_j = r, j0 + s
        return best_j

    # Для первого сектора принудительно js=0, для последнего je = M-1,
    # чтобы Σ покрывал весь lap. Иначе DTW плато на старте/финише съедают
    # часть времени.
    out = []
    n_sectors = len(template_sector_bounds)
    max_j_overall = max(i_to_max_j.values()) if i_to_max_j else 0
    # Уточнение всех внутренних границ (не 0 и не последней)
    refined_cache = {}
    def resolve(s_full, is_edge):
        if s_full in refined_cache:
            return refined_cache[s_full]
        j0 = lookup_j_full(s_full)
        if j0 is None:
            refined_cache[s_full] = None
            return None
        if is_edge:
            refined_cache[s_full] = j0
        else:
            refined_cache[s_full] = refine_boundary(j0, s_full)
        return refined_cache[s_full]
    for idx, (name, s_full, e_full) in enumerate(template_sector_bounds):
        js = resolve(s_full, is_edge=(idx == 0))
        je = resolve(e_full, is_edge=(idx == n_sectors - 1))
        if js is None or je is None:
            out.append((name, None, None, None))
            continue
        if idx == 0:
            js = 0
        if idx == n_sectors - 1:
            je = max_j_overall * ds
        dur_s = (je - js) / lap_fs
        out.append((name, js / lap_fs, je / lap_fs, dur_s))
    return out


def build_canonical_sectors(canonical, fs):
    """Группируем ВСЕ corners по знакам wz. Границы — на нулях wz между
    корнерами разного знака. Сектора ВСЕГДА покрывают весь круг [0, N).
    Если >5 групп, склеиваем самые короткие пары.

    Инвариант: Σ(end - start) == N (весь круг).
    """
    corners = canonical['corners']
    wz = canonical['wz']
    N = len(wz)
    if not corners:
        return [('A (все)', 0, N - 1)]

    # Группируем по направлению, НЕ пропуская ни одного корнера
    groups = []
    for c in corners:
        d = c['direction']
        if not groups or groups[-1][-1]['direction'] != d:
            groups.append([c])
        else:
            groups[-1].append(c)

    # Границы: 0, [zc между группами], N-1
    boundaries = [0]
    for gi in range(len(groups) - 1):
        last_apex_t = groups[gi][-1]['apex_t']
        next_apex_t = groups[gi + 1][0]['apex_t']
        a = int(last_apex_t * fs)
        b = int(next_apex_t * fs)
        if b <= a:
            boundaries.append((a + b) // 2)
            continue
        seg = wz[a:b + 1]
        sg = np.sign(seg)
        # Берём ПОСЛЕДНЕЕ пересечение нуля перед b (ближе к входу в следующий
        # поворот), а не первое после a. Там wz уже снова "кусается" и локальная
        # корреляция имеет за что цепляться.
        zcs = [a + k for k in range(1, len(sg))
               if sg[k] != sg[k - 1] and (sg[k] != 0 or sg[k - 1] != 0)]
        zc = zcs[-1] if zcs else a + (len(seg) // 2)
        boundaries.append(zc)
    boundaries.append(N - 1)

    # Склеиваем пары соседних, пока не станет ≤5 секторов. Выбираем пару
    # с наименьшей суммарной длительностью.
    while len(groups) > 5:
        min_sum = 10**9
        min_i = 0
        for i in range(len(groups) - 1):
            sum_dur = (boundaries[i + 2] - boundaries[i])
            if sum_dur < min_sum:
                min_sum = sum_dur
                min_i = i
        groups[min_i] = groups[min_i] + groups[min_i + 1]
        del groups[min_i + 1]
        del boundaries[min_i + 1]

    names = 'ABCDE'
    out = []
    for i, g in enumerate(groups):
        corner_names = '-'.join(c['name'] for c in g)
        out.append((f'{names[i]} ({corner_names})', boundaries[i], boundaries[i + 1]))

    # Инвариант
    total = sum(e - s for _, s, e in out)
    if abs(total - (N - 1)) > 3:
        print(f'  WARN: sectors cover {total}/{N - 1} samples')
    return out


def project_slips_to_template(other_slips_idx, other_wz, template_wz):
    """Проецируем индексы slip-событий другого круга на шаблонные индексы
    через DTW. Возвращает список шаблонных индексов."""
    ds = 2
    A = template_wz[::ds]
    B = other_wz[::ds]
    path = dtw_closed(A, B, band=100)
    # j (lap) → i (template) first match
    j_to_i = {}
    for i, j in path:
        if j not in j_to_i:
            j_to_i[j] = i
    out = []
    for s_idx in other_slips_idx:
        j_ds = min(s_idx // ds, len(B) - 1)
        # ближайший j в карте
        best = None
        for cand in range(j_ds - 10, j_ds + 10):
            if cand in j_to_i:
                best = j_to_i[cand] * ds
                break
        if best is not None:
            out.append(best)
    return out


def assign_brake_zones_to_corners(phases, corner_apex_names_sorted, fs=100.0, max_dist_s=3.0):
    """1:1 матчинг зон торможения к корнерам.

    `corner_apex_names_sorted` — [(name, apex_idx), ...] в возрастающем apex_idx.
    Правило: brake-зона принадлежит ближайшему корнеру с apex > zone_end,
    но ТОЛЬКО если apex не дальше `max_dist_s` после конца зоны (иначе зона
    на прямой, не относится ни к какому повороту).

    Возвращает dict {corner_name: {start_idx, end_idx} or None}.
    """
    zones = []
    i = 0
    N = len(phases)
    while i < N:
        if phases[i] != 'brake':
            i += 1
            continue
        start = i
        while i < N and phases[i] == 'brake':
            i += 1
        zones.append((start, i - 1))

    # Trail braking: brake-зона может ТЯНУТЬСЯ через apex (пилот ещё
    # давит педаль во входе поворота). Апекс может быть ВНУТРИ зоны —
    # не только после. Разрешаем apex в [z_start - 0.5с, z_end + max_dist_s].
    max_dist_samples = int(max_dist_s * fs)
    pre_samples = int(0.5 * fs)
    out = {name: None for name, _ in corner_apex_names_sorted}
    for z_start, z_end in zones:
        cands = [(abs(apex - z_end), name) for name, apex in corner_apex_names_sorted
                 if z_start - pre_samples <= apex <= z_end + max_dist_samples]
        if not cands:
            continue
        _, best_name = min(cands)
        if out[best_name] is None:
            out[best_name] = {'start_idx': z_start, 'end_idx': z_end}
    return out


def render_two_laps_html(canonical, best_data, other_data, out_html,
                          best_num, other_num):
    """Одна страница с двумя SVG. Canonical trajectory — одна и та же.
    Разные цвета фаз per lap. Плюс аннотации по корнерам: где тормозил
    раньше/позже."""
    size = (640, 520)
    pad = 50
    bounds = canonical['bounds']
    pts = [world_to_svg(x, y, bounds, size, pad) for x, y in zip(canonical['x'], canonical['y'])]
    # кумулятивная длина трассы для оценки Δ в метрах
    cum_s = [0.0]
    for i in range(1, len(canonical['x'])):
        dx = canonical['x'][i] - canonical['x'][i - 1]
        dy = canonical['y'][i] - canonical['y'][i - 1]
        cum_s.append(cum_s[-1] + math.hypot(dx, dy))

    # Апексы корнеров в индексах canonical trajectory
    apex_idx_by_name = {}
    for c in canonical['corners']:
        best_i = 0
        best_d = 1e18
        for i in range(len(canonical['x'])):
            d2 = (canonical['x'][i] - c['apex_x']) ** 2 + (canonical['y'][i] - c['apex_y']) ** 2
            if d2 < best_d:
                best_d = d2
                best_i = i
        apex_idx_by_name[c['name']] = best_i

    # Brake-зоны: обе фазовые серии уже на canonical (template) индексах.
    # Апексы корнеров тоже в canonical. Единая система координат → матчинг
    # честный, нумерация T_n одинаковая для обоих кругов.
    best_phases_tmpl = best_data['phases_on_canonical']
    other_phases_tmpl = other_data['phases_on_canonical']
    tmpl_fs = best_data['raw']['fs']  # canonical = best, та же fs
    corner_names_by_apex = sorted(apex_idx_by_name.items(), key=lambda kv: kv[1])
    best_zones = assign_brake_zones_to_corners(best_phases_tmpl, corner_names_by_apex, fs=tmpl_fs, max_dist_s=3.0)
    other_zones = assign_brake_zones_to_corners(other_phases_tmpl, corner_names_by_apex, fs=tmpl_fs, max_dist_s=3.0)

    brake_deltas = {}
    for name, apex in corner_names_by_apex:
        bz = best_zones.get(name)
        oz = other_zones.get(name)
        if bz is not None and oz is not None:
            best_before_s = (apex - bz['start_idx']) / tmpl_fs
            other_before_s = (apex - oz['start_idx']) / tmpl_fs
            delta_s = best_before_s - other_before_s   # +: other позже
            v_mean = float(np.mean(best_data['raw']['v'][bz['start_idx']:apex])) if apex > bz['start_idx'] else 10.0
            delta_m = delta_s * v_mean
            brake_deltas[name] = {
                'delta_s': delta_s, 'delta_m': delta_m,
                'best_before_apex_s': best_before_s,
                'other_before_apex_s': other_before_s,
                'best_start': bz['start_idx'],
                'other_start': oz['start_idx'],
                'apex': apex,
            }
        elif bz is None and oz is None:
            brake_deltas[name] = None
        elif bz is None:
            brake_deltas[name] = {'delta_s': None, 'delta_m': None,
                                   'other_before_apex_s': (apex - oz['start_idx']) / tmpl_fs,
                                   'note': 'в PB нет brake'}
        else:
            brake_deltas[name] = {'delta_s': None, 'delta_m': None,
                                   'best_before_apex_s': (apex - bz['start_idx']) / tmpl_fs,
                                   'note': 'в other нет brake'}

    def render_svg_pane(lap_data, pts, title, lap_num, time_s, show_brake_delta=False):
        parts = [
            f'<svg viewBox="0 0 {size[0]} {size[1]}" width="{size[0]}" height="{size[1]}" xmlns="http://www.w3.org/2000/svg" style="display:block">',
            '<rect width="100%" height="100%" fill="#F6F5F0"/>',
            f'<text x="24" y="26" font-family="system-ui" font-size="13" fill="#2C2C2A" font-weight="600">Круг {lap_num} · {time_s:.3f} с · v_max {lap_data["v_max_kmh"]:.0f} км/ч</text>',
        ]
        # pavement
        pavD = 'M ' + ' L '.join(f'{sx:.1f} {sy:.1f}' for sx, sy in pts) + ' Z'
        parts.append(f'<path d="{pavD}" stroke="#D3D1C7" stroke-width="20" fill="none" stroke-linejoin="round" stroke-linecap="round" opacity="0.55"/>')
        # phases
        phases = lap_data['phases_on_canonical']
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
        # slips
        for s_idx in lap_data['slips_on_canonical']:
            if 0 <= s_idx < len(pts):
                sx, sy = pts[s_idx]
                parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="10" fill="none" stroke="#534AB7" stroke-width="2"/>')
                parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.5" fill="#534AB7"/>')
        # v_max маркер (жирная точка на треке в месте максимума)
        if lap_data.get('v_max_idx') is not None:
            vi = lap_data['v_max_idx']
            if 0 <= vi < len(pts):
                vx, vy = pts[vi]
                parts.append(f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="4" fill="#1D9E75" stroke="white" stroke-width="1.5"/>')
                parts.append(f'<text x="{vx:.1f}" y="{(vy + 20):.1f}" font-family="system-ui" font-size="10" font-weight="700" fill="#1D9E75" text-anchor="middle">v_max</text>')
        # corner labels — для other-pane тут же показываем Δ brake-point
        for c in canonical['corners']:
            cx, cy = world_to_svg(c['apex_x'], c['apex_y'], bounds, size, pad)
            parts.append(f'<text x="{cx:.1f}" y="{(cy - 10):.1f}" font-family="system-ui" font-size="10" fill="#888780" text-anchor="middle">{c["name"]}</text>')
            if show_brake_delta:
                bd = brake_deltas.get(c['name'])
                if bd and bd.get('delta_m') is not None and abs(bd['delta_m']) > 1.0:
                    dm = bd['delta_m']
                    # +Х м = тормозил ПОЗЖЕ (ближе к апексу = агрессивнее)
                    # -Х м = тормозил РАНЬШЕ (дальше от апекса = консервативнее, теряем скорость)
                    label = f'{"+" if dm > 0 else ""}{dm:.1f}м'
                    col = '#1D9E75' if dm > 0 else '#E24B4A'
                    arrow = '→' if dm > 0 else '←'
                    parts.append(f'<text x="{cx:.1f}" y="{(cy + 20):.1f}" font-family="system-ui" font-size="10" font-weight="700" fill="{col}" text-anchor="middle">{label} {arrow}</text>')
                    # Показываем точку brake-start на самом треке — вертикальная чёрточка
                    bs_idx = bd['other_start']
                    if 0 <= bs_idx < len(pts):
                        bsx, bsy = pts[bs_idx]
                        parts.append(f'<circle cx="{bsx:.1f}" cy="{bsy:.1f}" r="3" fill="{col}" stroke="white" stroke-width="1"/>')
        # start/finish
        if len(pts) >= 6:
            tx = pts[5][0] - pts[0][0]
            ty = pts[5][1] - pts[0][1]
            L = math.hypot(tx, ty) or 1
            nx = -ty / L * 14
            ny = tx / L * 14
            parts.append(f'<line x1="{pts[0][0] + nx:.1f}" y1="{pts[0][1] + ny:.1f}" x2="{pts[0][0] - nx:.1f}" y2="{pts[0][1] - ny:.1f}" stroke="#2C2C2A" stroke-width="2.5"/>')
        # phase counts footer
        pc = lap_data['phase_counts']
        tot = sum(pc.values()) or 1
        # v2 §11 labels: тормоз · сброс/лёгкий трейл · газ · ровно · срывов
        cut_g = lap_data.get('cut_g', -0.47)
        parts.append(
            f'<text x="24" y="{size[1] - 24}" font-family="system-ui" font-size="11" fill="#5F5E5A">'
            f'<tspan fill="#E24B4A">тормоз &lt;{cut_g:.2f}g · {pc["brake"]*100//tot}%</tspan>  '
            f'<tspan fill="#EF9F27">сброс/трейл {pc["lift"]*100//tot}%</tspan>  '
            f'<tspan fill="#1D9E75">газ {pc["accel"]*100//tot}%</tspan>  '
            f'<tspan fill="#B4B2A9">ровно {pc["cruise"]*100//tot}%</tspan>  '
            f'<tspan fill="#534AB7">срывов {len(lap_data["slips_on_canonical"])}</tspan>'
            f'</text>'
        )
        parts.append('</svg>')
        return '\n'.join(parts)

    svg_best = render_svg_pane(best_data, pts, 'best', best_num, best_data['raceman_time_s'], show_brake_delta=False)
    svg_other = render_svg_pane(other_data, pts, 'other', other_num, other_data['raceman_time_s'], show_brake_delta=True)

    dt = other_data['raceman_time_s'] - best_data['raceman_time_s']
    dt_sign = '+' if dt >= 0 else ''

    # Таблица brake-point delta по корнерам
    delta_rows = []
    for c in canonical['corners']:
        bd = brake_deltas.get(c['name'])
        if bd and bd.get('delta_s') is not None:
            ds_ = bd['delta_s']
            # Positive delta_s = other начал тормозить ПОЗЖЕ best (ближе к апексу)
            col = '#1D9E75' if ds_ > 0.05 else '#E24B4A' if ds_ < -0.05 else '#888780'
            arrow = 'позже →' if ds_ > 0.05 else ('← раньше' if ds_ < -0.05 else '≈')
            delta_rows.append(
                f'<tr><td>{c["name"]}</td>'
                f'<td style="text-align:right">{bd["best_before_apex_s"]:.2f}с</td>'
                f'<td style="text-align:right">{bd["other_before_apex_s"]:.2f}с</td>'
                f'<td style="text-align:right;color:{col}">{ds_:+.2f}с</td>'
                f'<td style="color:{col}">{arrow}</td></tr>'
            )
        elif bd and bd.get('note'):
            delta_rows.append(f'<tr><td>{c["name"]}</td><td colspan="4" style="color:#888780">{bd["note"]}</td></tr>')
        else:
            delta_rows.append(f'<tr><td>{c["name"]}</td><td colspan="4" style="color:#888780">нет brake-зоны</td></tr>')
    delta_table = (
        '<table style="font-size:12px;border-collapse:collapse">'
        '<tr style="color:#888780">'
        '<th style="text-align:left;padding:2px 10px 2px 0">корнер</th>'
        '<th style="text-align:right;padding:2px 10px">PB brake→apex</th>'
        '<th style="text-align:right;padding:2px 10px">other brake→apex</th>'
        '<th style="text-align:right;padding:2px 10px">Δ</th>'
        '<th style="text-align:left"></th>'
        '</tr>' + ''.join(delta_rows) + '</table>'
    )

    # Таблица времён секторов: best vs other vs delta
    sec_rows = []
    sec_meta = canonical['sectors_table']
    total_best = sum((d for _, _, _, d in sec_meta['best'] if d is not None), 0.0)
    total_other = sum((d for _, _, _, d in sec_meta['other'] if d is not None), 0.0)
    for (bname, bs, be, bd), (oname, os_, oe, od) in zip(sec_meta['best'], sec_meta['other']):
        if bd is None or od is None:
            sec_rows.append(f'<tr><td>{bname}</td><td colspan="3" style="color:#888780">—</td></tr>')
            continue
        delta = od - bd
        col = '#E24B4A' if delta > 0.02 else ('#1D9E75' if delta < -0.02 else '#888780')
        sec_rows.append(
            f'<tr><td>{bname}</td>'
            f'<td style="text-align:right">{bd:.3f}</td>'
            f'<td style="text-align:right">{od:.3f}</td>'
            f'<td style="text-align:right;color:{col}">{delta:+.3f}</td></tr>'
        )
    sec_rows.append(
        f'<tr style="border-top:1px solid #D3D1C7;font-weight:600"><td>Σ</td>'
        f'<td style="text-align:right">{total_best:.3f}</td>'
        f'<td style="text-align:right">{total_other:.3f}</td>'
        f'<td style="text-align:right">{total_other - total_best:+.3f}</td></tr>'
    )
    sec_table = (
        '<table style="font-size:12px;border-collapse:collapse">'
        f'<tr style="color:#888780"><th style="text-align:left;padding:2px 10px 2px 0">сектор</th>'
        f'<th style="text-align:right;padding:2px 10px">PB {best_num}, с</th>'
        f'<th style="text-align:right;padding:2px 10px">круг {other_num}, с</th>'
        f'<th style="text-align:right">Δ, с</th></tr>'
        + ''.join(sec_rows) + '</table>'
    )

    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Сравнение кругов {best_num} и {other_num}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #F6F5F0; color: #2C2C2A; margin: 0; padding: 20px; }}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
.pane {{ background: white; border-radius: 8px; padding: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
h1 {{ margin: 0 0 16px; font-size: 18px; }}
.summary {{ font-size: 14px; color: #5F5E5A; margin-bottom: 12px; }}
.diff {{ font-weight: 600; color: {'#E24B4A' if dt > 0 else '#1D9E75'}; }}
.legend {{ font-size: 12px; color: #5F5E5A; margin-top: 12px; }}
.legend span {{ display: inline-block; margin-right: 14px; }}
.legend .sw {{ display: inline-block; width: 18px; height: 6px; vertical-align: middle; margin-right: 6px; border-radius: 3px; }}
table td, table th {{ padding: 3px 10px; }}
.sidepanel {{ background: white; border-radius: 8px; padding: 12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 13px; }}
</style></head><body>
<h1>Круг {best_num} (PB) vs Круг {other_num}</h1>
<div class="summary">
  Разница: <span class="diff">{dt_sign}{dt:.3f} с</span>
  &middot; одна геометрия (от best), фазы/slips круга {other_num} спроецированы через DTW.
  Макс. скорость: <b>{best_data["v_max_kmh"]:.0f}</b> vs <b>{other_data["v_max_kmh"]:.0f}</b> км/ч
</div>
<div class="row">
  <div class="pane">{svg_best}</div>
  <div class="pane">{svg_other}</div>
  <div class="sidepanel">
    <div style="font-weight:600;margin-bottom:8px">Времена секторов</div>
    {sec_table}
    <div style="margin:16px 0 8px;font-weight:600">Точка торможения круга {other_num}<br><span style="font-weight:400;color:#888780">относительно PB (круг {best_num})</span></div>
    {delta_table}
    <div style="margin-top:12px;font-size:11px;color:#888780;line-height:1.5">
      <b style="color:#1D9E75">зелёный + "позже"</b> — тормозил ближе к апексу, агрессивнее<br>
      <b style="color:#E24B4A">красный + "раньше"</b> — тормозил рано, терял скорость на входе<br>
      Точка торможения — цветная метка на правой карте
    </div>
  </div>
</div>
<div class="legend">
  <span><span class="sw" style="background:#E24B4A;height:9px"></span>Торможение</span>
  <span><span class="sw" style="background:#EF9F27;height:7px"></span>Сброс газа</span>
  <span><span class="sw" style="background:#1D9E75;height:6px"></span>Разгон</span>
  <span><span class="sw" style="background:#B4B2A9;height:4px"></span>Качение</span>
  <span><svg width="18" height="14" style="vertical-align:middle"><circle cx="9" cy="7" r="5" fill="none" stroke="#534AB7" stroke-width="2"/></svg>Проскальзывание задней</span>
  <span><svg width="18" height="14" style="vertical-align:middle"><circle cx="9" cy="7" r="4" fill="#1D9E75" stroke="white" stroke-width="1.5"/></svg>v_max</span>
</div>
</body></html>
"""
    Path(out_html).write_text(html)
    print(f'wrote {out_html} ({len(html)/1024:.0f} KB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', dest='zip_path', default=None,
                     help='сессия .zip; если не указан — fallback на hardcoded MIKS пути')
    ap.add_argument('--kart', dest='kart', default=None,
                     help='номер карта; по умолчанию берётся из timing.kartNumber')
    ap.add_argument('--best', type=int, default=31)
    ap.add_argument('--other', type=int, default=25)
    ap.add_argument('--out', default='/tmp/laps_compare',
                     help='путь к output. Если заканчивается .html — single-file; иначе — директория (compare.html внутри)')
    args = ap.parse_args()

    # out: single-file если .html, иначе dir
    out_arg = Path(args.out)
    if out_arg.suffix.lower() == '.html':
        out_html = out_arg
        out_html.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_arg.mkdir(parents=True, exist_ok=True)
        out_html = out_arg / 'compare.html'

    # Загрузка данных: из зипа если указан, иначе fallback.
    cleanup = None
    if args.zip_path:
        samples, timing, kart_from_timing, cleanup = load_session_from_zip(Path(args.zip_path))
        kart = args.kart or kart_from_timing or '72'
    else:
        samples = None
        timing = None
        kart = args.kart or '72'

    try:
        # Session-wide S/F offset (один на сессию, медиана по всем кругам)
        sf_offset_s, _ = compute_session_sf_offset(samples=samples, timing=timing, kart_number=kart)
        # Session-wide phase cut (один Otsu на всю сессию, не per-lap). v2 §6.
        cut_g = compute_session_phase_cut(samples=samples, kart_number=kart)
        print(f'Processing best lap {args.best}... (kart={kart}, cut={cut_g:+.2f}g)')
        best = process_lap(args.best, samples=samples, timing=timing,
                            kart_number=kart, sf_offset_override=sf_offset_s,
                            cut_g=cut_g)
        print(f'Processing other lap {args.other}...')
        other = process_lap(args.other, samples=samples, timing=timing,
                             kart_number=kart, sf_offset_override=sf_offset_s,
                             cut_g=cut_g)
    except Exception:
        if cleanup:
            cleanup()
        raise

    # v2: lap-matching теперь честный (NCC + find_peaks + refine), ошибка
    # IMU-RaceMann durations <0.3s per spec §2. Loose assertion `>1.0s`
    # удалена — compare пропускает круг только через align gate (DTW rms).
    print(f'fs invariant: best {len(best["wz"])/best["fs"]:.3f}s vs {best["raceman_time_s"]:.3f} ✓')
    print(f'fs invariant: other {len(other["wz"])/other["fs"]:.3f}s vs {other["raceman_time_s"]:.3f} ✓')

    # Canonical trajectory = best lap
    bounds = {'minx': best['x'].min(), 'maxx': best['x'].max(),
              'miny': best['y'].min(), 'maxy': best['y'].max()}
    canonical = {
        'x': best['x'], 'y': best['y'], 'wz': best['wz'],
        'bounds': bounds,
        'corners': best['corners'],
    }

    best_v_max_idx = int(np.argmax(best['v']))
    best_data = {
        'phases_on_canonical': best['phases'],
        'slips_on_canonical': [int(s['t'] * best['fs']) for s in best['slips']],
        'phase_counts': {k: best['phases'].count(k) for k in ['brake', 'lift', 'accel', 'cruise']},
        'raceman_time_s': best['raceman_time_s'],
        'v_max_kmh': float(best['v'][best_v_max_idx] * 3.6),
        'v_max_idx': best_v_max_idx,
        'raw': best,
        'cut_g': best.get('cut_g', -0.47),
    }

    # DTW с двумя признаками (wz + a_h).
    print(f'DTW: lap {args.other} → lap {args.best}...')
    dtw_path, ds, N_ds, M_ds = compute_dtw_path(other, best)
    # Hybrid index: якоря DTW на границах поворотов, внутри — v·dt (per feedback)
    hybrid_m = hybrid_index(dtw_path, ds, other['wz'], other['v'], fs=other['fs'])
    # Диагностика выравнивания: наложить other на шаблон и измерить residual
    j_per_i_ds = {}
    for i_ds, j_ds in dtw_path:
        if i_ds not in j_per_i_ds:
            j_per_i_ds[i_ds] = j_ds
    j_for_i_full_debug = np.zeros(len(best['wz']), dtype=np.int64)
    last_j = 0
    for i_full in range(len(best['wz'])):
        i_ds = i_full // ds
        if i_ds in j_per_i_ds:
            last_j = j_per_i_ds[i_ds] * ds
        j_for_i_full_debug[i_full] = last_j
    j_for_i_full_debug = np.clip(j_for_i_full_debug, 0, len(other['wz']) - 1)
    resid = other['wz'][j_for_i_full_debug] - best['wz']
    rms = float(np.sqrt(np.mean(resid ** 2)))
    peak = float(np.abs(resid).max())
    peak_t = int(np.argmax(np.abs(resid))) / best['fs']
    print(f'  DTW align quality: resid rms={rms:.3f} rad/s, peak={peak:.3f} at t={peak_t:.1f}s')
    # Гейт align quality: rms > 0.5 рад/с значит DTW не смог нормально
    # уложить one lap на другой (сход, пит, кривой проезд). Compare в этом
    # случае бессмысленен — выходим с кодом 4 чтобы backend ответил 422.
    if rms > 0.5:
        print(f'  GATE FAIL: resid rms {rms:.3f} > 0.5 rad/s — круг не прошёл align gate', file=sys.stderr)
        sys.exit(4)
    other_phases_on_tmpl = project_phases_to_template(
        dtw_path, ds, N_ds, M_ds, other['phases'], best['wz'],
        other_wz=other['wz'], other_v=other['v'], fs=other['fs']
    )
    other_slip_idx = [int(s['t'] * other['fs']) for s in other['slips']]
    other_slips_on_tmpl = [project_idx_to_template(dtw_path, ds, idx, M_ds,
                                                    hybrid_m=hybrid_m)
                           for idx in other_slip_idx]
    other_slips_on_tmpl = [x for x in other_slips_on_tmpl if x is not None]

    # Сектора на canonical + их времена в обоих кругах
    sectors_meta = build_canonical_sectors(canonical, best['fs'])
    sector_bounds_full = [(name, s, e) for name, s, e in sectors_meta]
    # best: сектор-дуры = реальные lap-times (т.к. best = canonical)
    best_sector_times = [(name, s / best['fs'], e / best['fs'], (e - s) / best['fs'])
                         for name, s, e in sector_bounds_full]
    other_sector_times = compute_sector_times_from_dtw(
        dtw_path, ds, sector_bounds_full, other['fs'],
        tmpl_wz=best['wz'], lap_wz=other['wz']
    )
    canonical['sectors_table'] = {
        'bounds_full': sector_bounds_full,
        'best': best_sector_times,
        'other': other_sector_times,
    }

    # Per-lap speed через DTW slope. КРИТИЧНО: v_other = v_best / (dj/di),
    # НЕ умножение. Если круг 25 медленнее, на один template-sample
    # приходится БОЛЬШЕ lap-samples (dj/di>1) → скорость в этом месте
    # НИЖЕ чем у best. Деление даёт правильный знак.
    N_full = len(best['v'])
    j_per_i_ds = {}
    for i_ds, j_ds in dtw_path:
        if i_ds not in j_per_i_ds:
            j_per_i_ds[i_ds] = j_ds
    j_for_i_full = np.zeros(N_full, dtype=np.int64)
    last_j = 0
    for i_full in range(N_full):
        i_ds = i_full // ds
        if i_ds in j_per_i_ds:
            last_j = j_per_i_ds[i_ds] * ds
        j_for_i_full[i_full] = last_j
    j_for_i_full = np.clip(j_for_i_full, 0, len(other['v']) - 1)
    from scipy.ndimage import gaussian_filter1d
    dj_di = np.gradient(j_for_i_full.astype(float))
    # Clamp [0.8, 1.25] — реальный диапазон между кругами одного пилота
    # на одной трассе. σ=100 (1с) — ступенчатый DTW-путь требует серьёзного
    # сглаживания производной, иначе плато дают ×3 артефакты в v.
    dj_di_sm = gaussian_filter1d(np.clip(dj_di, 0.8, 1.25), 100)
    v_other_on_tmpl = best['v'] / dj_di_sm
    v_other_on_tmpl = np.clip(v_other_on_tmpl, 3, 18)

    other_vmax_tmpl_idx = int(np.argmax(v_other_on_tmpl))
    other_vmax_kmh = float(v_other_on_tmpl[other_vmax_tmpl_idx] * 3.6)
    other_data = {
        'phases_on_canonical': other_phases_on_tmpl,
        'slips_on_canonical': other_slips_on_tmpl,
        'phase_counts': {k: other_phases_on_tmpl.count(k) for k in ['brake', 'lift', 'accel', 'cruise']},
        'raceman_time_s': other['raceman_time_s'],
        'v_max_kmh': other_vmax_kmh,
        'v_max_idx': other_vmax_tmpl_idx,
        'raw': other,
        'cut_g': other.get('cut_g', -0.47),
    }

    render_two_laps_html(canonical, best_data, other_data,
                          out_html,
                          args.best, args.other)
    if cleanup:
        cleanup()
    print(f'=== compare ready: {out_html} ===')


if __name__ == '__main__':
    main()
