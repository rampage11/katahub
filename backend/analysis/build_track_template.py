"""Строит шаблон трассы из эталонной сессии: (wz, |a|, v, x, y @ 50 Гц)
плюс метаданные (track, kart, lap_number, time_s). Используется в live
для bounds.refine (локальный поиск границы круга по корреляции с
"хвост + голова" шаблона).

Шаблон строится из ЛУЧШЕГО круга сессии — это минимизирует residual для
DTW с типичным кругом и даёт самую чистую геометрию.

Запуск:
  # Из зипа:
  python3 build_track_template.py --zip SESSION.zip --track miks --out tracks/miks.npz

  # Из распакованной директории:
  python3 build_track_template.py --dir /path/to/session --track miks --out tracks/miks.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from karttelemetry import loader
from lap31_standalone import process_lap, compute_session_sf_offset
from laps_compare import load_session_from_zip, _find_in_dir


def _load_session(zip_path, dir_path):
    if zip_path:
        return load_session_from_zip(Path(zip_path))
    root = Path(dir_path)
    watch_imu = _find_in_dir(root, 'watch_imu.bin')
    timing_path = _find_in_dir(root, 'timing.json')
    if watch_imu is None:
        raise RuntimeError(f'watch_imu.bin not found in {root}')
    if timing_path is None:
        raise RuntimeError(f'timing.json not found in {root}')
    samples = np.fromfile(watch_imu, dtype=loader.IMU_DTYPE)
    timing = json.loads(timing_path.read_text())
    kart = timing.get('kartNumber') or ''
    return samples, timing, str(kart), lambda: None


def main():
    ap = argparse.ArgumentParser()
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument('--zip', dest='zip_path', default=None)
    gp.add_argument('--dir', dest='dir_path', default=None)
    ap.add_argument('--track', required=True, help='имя трассы, в метаданные')
    ap.add_argument('--kart', default=None)
    ap.add_argument('--lap', type=int, default=None,
                     help='конкретный круг; по умолчанию — best по времени')
    ap.add_argument('--out', required=True, help='.npz файл')
    args = ap.parse_args()

    samples, timing, kart_from_timing, cleanup = _load_session(args.zip_path, args.dir_path)
    kart = args.kart or kart_from_timing
    try:
        sf_offset_s, _ = compute_session_sf_offset(samples=samples, timing=timing, kart_number=kart)
        our = [p for p in timing['allPasses']
               if str(p.get('kartNumber')) == str(kart) and p.get('lapNumber') and p.get('lastLapMs')]
        if not our:
            raise RuntimeError(f'no laps for kart {kart} in timing')
        if args.lap:
            lap_number = args.lap
        else:
            best = min(our, key=lambda p: p['lastLapMs'])
            lap_number = best['lapNumber']
            print(f'best lap = {lap_number} @ {best["lastLapMs"]/1000:.3f}s')

        d = process_lap(lap_number, samples=samples, timing=timing,
                         kart_number=kart, sf_offset_override=sf_offset_s)
        # Downsample 100 → 50 Гц (каждая 2-я точка). Для bounds.refine такого
        # разрешения более чем достаточно, и шаблон становится в 2 раза
        # компактнее + корреляция считается быстрее.
        ds = slice(None, None, 2)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            fs=np.array([50.0]),
            wz=d['wz'][ds].astype(np.float32),
            ah=d['ah'][ds].astype(np.float32),
            v=d['v'][ds].astype(np.float32),
            x=d['x'][ds].astype(np.float32),
            y=d['y'][ds].astype(np.float32),
            meta=np.array([json.dumps({
                'track': args.track,
                'kart': str(kart),
                'lap_number': lap_number,
                'time_s': d['raceman_time_s'],
                'theta_deg': d['theta_deg'],
                'corners': len(d['corners']),
                'lap_len_m': d['lap_len_m'],
                'samples_at_50hz': len(d['wz'][ds]),
            }, ensure_ascii=False)], dtype=object),
        )
        print(f'wrote {out_path} ({out_path.stat().st_size} bytes)')
    finally:
        cleanup()


if __name__ == '__main__':
    main()
