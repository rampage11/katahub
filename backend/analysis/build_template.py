"""Admin-op: rebuild `tracks/<name>/{tmpl.npy, track.npz, features.npy}`.

Собирает шаблон трассы из хорошей сессии (watch_imu.bin + timing.json)
плюс аннотации (`--lap N`, лучший круг пилота). Три artifact'а:

1. `tmpl.npy`  — wz-профиль одного круга @100Hz. Используется в
   run-time для NCC-детекции границ (reference sess2.py).
2. `track.npz` — x, y, v, wz @50Hz. x,y → для карты. v → для hybrid
   dot placement. wz → для sector cuts refine.
3. `features.npy` — DTW features (wz/1.2, 0.5·z(|ah|)) @50Hz того же
   lap. Используется в `dtw_closed(T, F_lap[::2])` без пересчёта
   на каждом круге.

Запуск:
  # По zip (session upload):
  python3 analysis/build_template.py --track miks --zip SESSION.zip --lap 63

  # По уже распакованной директории:
  python3 analysis/build_template.py --track miks --dir SESSION_DIR --lap 63

  # По умолчанию берётся best lap из timing.json для указанного карта.

CORN и cuts_s **не** обновляются этим скриптом — они живут константами в
`track_template.py` и верифицируются оффлайн при изменении шаблона.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from karttelemetry import loader  # noqa: E402


def _load_imu(path: Path) -> dict[str, np.ndarray]:
    arr = loader.load_imu(path)
    t = arr["ts_ns"].astype(float)
    t = t / 1e3 if t[0] < 1e13 else t / 1e9
    quat = np.c_[arr["qx"], arr["qy"], arr["qz"], arr["qw"]].astype(float)
    gyro = np.c_[arr["gx"], arr["gy"], arr["gz"]].astype(float)
    accel = np.c_[arr["ax"], arr["ay"], arr["az"]].astype(float)
    rot = R.from_quat(quat)
    w = rot.apply(gyro)
    a = rot.apply(accel)
    yaw = np.unwrap(rot.as_euler("ZYX")[:, 0])
    return {
        "t": t,
        "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2],
        "wx": w[:, 0], "wy": w[:, 1], "wz": w[:, 2],
        "yaw": yaw,
    }


def _lowpass(x, cutoff, fs=100.0, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x)


def _unzip_to_tmp(zip_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="build_tmpl_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    return tmp


def _find(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    for p in root.rglob(name):
        return p
    raise FileNotFoundError(f"{name} not found under {root}")


def main() -> int:
    ap = argparse.ArgumentParser()
    gp = ap.add_mutually_exclusive_group(required=True)
    gp.add_argument("--zip", dest="zip_path", type=Path)
    gp.add_argument("--dir", dest="dir_path", type=Path)
    ap.add_argument("--track", required=True)
    ap.add_argument("--kart", default=None)
    ap.add_argument("--lap", type=int, default=None,
                     help="lap number (RaceMann). Default = best lap in session.")
    ap.add_argument("--out", default=None, type=Path,
                     help="output dir (default: analysis/tracks/<track>/)")
    args = ap.parse_args()

    cleanup = lambda: None
    if args.zip_path:
        root = _unzip_to_tmp(args.zip_path)
        cleanup = lambda: shutil.rmtree(root, ignore_errors=True)
    else:
        root = args.dir_path

    try:
        watch_imu_path = _find(root, "watch_imu.bin")
        timing_path = _find(root, "timing.json")
        timing = json.loads(timing_path.read_text())
        kart = args.kart or timing.get("kartNumber")
        if kart is None:
            raise RuntimeError("no kart number — pass --kart")
        passes = [p for p in timing.get("allPasses", [])
                  if str(p.get("kartNumber")) == str(kart)
                  and p.get("lapNumber") and p.get("lastLapMs")
                  and p["lastLapMs"] < 60_000]
        if not passes:
            raise RuntimeError(f"no laps for kart {kart} in timing")
        if args.lap:
            lap_rec = next((p for p in passes if p["lapNumber"] == args.lap), None)
            if lap_rec is None:
                raise RuntimeError(f"lap {args.lap} not found")
        else:
            lap_rec = min(passes, key=lambda p: p["lastLapMs"])

        print(f"=== Building track '{args.track}' from kart {kart} lap "
              f"{lap_rec['lapNumber']} ({lap_rec['lastLapMs']/1000:.3f} s)")

        df = _load_imu(watch_imu_path)
        fs = 100.0
        t = df["t"]
        tu = np.arange(t[0], t[-1], 1.0 / fs)
        w: dict[str, np.ndarray] = {"t": tu}
        for col in ("wz", "ax", "ay", "yaw"):
            w[col] = np.interp(tu, t, df[col])
        w["wz_f"] = _lowpass(w["wz"], 1.5, fs=fs)
        ah = np.hypot(w["ax"], w["ay"])
        w["ah_f"] = _lowpass(ah, 1.5, fs=fs)

        # Find lap window via existing lap detection
        # (детектор boundaries из v2 pipeline использует NCC с существующим
        # tmpl.npy. Для первого запуска: fallback — ∫wz = 2π*lap_count.)
        from karttelemetry.full_pipeline import _detect_lap_starts
        tracks_dir = Path(__file__).parent / "tracks" / args.track
        tmpl_existing = tracks_dir / "tmpl.npy"
        starts = None
        if tmpl_existing.exists():
            tmpl = np.load(tmpl_existing).astype(np.float64)
            starts, _ = _detect_lap_starts(w["wz_f"], tmpl, fs=fs)

        if starts is None or len(starts) < 2:
            print("  bootstrapping template from scratch (no existing tmpl.npy)")
            # Use heading-based detection (unwrap yaw)
            psi = w["yaw"]
            psi_rel = (psi - psi[0]) * np.sign(w["wz_f"].sum())
            n_laps_est = int(abs(psi_rel[-1]) / (2 * np.pi))
            if n_laps_est < 2:
                raise RuntimeError("cannot detect laps")
            # Boundary = где psi_rel пересекает kol*2π
            starts = []
            for k in range(1, n_laps_est + 1):
                idx = np.argmin(np.abs(psi_rel - k * 2 * np.pi))
                starts.append(int(idx))
            starts = np.array(starts, dtype=np.int64)

        # Match RaceMann lap_number → IMU boundary index
        dur_imu = np.diff(starts) / fs
        rm_laps = sorted(passes, key=lambda p: p.get("lapNumber", 0))
        dur_rm = np.array([p["lastLapMs"] / 1000.0 for p in rm_laps])
        # Match offset
        best = (0, float("inf"))
        for k in range(-5, 6):
            if k >= 0:
                n = min(len(dur_imu), len(dur_rm) - k)
            else:
                n = min(len(dur_imu) + k, len(dur_rm))
            if n <= 0:
                continue
            a_start = max(0, -k)
            b_start = max(0, k)
            e = np.abs(dur_imu[a_start:a_start + n] - dur_rm[b_start:b_start + n])
            m = float(np.median(e))
            if m < best[1]:
                best = (k, m)
        offset_k = best[0]
        print(f"  IMU-RaceMann match offset={offset_k} median_err={best[1]:.2f}s")

        # Find IMU index for selected lap
        rm_idx = rm_laps.index(lap_rec)
        imu_idx = rm_idx - offset_k
        if imu_idx < 0 or imu_idx >= len(starts):
            raise RuntimeError(f"lap {lap_rec['lapNumber']} outside IMU range")
        a_lap = int(starts[imu_idx])
        b_lap = a_lap + int(round(lap_rec["lastLapMs"] / 1000.0 * fs))
        if b_lap > len(w["t"]):
            b_lap = len(w["t"])
        print(f"  lap window: [{a_lap}:{b_lap}] = {(b_lap-a_lap)/fs:.3f}s")

        # Extract wz for tmpl.npy (@100Hz)
        tmpl_new = w["wz_f"][a_lap:b_lap].astype(np.float64)

        # θ grid-search for a_lat/a_long
        best_c, best_th = -2.0, 0.0
        wz_seg = w["wz_f"][a_lap:b_lap]
        yaw_seg = w["yaw"][a_lap:b_lap]
        ax_seg = _lowpass(w["ax"], 1.5, fs=fs)[a_lap:b_lap]
        ay_seg = _lowpass(w["ay"], 1.5, fs=fs)[a_lap:b_lap]
        for deg in range(0, 360, 2):
            th = np.radians(deg)
            psi = yaw_seg + th
            alat = -ax_seg * np.sin(psi) + ay_seg * np.cos(psi)
            if alat.std() < 1e-6 or wz_seg.std() < 1e-6:
                continue
            c = float(np.corrcoef(alat, wz_seg)[0, 1])
            if c > best_c:
                best_c = c
                best_th = th
        psi = yaw_seg + best_th
        alat = -ax_seg * np.sin(psi) + ay_seg * np.cos(psi)
        along = ax_seg * np.cos(psi) + ay_seg * np.sin(psi)
        along_sm = gaussian_filter1d(along, 8)
        print(f"  θ = {np.degrees(best_th):.0f}°  corr = {best_c:.3f}")

        # Physical sign check
        from karttelemetry.full_pipeline import _flip_theta_if_needed
        if _flip_theta_if_needed(wz_seg, along_sm, [0, b_lap - a_lap], fs=fs):
            print("  sign check: flipping")
            alat = -alat
            along = -along
            along_sm = -along_sm

        # Build trajectory (v, x, y)
        v = np.where(np.abs(wz_seg) > 1.0,
                      np.abs(alat) * 9.81 / np.maximum(np.abs(wz_seg), 0.1),
                      np.nan)
        v = np.clip(v, 4, 20)
        good = ~np.isnan(v)
        idx = np.arange(len(v))
        if good.sum() > 1:
            v = np.interp(idx, idx[good], v[good])
        v = gaussian_filter1d(v, 15, mode="wrap")

        psi_i = np.cumsum(wz_seg) / fs
        total = psi_i[-1] - psi_i[0]
        target = round(total / (2 * np.pi)) * 2 * np.pi
        if abs(target) > 1e-6:
            psi_i = psi_i - (total - target) * idx / max(len(psi_i) - 1, 1)
        x = np.cumsum(v * np.cos(psi_i)) / fs
        y = np.cumsum(v * np.sin(psi_i)) / fs
        x = x - x[-1] * idx / max(len(x) - 1, 1)
        y = y - y[-1] * idx / max(len(y) - 1, 1)

        print(f"  trajectory len={v.sum()/fs:.1f}m, closure after fix={np.hypot(x[-1], y[-1]):.2f}m")

        # Downsample 100 → 50 Hz
        ds = slice(None, None, 2)
        x50 = x[ds]
        y50 = y[ds]
        v50 = v[ds]
        wz50 = wz_seg[ds]
        tt50 = np.arange(len(x50)) / 50.0

        # Features T @50Hz
        ah_seg = w["ah_f"][a_lap:b_lap]
        ah50 = ah_seg[ds]
        mu = float(ah50.mean())
        sd = max(float(ah50.std()), 1e-6)
        features_50 = np.c_[wz50 / 1.2, (ah50 - mu) / sd * 0.5]

        # Output
        out_dir = args.out or (Path(__file__).parent / "tracks" / args.track)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "tmpl.npy", tmpl_new.astype(np.float64))
        np.savez(out_dir / "track.npz",
                  x=x50.astype(np.float64), y=y50.astype(np.float64),
                  v=v50.astype(np.float64), tt=tt50.astype(np.float64),
                  wz=wz50.astype(np.float64))
        np.save(out_dir / "features.npy", features_50.astype(np.float64))
        print(f"wrote {out_dir}/{{tmpl.npy, track.npz, features.npy}}")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
