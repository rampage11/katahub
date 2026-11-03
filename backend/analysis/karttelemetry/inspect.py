"""CLI entry point: summarise a session dir and emit summary.png."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import loader


def _format_size(nbytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TiB"


def _print_summary(session_dir: Path, session: dict[str, Any]) -> None:
    manifest = session["manifest"]
    sid = manifest.get("id", "?")
    started = manifest.get("startedAt", "?")
    ended = manifest.get("endedAt", "?")
    device = manifest.get("deviceModel") or manifest.get("deviceName") or "?"

    print(f"session: {sid}")
    print(f"device:  {device}")
    print(f"started: {started}")
    print(f"ended:   {ended}")
    print(f"path:    {session_dir}")
    # Контекст из обогащённого манифеста (новые optional поля). Старые сессии
    # этих ключей не имеют — молча пропускаем.
    extras: list[str] = []
    if (driver := manifest.get("driverId")):
        extras.append(f"driver={str(driver)[:8]}")
    if (kart := manifest.get("kartNumber")):
        extras.append(f"kart={kart}")
    if (track := manifest.get("trackName")):
        extras.append(f"track={track}")
    if (rid := manifest.get("timingRaceID")):
        extras.append(f"timingRace={rid[:8]}")
    if extras:
        print("ctx:     " + "  ".join(extras))

    # Собираем сводку по файлам: берём и manifest, и реально лежащие файлы.
    manifest_by_filename = {s.get("filename"): s for s in manifest.get("sources", [])}
    known_files = ("iphone_imu.bin", "iphone_audio.wav", "watch_imu.bin", "heart_rate.bin")

    print("sources:")
    for filename in known_files:
        fpath = session_dir / filename
        if not fpath.exists():
            continue
        src = manifest_by_filename.get(filename, {})
        kind = src.get("kind") or filename.replace(".bin", "").replace(".wav", "")
        count = src.get("sampleCount")
        rate = src.get("sampleRateHz")
        size = fpath.stat().st_size
        in_manifest = filename in manifest_by_filename
        status = "ok" if in_manifest else "orphan"
        rate_s = f"{rate:g} Hz" if rate else "variable"
        count_s = f"samples={count}" if count is not None else "samples=?"
        print(
            f"  - {kind:<12} {filename:<24} {status:<7} "
            f"{_format_size(size):>10}  {count_s}  rate={rate_s}"
        )

    orphans = [
        f for f in known_files
        if (session_dir / f).exists() and f not in manifest_by_filename
    ]
    if orphans:
        print(f"note: {len(orphans)} file(s) on disk не упомянуты в manifest — грузим напрямую.")

    timing = session.get("timing")
    if timing:
        laps = timing.get("laps", [])
        flags = timing.get("flags", [])
        kart = timing.get("kartNumber") or "?"
        race_id = (timing.get("raceID") or "")[:8]
        print(f"timing: race={race_id} kart={kart}  laps={len(laps)}  flags={len(flags)}")
        if laps:
            # Форматируем как MM:SS.mmm — лучший круг выделяем звёздочкой.
            best_ms = min(l["lapMs"] for l in laps if l.get("lapMs"))
            for lap in laps:
                ms = lap.get("lapMs", 0)
                m, s_ms = divmod(ms, 1000)
                mm, s = divmod(m, 60)
                mark = " *" if ms == best_ms else "  "
                chaser = lap.get("chaserGapMs")
                chaser_s = f"  сзади +{chaser/1000:.1f}" if chaser is not None else ""
                print(
                    f"  lap {lap.get('lapNumber','?'):>3} "
                    f"{mm:02d}:{s:02d}.{s_ms:03d}{mark}  pos={lap.get('position','?')}"
                    f"{chaser_s}"
                )

    if session.get("_errors"):
        print("warnings:")
        for e in session["_errors"]:
            print(f"  - {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="karttelemetry.inspect",
        description="Summarise a session directory and emit summary.png.",
    )
    parser.add_argument("session_dir", type=Path, help="Path to session directory")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip loading audio (useful for quick summaries).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (defaults to <session_dir>/summary.png).",
    )
    args = parser.parse_args(argv)

    session_dir: Path = args.session_dir
    if not session_dir.is_dir():
        print(f"error: not a directory: {session_dir}", file=sys.stderr)
        return 2
    if not (session_dir / "manifest.json").exists():
        print(f"error: manifest.json not found in {session_dir}", file=sys.stderr)
        return 2

    session = loader.load_session(session_dir, load_audio_data=not args.no_audio)
    _print_summary(session_dir, session)

    # Импортируем plots лениво, чтобы импорт модуля не тянул matplotlib.
    from . import plots

    fig = plots.summary_figure(session)
    out_path = args.output or (session_dir / "summary.png")
    fig.savefig(out_path, dpi=120)
    # Закрываем фигуру явно, чтобы освободить ресурсы (важно для CLI).
    import matplotlib.pyplot as plt

    plt.close(fig)

    print(f"wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
