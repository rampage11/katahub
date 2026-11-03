"""Replay-harness Wave 2 live-compare.

Прокатывает watch_imu.bin из сохранённого zip-а через два WS-эндпоинта
бэкенда так, как это делает связка Watch→iPhone→katahub во время живого
заезда:

  1) `/live/{id}/publish` (JSON) — для `timing.lap`-событий.
     Шлём один event на каждый круг из timing.json по мере достижения
     его wallTime относительно начала replay.
  2) `/live/{id}/watch-imu` (binary) — для raw IMU батчей.
     1-секундные батчи в формате WatchIMUBatchFormat (header 8 байт +
     100 × 88 байт сэмплов).

Параметры:
  --zip <path>       — сессия с watch_imu.bin + timing.json
  --session-id <id>  — UUID сессии в katahub (должна существовать)
  --host <url>       — базовый URL бэкенда, default http://127.0.0.1:8080
  --speed <x>        — ускорение проигрывания (default 1.0 — реальное время)
  --max-laps <n>     — остановиться после n сымитированных кругов

Запуск:
  python3 replay_live.py --zip /tmp/test_session.zip \\
      --session-id 00000000-0000-0000-0000-000000000001 \\
      --host http://127.0.0.1:8080 --speed 20

Требует `websocket-client` (pip install websocket-client).
Для демо на prod — указать wss://kart.katahub.ru.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

try:
    import websocket  # type: ignore
except ImportError:
    print("нужен websocket-client: pip install websocket-client", file=sys.stderr)
    sys.exit(1)

IMU_SIZE = 88
BATCH_MAGIC = 0x494D
BATCH_HEADER = struct.Struct("<HBBI")  # magic, version, reserved, count


def ws_url(base: str, path: str) -> str:
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):] + path
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):] + path
    return base + path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", required=True)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--host", default="http://127.0.0.1:8080")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--max-laps", type=int, default=0)
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        imu_name = next((n for n in names if n.endswith("watch_imu.bin")), None)
        timing_name = next((n for n in names if n.endswith("timing.json")), None)
        if not imu_name or not timing_name:
            print("zip missing watch_imu.bin or timing.json", file=sys.stderr)
            return 2
        imu_bytes = zf.read(imu_name)
        timing = json.loads(zf.read(timing_name))

    n_samples = len(imu_bytes) // IMU_SIZE
    if n_samples < 100:
        print("too few IMU samples", file=sys.stderr)
        return 2
    # ts_ns первого сэмпла → базовое wall-time replay.
    first_ts_ns = struct.unpack_from("<Q", imu_bytes, 0)[0]
    print(f"loaded {n_samples} IMU samples, first ts={first_ts_ns}")

    # Строим список кругов (один pass на lapNumber, самый ранний).
    kart = str(timing.get("kartNumber", ""))
    passes = []
    seen = set()
    for p in timing.get("allPasses", []):
        if str(p.get("kartNumber")) != kart:
            continue
        n = p.get("lapNumber")
        ms = p.get("lastLapMs")
        wall = p.get("wallTime")
        if not n or not ms or not wall or n in seen:
            continue
        seen.add(n)
        passes.append({"lapNumber": n, "lapMs": ms, "wallTime": wall})
    passes.sort(key=lambda x: x["lapNumber"])
    if args.max_laps > 0:
        passes = passes[: args.max_laps]
    print(f"will replay {len(passes)} laps for kart {kart}")

    # Открываем оба WS.
    publish_url = ws_url(args.host, f"/live/{args.session_id}/publish")
    imu_url = ws_url(args.host, f"/live/{args.session_id}/watch-imu")
    print(f"publish: {publish_url}")
    print(f"watch-imu: {imu_url}")
    pub = websocket.create_connection(publish_url, timeout=10)
    imu_ws = websocket.create_connection(imu_url, timeout=10)

    # Делим IMU-данные на 1-секундные батчи по ts_ns.
    # ВАЖНО: ts_ns в оригинальном watch_imu.bin — это wall-time ИСХОДНОЙ записи
    # (несколько дней/недель назад). Backend принимает wallTime timing.lap-ов
    # в current UTC → если не переписать ts_ns в батчах на current wall, то
    # buffer будет содержать samples из прошлого, а refine ищет window в
    # current → Slice вернёт 0. Переписываем: ts_ns' = now + (ts_ns - first_ts_ns).
    now_ns = int(time.time() * 1e9)
    offset_ns = now_ns - first_ts_ns
    batches: list[tuple[int, bytes]] = []
    cur_start_ns = first_ts_ns
    cur_samples: list[bytes] = []
    for i in range(n_samples):
        off = i * IMU_SIZE
        ts_ns = struct.unpack_from("<Q", imu_bytes, off)[0]
        if ts_ns - cur_start_ns >= 1_000_000_000 and cur_samples:
            payload = BATCH_HEADER.pack(BATCH_MAGIC, 1, 0, len(cur_samples)) + b"".join(cur_samples)
            batches.append((cur_start_ns, payload))
            cur_samples = []
            cur_start_ns = ts_ns
        # Переписываем ts_ns в byte-slice перед добавлением
        sample = bytearray(imu_bytes[off: off + IMU_SIZE])
        struct.pack_into("<Q", sample, 0, ts_ns + offset_ns)
        cur_samples.append(bytes(sample))
    if cur_samples:
        payload = BATCH_HEADER.pack(BATCH_MAGIC, 1, 0, len(cur_samples)) + b"".join(cur_samples)
        batches.append((cur_start_ns, payload))
    print(f"built {len(batches)} 1-second IMU batches (ts_ns shifted by {offset_ns} ns)")

    # Запуск: wall-time 0 = первому IMU батчу. Lap wallTime из timing.json
    # относится к ДРУГОЙ записи в этом тестовом zip-е, поэтому для replay
    # мы маппим круги на IMU-таймлайн ЛИНЕЙНО: lap i шлём в момент
    # cumulative_sum(lapMs[:i+1]) по IMU-таймлайну.
    t0_wall = time.time()
    speed = max(args.speed, 0.01)

    def imu_time_at_batch(batch_idx: int) -> float:
        """Wall-time (sec since replay start) когда отправлять batch_idx."""
        return (batches[batch_idx][0] - first_ts_ns) / 1e9 / speed

    def send_batch(idx: int):
        imu_ws.send(batches[idx][1], opcode=websocket.ABNF.OPCODE_BINARY)
        cursor_pct = 100 * idx // max(len(batches) - 1, 1)
        if idx % 10 == 0:
            print(f"  imu batch {idx}/{len(batches)} ({cursor_pct}%)")

    # Собираем расписание lap-событий. ВАЖНО: wallTime события должен
    # указывать на END-ts_ns круга В СДВИНУТОМ БУФЕРЕ (now_at_t0 + cum_ms),
    # а не на current time в момент отправки. Иначе backend ищет window
    # вокруг wall_at_send, а последний sample круга в buffer-е находится
    # по ts_ns = now_at_t0 + cum_ms/speed... нет, по оригинальному offset,
    # т.к. ts_ns в batch'е = original_ts + offset_ns = now_at_t0 + original_offset.
    # Так что wallTime = (now_at_t0 + cum_ms/1000) в UNIX-сек.
    lap_schedule = []
    cum_ms = 0
    for p in passes:
        cum_ms += p["lapMs"]
        wall_end_s = t0_wall + cum_ms / 1000.0   # в сдвинутой IMU-таймлайн
        send_at_rel = cum_ms / 1000.0 / speed    # когда реально отправить (с учётом speed)
        lap_schedule.append((send_at_rel, p, wall_end_s))
    lap_cursor = 0

    def iso_utc(sec: float) -> str:
        gm = time.gmtime(sec)
        frac_ms = int((sec - int(sec)) * 1000)
        return time.strftime("%Y-%m-%dT%H:%M:%S", gm) + f".{frac_ms:03d}Z"

    for i in range(len(batches)):
        now_rel = time.time() - t0_wall
        target = imu_time_at_batch(i)
        if target > now_rel:
            time.sleep(target - now_rel)
        send_batch(i)
        # Проверяем, не настал ли момент отправки какого-то lap-события.
        while lap_cursor < len(lap_schedule) and lap_schedule[lap_cursor][0] <= (time.time() - t0_wall):
            _, p, wall_end_s = lap_schedule[lap_cursor]
            event = {
                "kind": "timing.lap",
                "wallNs": int(time.time() * 1e9),
                "payload": {
                    "lapNumber": p["lapNumber"],
                    "lapMs": p["lapMs"],
                    # wallTime указывает на END-ts_ns круга в сдвинутом буфере
                    "wallTime": iso_utc(wall_end_s),
                },
            }
            pub.send(json.dumps(event))
            print(f"  >> timing.lap lap={p['lapNumber']} lapMs={p['lapMs']} wallTime={event['payload']['wallTime']}")
            lap_cursor += 1

    print("replay complete; waiting 10s for compare.done to flush")
    time.sleep(10)
    pub.close()
    imu_ws.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
