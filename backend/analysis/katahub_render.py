#!/usr/bin/env python3
"""CLI wrapper: session zip or dir → analysis JSON for katahub dashboard.

Usage:
  python3 katahub_render.py --zip /path/to/session.zip > /tmp/analysis.json
  python3 katahub_render.py --dir /path/to/session_dir > /tmp/analysis.json

Exits 0 on success with compact JSON on stdout.
Non-zero exit with a message on stderr on error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _repo_root() -> Path:
    # analysis/katahub_render.py → repo/analysis
    return Path(__file__).resolve().parent


def _ensure_import_path() -> None:
    sys.path.insert(0, str(_repo_root()))


def _find_in_dir(root: Path, filename: str) -> Path | None:
    """Finds `filename` directly in `root` or in a single subdirectory of root."""
    direct = root / filename
    if direct.exists():
        return direct
    for p in root.rglob(filename):
        if p.is_file():
            return p
    return None


def _extract_zip(zip_path: Path, tmp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    return tmp_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a session into dashboard JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", dest="zip_path", type=Path, help="session .zip")
    group.add_argument("--dir", dest="dir_path", type=Path, help="session directory")
    parser.add_argument("--indent", type=int, default=None, help="JSON indent (default: compact)")
    args = parser.parse_args()

    _ensure_import_path()
    from karttelemetry.full_pipeline import analyze_session

    tmp = tempfile.mkdtemp(prefix="katahub_")
    try:
        tmp_path = Path(tmp)
        if args.zip_path:
            if not args.zip_path.exists():
                print(f"zip not found: {args.zip_path}", file=sys.stderr)
                return 2
            _extract_zip(args.zip_path, tmp_path)
            root = tmp_path
        else:
            root = args.dir_path
            if not root.exists() or not root.is_dir():
                print(f"dir not found: {root}", file=sys.stderr)
                return 2

        watch_imu = _find_in_dir(root, "watch_imu.bin")
        if watch_imu is None:
            print("watch_imu.bin not found in session", file=sys.stderr)
            return 3
        timing = _find_in_dir(root, "timing.json")

        result = analyze_session(watch_imu, timing)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    json.dump(result, sys.stdout, indent=args.indent, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
