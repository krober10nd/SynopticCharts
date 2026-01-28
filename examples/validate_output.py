"""\
Example Output Validation Script

This script validates that the SynopticCharts examples produced reasonable output.

Checks:
- PNG chart files exist and exceed a minimum size threshold
- MP4 video file exists and appears to be a valid MP4 container
- Frame directory contains at least one PNG frame

Usage:
  python examples/validate_output.py
"""

from __future__ import annotations

from pathlib import Path


def _check_file(path: Path, min_bytes: int) -> tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING: {path}"
    size = path.stat().st_size
    if size < min_bytes:
        return False, f"TOO SMALL: {path} ({size} bytes < {min_bytes})"
    return True, f"OK: {path} ({size/1024:.1f} KB)"


def _looks_like_mp4(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING: {path}"
    # MP4 commonly starts with an ftyp box near the beginning.
    with path.open("rb") as f:
        header = f.read(64)
    if b"ftyp" not in header:
        return False, f"NOT MP4?: {path} (missing 'ftyp' in header)"
    return True, f"OK: {path} (looks like MP4)"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "output"

    expected_pngs = [
        output / "basic_chart.png",
        output / "northeast_chart.png",
        output / "west_coast_chart.png",
    ]

    expected_video = output / "forecast_animation.mp4"
    frames_dir = output / "frames"

    print("Validating SynopticCharts example outputs")
    print("=" * 60)

    ok_all = True

    print("\nCharts:")
    for p in expected_pngs:
        ok, msg = _check_file(p, min_bytes=100_000)
        print(f"  {msg}")
        ok_all = ok_all and ok

    print("\nFrames:")
    if not frames_dir.exists():
        print(f"  MISSING: {frames_dir}")
        ok_all = False
    else:
        frames = sorted(frames_dir.glob("*.png"))
        if not frames:
            print(f"  EMPTY: {frames_dir} (no .png frames)")
            ok_all = False
        else:
            print(f"  OK: {frames_dir} ({len(frames)} frames)")

    print("\nVideo:")
    ok, msg = _looks_like_mp4(expected_video)
    print(f"  {msg}")
    ok_all = ok_all and ok

    print("\nSummary:")
    if ok_all:
        print("  SUCCESS: All expected outputs look reasonable")
        return 0

    print("  FAIL: One or more outputs missing/invalid")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
