"""
CLI: process a MOT17 sequence into footfall events.

Example
-------
    python scripts/process_mot17.py \
        --sequence MOT17/train/MOT17-11-FRCNN \
        --line horizontal \
        --start-time "2026-07-09 10:00:00" \
        --export-video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.video_processor import process_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an input source (MOT17 sequence, image folder, or "
        "video file): detect, track, and count footfall."
    )
    parser.add_argument(
        "--sequence",
        "--source",
        dest="sequence",
        required=True,
        help="Path to a MOT17 sequence dir, an image folder (e.g. "
        "MallDataset/frames/frames), or a video file (.mp4/.avi/…).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Label for this source (used for the DB + output filename). "
        "Defaults to the folder/file name.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override frame rate (for image folders / videos with no metadata).",
    )
    parser.add_argument(
        "--line",
        choices=["horizontal", "vertical"],
        default=config.LINE_ORIENTATION,
        help="Counting line orientation (default: %(default)s).",
    )
    parser.add_argument(
        "--line-position",
        type=float,
        default=config.LINE_POSITION,
        help="Line position as a fraction of the frame (0-1, default: %(default)s).",
    )
    parser.add_argument(
        "--swap-in-out",
        action="store_true",
        help="Swap the in/out direction mapping.",
    )
    parser.add_argument(
        "--start-time",
        default=config.DEFAULT_START_TIME,
        help='Clock anchor for timestamps "YYYY-MM-DD HH:MM:SS"; timestamps then '
        "advance by real elapsed video time (default: %(default)s).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONF_THRESHOLD,
        help="Detection confidence threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        help="Also write an annotated mp4 to outputs/annotated_videos/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        summary = process_sequence(
            sequence_path=args.sequence,
            line_orientation=args.line,
            line_position=args.line_position,
            swap_in_out=args.swap_in_out,
            start_time=args.start_time,
            conf_threshold=args.conf,
            export_video=args.export_video,
            name=args.name,
            fps=args.fps,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # Final summary.
    print("\n" + "=" * 48)
    print("  PROCESSING COMPLETE")
    print("=" * 48)
    print(f"  Sequence          : {summary['sequence_name']}")
    print(f"  Frames processed  : {summary['total_frames']}")
    print(f"  Total IN          : {summary['total_in']}")
    print(f"  Total OUT         : {summary['total_out']}")
    print(f"  Current occupancy : {summary['occupancy']}")
    print(f"  Unique tracks     : {summary['unique_tracks']}")
    print(f"  Output video      : {summary['output_video'] or '(not exported)'}")
    print("=" * 48)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
