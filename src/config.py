"""
Central configuration for the AI Footfall Tracking & Retail Analytics project.

All tunable thresholds, paths, and default settings live here so the rest of the
codebase has a single source of truth. Values here are *defaults* — the CLI in
``scripts/process_mot17.py`` can override several of them per run.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = the directory that contains this ``src`` package's parent.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# The MOT17 dataset is large (~5.7 GB) and is kept in place rather than copied
# into ``data/``. Point this at wherever the dataset lives.
MOT17_ROOT: Path = PROJECT_ROOT / "MOT17"

# SQLite database that stores footfall events and processing runs.
DATA_DIR: Path = PROJECT_ROOT / "data"
DB_PATH: Path = DATA_DIR / "footfall.db"

# Where annotated output videos are written.
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
ANNOTATED_VIDEO_DIR: Path = OUTPUT_DIR / "annotated_videos"

# --------------------------------------------------------------------------- #
# Detection (YOLO11)
# --------------------------------------------------------------------------- #
# ``yolo11n.pt`` (nano) is chosen for speed; weights auto-download on first use.
MODEL_NAME: str = "yolo11n.pt"

# COCO class id for "person". YOLO11 pretrained on COCO uses id 0 for person.
PERSON_CLASS_ID: int = 0

# Minimum detection confidence to keep a box.
CONF_THRESHOLD: float = 0.30

# Compute device: "cuda" if a GPU is available, else "cpu". Resolved lazily so
# importing this module never requires torch to be installed.
def resolve_device() -> str:
    """Return "cuda" when a CUDA GPU is available, otherwise "cpu"."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Line-crossing counter
# --------------------------------------------------------------------------- #
# "horizontal" draws a line across the width (people cross vertically);
# "vertical" draws a line down the height (people cross horizontally).
LINE_ORIENTATION: str = "horizontal"

# Where the line sits, as a fraction of the frame dimension (0.0–1.0).
# 0.5 = middle of the frame.
LINE_POSITION: float = 0.5

# A friendly name recorded with each crossing event.
LINE_NAME: str = "main_entrance"

# LineZone decides "in" vs "out" from the geometry of the line. If the mapping
# comes out reversed for a given camera, flip this flag instead of redrawing.
SWAP_IN_OUT: bool = False

# --------------------------------------------------------------------------- #
# Timestamp simulation
# --------------------------------------------------------------------------- #
# MOT17 clips are short, so we simulate a retail business clock:
#   timestamp = START_TIME + frame_index / fps
DEFAULT_START_TIME: str = "2026-07-09 10:00:00"
TIMESTAMP_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Fallback frame rate when a sequence has no readable seqinfo.ini.
DEFAULT_FPS: float = 30.0

# Timestamps are anchored at START_TIME and advance by REAL elapsed video time
# (frame_index / fps). We deliberately do NOT fake a "retail day" clock — the
# dashboard plots crossings against the true video timeline so nothing is
# misrepresented as real business hours.

# Width (in real video seconds) of each bucket when plotting crossings over the
# clip timeline.
TIMELINE_BIN_SECONDS: float = 2.0

# --------------------------------------------------------------------------- #
# Analytics alert thresholds
# --------------------------------------------------------------------------- #
# Trigger an overcrowding alert when current occupancy exceeds this value.
OCCUPANCY_ALERT_THRESHOLD: int = 30

# Trigger a low-traffic alert when total unique visitors falls below this value.
LOW_TRAFFIC_THRESHOLD: int = 5
