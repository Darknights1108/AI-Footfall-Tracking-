"""
MOT17 sequence loader.

Reads a MOT17 sequence directly from its ``img1`` folder (no mp4 conversion
needed) and yields frames in correct numerical order together with a simulated
retail timestamp. Sequence metadata (frame rate, resolution, length) is parsed
from ``seqinfo.ini`` when present, with graceful fallbacks otherwise.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import cv2

from . import config


@dataclass
class SequenceInfo:
    """Metadata describing a single MOT17 sequence."""

    name: str
    frame_rate: float
    seq_length: int
    width: int
    height: int
    image_dir: Path


@dataclass
class FrameData:
    """A single decoded frame together with its index and simulated timestamp."""

    image: "cv2.Mat"
    frame_index: int  # 1-based, matching MOT17 file numbering
    timestamp: datetime


class Mot17SequenceLoader:
    """
    Load and iterate over a MOT17 image sequence.

    Parameters
    ----------
    sequence_path:
        Path to a sequence directory, e.g. ``MOT17/train/MOT17-11-FRCNN``.
    start_time:
        Simulated wall-clock time assigned to the first frame. Subsequent frames
        advance by ``1 / frame_rate`` seconds.
    """

    def __init__(
        self,
        sequence_path: str | Path,
        start_time: str | None = None,
    ) -> None:
        self.sequence_path = Path(sequence_path)
        if not self.sequence_path.exists():
            raise FileNotFoundError(
                f"MOT17 sequence folder not found: {self.sequence_path}\n"
                f"Expected a path like 'MOT17/train/MOT17-11-FRCNN'."
            )

        self.image_dir = self.sequence_path / "img1"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"'img1' folder not found inside sequence: {self.image_dir}"
            )

        # Collect and numerically sort the frame images (e.g. 000001.jpg ...).
        self._frame_paths = sorted(
            self.image_dir.glob("*.jpg"),
            key=lambda p: int(p.stem),
        )
        if not self._frame_paths:
            raise FileNotFoundError(f"No .jpg frames found in {self.image_dir}")

        start = start_time or config.DEFAULT_START_TIME
        self.start_time = datetime.strptime(start, config.TIMESTAMP_FORMAT)

        self.info = self._read_seqinfo()

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def _read_seqinfo(self) -> SequenceInfo:
        """Parse seqinfo.ini, falling back to sensible defaults when missing."""
        name = self.sequence_path.name
        seqinfo_path = self.sequence_path / "seqinfo.ini"

        frame_rate = config.DEFAULT_FPS
        seq_length = len(self._frame_paths)
        width = height = 0

        if seqinfo_path.is_file():
            parser = configparser.ConfigParser()
            # MOT17 seqinfo.ini files are simple; read defensively.
            try:
                parser.read(seqinfo_path)
                section = parser["Sequence"]
                name = section.get("name", name)
                frame_rate = section.getfloat("frameRate", config.DEFAULT_FPS)
                seq_length = section.getint("seqLength", seq_length)
                width = section.getint("imWidth", 0)
                height = section.getint("imHeight", 0)
            except Exception:
                # Corrupt/unexpected file — fall through to image-derived values.
                pass

        # If resolution was not available from seqinfo, read the first image.
        if width == 0 or height == 0:
            first = cv2.imread(str(self._frame_paths[0]))
            if first is None:
                raise IOError(f"Failed to read first frame: {self._frame_paths[0]}")
            height, width = first.shape[:2]

        if frame_rate <= 0:
            frame_rate = config.DEFAULT_FPS

        return SequenceInfo(
            name=name,
            frame_rate=frame_rate,
            seq_length=seq_length,
            width=width,
            height=height,
            image_dir=self.image_dir,
        )

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._frame_paths)

    def _real_timestamp(self, frame_index: int) -> datetime:
        """
        Map a 1-based frame index onto its REAL elapsed time in the clip.

        The timestamp is anchored at ``start_time`` and advances by the true
        elapsed video time, ``(frame_index - 1) / fps``. No stretching or faked
        business-day clock is applied.
        """
        return self.start_time + timedelta(
            seconds=(frame_index - 1) / self.info.frame_rate
        )

    def __iter__(self) -> Iterator[FrameData]:
        """Yield frames in order with a real-time timestamp for each."""
        for idx, path in enumerate(self._frame_paths, start=1):
            image = cv2.imread(str(path))
            if image is None:
                # Skip unreadable frames rather than aborting the whole run.
                continue
            timestamp = self._real_timestamp(idx)
            yield FrameData(image=image, frame_index=idx, timestamp=timestamp)
