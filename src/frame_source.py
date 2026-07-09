"""
Flexible frame sources.

The pipeline can ingest three kinds of input, all exposed through the same
``open_source()`` factory so the rest of the code is source-agnostic:

* **MOT17 sequence directory** — a folder containing ``img1/`` (+ ``seqinfo.ini``).
* **Any image folder** — e.g. the Mall dataset's ``frames/`` folder, with
  arbitrary file names such as ``seq_000001.jpg``.
* **A video file** — ``.mp4`` / ``.avi`` / ``.mov`` / ``.mkv``.

Every source yields :class:`FrameData` (image, 1-based index, real timestamp)
and exposes a :class:`SequenceInfo` describing name, fps and resolution.
Timestamps advance by REAL elapsed video time (``frame_index / fps``).
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import cv2

from . import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

# Folder names that don't make good sequence labels — fall back to the parent.
_GENERIC_DIR_NAMES = {"frames", "images", "img", "img1"}


@dataclass
class SequenceInfo:
    """Metadata describing an input source."""

    name: str
    frame_rate: float
    seq_length: int
    width: int
    height: int
    source_type: str  # "images" | "video"


@dataclass
class FrameData:
    """A single decoded frame with its index and real-elapsed timestamp."""

    image: "cv2.Mat"
    frame_index: int  # 1-based
    timestamp: datetime


def _natural_key(path: Path):
    """Sort key that orders by the last number in the filename (000001, seq_42…)."""
    nums = re.findall(r"\d+", path.stem)
    return (int(nums[-1]) if nums else -1, path.stem)


def _clean_name(path: Path, override: str | None) -> str:
    """Pick a readable sequence label."""
    if override:
        return override
    name = path.name
    # If the folder is generically named (frames/, images/…), use its parent.
    if name.lower() in _GENERIC_DIR_NAMES and path.parent != path:
        name = path.parent.name
    return name


def _parse_start(start_time: str | None) -> datetime:
    return datetime.strptime(
        start_time or config.DEFAULT_START_TIME, config.TIMESTAMP_FORMAT
    )


# --------------------------------------------------------------------------- #
# Image folder / MOT17 sequence
# --------------------------------------------------------------------------- #
class ImageSequenceLoader:
    """
    Load a folder of image frames.

    Works for MOT17 sequences (``<seq>/img1/*.jpg`` + ``seqinfo.ini``) *and* for
    arbitrary image folders with any naming convention.
    """

    def __init__(
        self,
        path: str | Path,
        start_time: str | None = None,
        name: str | None = None,
        fps: float | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Input folder not found: {self.path}")

        # Prefer a MOT17-style ``img1`` subfolder; otherwise use the folder itself.
        img1 = self.path / "img1"
        self.image_dir = img1 if img1.is_dir() else self.path

        self._frame_paths = sorted(
            (p for p in self.image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=_natural_key,
        )
        if not self._frame_paths:
            raise FileNotFoundError(f"No image frames found in {self.image_dir}")

        self.start_time = _parse_start(start_time)
        self._name_override = name
        self._fps_override = fps
        self.info = self._read_info()

    def _read_info(self) -> SequenceInfo:
        name = _clean_name(self.path, self._name_override)
        frame_rate = self._fps_override or config.DEFAULT_FPS
        seq_length = len(self._frame_paths)
        width = height = 0

        # Use MOT17 seqinfo.ini when present (unless the caller forced values).
        seqinfo = self.path / "seqinfo.ini"
        if seqinfo.is_file():
            try:
                parser = configparser.ConfigParser()
                parser.read(seqinfo)
                section = parser["Sequence"]
                if self._name_override is None:
                    name = section.get("name", name)
                if self._fps_override is None:
                    frame_rate = section.getfloat("frameRate", frame_rate)
                width = section.getint("imWidth", 0)
                height = section.getint("imHeight", 0)
            except Exception:
                pass  # fall through to image-derived values

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
            source_type="images",
        )

    def __len__(self) -> int:
        return len(self._frame_paths)

    def _timestamp(self, frame_index: int) -> datetime:
        return self.start_time + timedelta(
            seconds=(frame_index - 1) / self.info.frame_rate
        )

    def __iter__(self) -> Iterator[FrameData]:
        for idx, path in enumerate(self._frame_paths, start=1):
            image = cv2.imread(str(path))
            if image is None:
                continue  # skip unreadable frames rather than aborting
            yield FrameData(image=image, frame_index=idx, timestamp=self._timestamp(idx))


# --------------------------------------------------------------------------- #
# Video file
# --------------------------------------------------------------------------- #
class VideoLoader:
    """Load frames directly from a video file via OpenCV."""

    def __init__(
        self,
        path: str | Path,
        start_time: str | None = None,
        name: str | None = None,
        fps: float | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {self.path}")
        frame_rate = fps or cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.start_time = _parse_start(start_time)
        self.info = SequenceInfo(
            name=name or self.path.stem,
            frame_rate=frame_rate if frame_rate > 0 else config.DEFAULT_FPS,
            seq_length=max(length, 0),
            width=width,
            height=height,
            source_type="video",
        )

    def __len__(self) -> int:
        return self.info.seq_length

    def _timestamp(self, frame_index: int) -> datetime:
        return self.start_time + timedelta(
            seconds=(frame_index - 1) / self.info.frame_rate
        )

    def __iter__(self) -> Iterator[FrameData]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                yield FrameData(image=frame, frame_index=idx, timestamp=self._timestamp(idx))
        finally:
            cap.release()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def open_source(
    path: str | Path,
    start_time: str | None = None,
    name: str | None = None,
    fps: float | None = None,
):
    """
    Return the right loader for ``path``.

    * A video file (by extension) → :class:`VideoLoader`.
    * A directory → :class:`ImageSequenceLoader` (MOT17 or plain image folder).
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return VideoLoader(p, start_time=start_time, name=name, fps=fps)
    if p.is_dir():
        return ImageSequenceLoader(p, start_time=start_time, name=name, fps=fps)
    if not p.exists():
        raise FileNotFoundError(f"Input source not found: {p}")
    raise ValueError(
        f"Unsupported input: {p}. Provide an image folder or a video file "
        f"({', '.join(sorted(VIDEO_EXTS))})."
    )
