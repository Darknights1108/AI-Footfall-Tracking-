"""
Person detection using a pretrained Ultralytics YOLO11 model.

Only the COCO ``person`` class is kept; all other classes are ignored. The raw
YOLO result is converted to a ``supervision.Detections`` object so it flows
cleanly into the tracker and line-counter.
"""

from __future__ import annotations

import supervision as sv
from ultralytics import YOLO

from . import config


class PersonDetector:
    """
    Wrap YOLO11 for person-only detection.

    Parameters
    ----------
    model_name:
        Weights file name (default ``yolo11n.pt``, auto-downloaded on first use).
    conf_threshold:
        Minimum confidence to keep a detection.
    device:
        "cuda" or "cpu". Defaults to whatever ``config.resolve_device()`` finds.
    """

    def __init__(
        self,
        model_name: str | None = None,
        conf_threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or config.MODEL_NAME
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
        )
        self.device = device or config.resolve_device()
        self.model = YOLO(self.model_name)

    def detect(self, frame) -> sv.Detections:
        """
        Run detection on a single BGR frame and return person detections.

        Filtering to the person class is done at inference time via the
        ``classes`` argument for efficiency.
        """
        results = self.model(
            frame,
            classes=[config.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(results)
