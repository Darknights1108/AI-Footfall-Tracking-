"""
Multi-object tracking with ByteTrack (via the Supervision implementation).

ByteTrack assigns a stable ``tracker_id`` to each detected person across frames.
Using ``supervision.ByteTrack`` keeps detection, tracking, and counting as three
clean, separately testable stages while still being ByteTrack under the hood.
"""

from __future__ import annotations

import warnings

import supervision as sv


class PersonTracker:
    """Thin wrapper around ``supervision.ByteTrack``."""

    TRACKER_NAME = "ByteTrack"

    def __init__(self, frame_rate: float = 30.0) -> None:
        # ``sv.ByteTrack`` is marked deprecated from supervision 0.28 (bundled
        # trackers are being removed in 0.30) but is fully functional in the
        # pinned 0.29.x range we depend on. Silence just that FutureWarning so
        # the console stays clean; requirements pin supervision < 0.30.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # ByteTrack's internal track buffer is expressed in frames, so
            # passing the real frame rate keeps its time-based logic consistent.
            try:
                self.tracker = sv.ByteTrack(frame_rate=int(round(frame_rate)))
            except TypeError:
                # Older/newer signatures may not accept frame_rate — fall back.
                self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update the tracker with the current frame's detections.

        Returns the same detections annotated with ``tracker_id`` for every
        successfully tracked box (untracked boxes are dropped by ByteTrack).
        """
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Reset tracker state between sequences (if supported)."""
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
