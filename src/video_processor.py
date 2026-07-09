"""
End-to-end video processing engine.

Ties the pipeline together for a single MOT17 sequence:

    load frames -> detect people -> track -> count line crossings
                -> persist events + run summary -> (optional) annotated mp4

Both the CLI (``scripts/process_mot17.py``) and any other caller use
:func:`process_sequence`. Annotation uses Supervision annotators, selected
defensively so the code works across a range of Supervision versions.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from . import config, database
from .counter import LineCrossingCounter
from .detector import PersonDetector
from .mot17_loader import Mot17SequenceLoader
from .tracker import PersonTracker


# --------------------------------------------------------------------------- #
# Annotator selection (handles Supervision API differences across versions)
# --------------------------------------------------------------------------- #
def _make_box_annotator():
    """Return a box annotator, preferring the current class name."""
    if hasattr(sv, "BoxAnnotator"):
        return sv.BoxAnnotator()
    # Older Supervision releases named it BoundingBoxAnnotator.
    return sv.BoundingBoxAnnotator()


class _Annotators:
    """Bundle the Supervision annotators used to draw a frame."""

    def __init__(self, line_zone: sv.LineZone) -> None:
        self.box = _make_box_annotator()
        self.label = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        self.trace = sv.TraceAnnotator(thickness=2, trace_length=30)
        self.line = sv.LineZoneAnnotator(
            thickness=2, text_thickness=2, text_scale=0.7
        )
        self.line_zone = line_zone

    def draw(self, frame, detections: sv.Detections):
        """Draw traces, boxes, per-track labels, and the counting line."""
        annotated = frame.copy()
        annotated = self.trace.annotate(annotated, detections)
        annotated = self.box.annotate(annotated, detections)

        labels = _build_labels(detections)
        annotated = self.label.annotate(annotated, detections, labels)
        annotated = self.line.annotate(annotated, line_counter=self.line_zone)
        return annotated


def _build_labels(detections: sv.Detections) -> list[str]:
    """Produce '#<track_id>' labels for each detection."""
    if detections.tracker_id is None:
        return ["person" for _ in range(len(detections))]
    return [f"#{int(tid)}" for tid in detections.tracker_id]


def _draw_overlay(frame, total_in: int, total_out: int, seq_name: str):
    """Draw an In/Out/Occupancy status panel in the top-left corner."""
    occupancy = max(total_in - total_out, 0)
    lines = [
        f"Sequence: {seq_name}",
        f"IN:  {total_in}",
        f"OUT: {total_out}",
        f"Occupancy: {occupancy}",
    ]
    # Semi-transparent background box for readability.
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 20 + 28 * len(lines)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 40 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def process_sequence(
    sequence_path: str | Path,
    line_orientation: str | None = None,
    line_position: float | None = None,
    swap_in_out: bool | None = None,
    start_time: str | None = None,
    conf_threshold: float | None = None,
    export_video: bool = False,
    output_path: str | Path | None = None,
    db_path: str | Path | None = None,
    progress_every: int = 50,
    log: bool = True,
) -> dict:
    """
    Process one MOT17 sequence end-to-end.

    Returns a summary dict:
    ``{sequence_name, total_frames, total_in, total_out, occupancy,
       unique_tracks, output_video}``.
    """
    loader = Mot17SequenceLoader(sequence_path, start_time=start_time)
    info = loader.info

    detector = PersonDetector(conf_threshold=conf_threshold)
    tracker = PersonTracker(frame_rate=info.frame_rate)
    counter = LineCrossingCounter(
        frame_width=info.width,
        frame_height=info.height,
        orientation=line_orientation,
        position=line_position,
        swap_in_out=swap_in_out,
    )

    # Prepare the database and clear any prior data for this sequence so a
    # re-run replaces its events instead of appending duplicates.
    database.init_db(db_path)
    database.delete_sequence(info.name, db_path)

    # Set up the video writer only if requested.
    writer = None
    annotators = None
    resolved_output = None
    if export_video:
        annotators = _Annotators(counter.line_zone)
        resolved_output = _resolve_output_path(output_path, info.name)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(resolved_output),
            fourcc,
            info.frame_rate,
            (info.width, info.height),
        )

    event_rows: list[tuple] = []
    seen_tracks: set[int] = set()
    frames_processed = 0

    for frame_data in loader:
        frames_processed += 1

        detections = detector.detect(frame_data.image)
        detections = tracker.update(detections)

        if detections.tracker_id is not None:
            seen_tracks.update(int(t) for t in detections.tracker_id)

        crossings = counter.update(detections, frame_data.frame_index)
        for ev in crossings:
            event_rows.append(
                (
                    info.name,
                    ev.frame_index,
                    frame_data.timestamp.strftime(config.TIMESTAMP_FORMAT),
                    ev.track_id,
                    ev.direction,
                    config.LINE_NAME,
                )
            )

        if writer is not None:
            annotated = annotators.draw(frame_data.image, detections)
            annotated = _draw_overlay(
                annotated, counter.total_in, counter.total_out, info.name
            )
            writer.write(annotated)

        if log and progress_every and frames_processed % progress_every == 0:
            print(
                f"  frame {frames_processed}/{len(loader)}  "
                f"in={counter.total_in} out={counter.total_out}",
                flush=True,
            )

    if writer is not None:
        writer.release()

    # Persist events and the run summary.
    database.insert_events_bulk(event_rows, db_path)
    total_in = counter.total_in
    total_out = counter.total_out
    unique = len(seen_tracks)
    database.insert_run(
        sequence_name=info.name,
        model_name=detector.model_name,
        tracker_name=tracker.TRACKER_NAME,
        total_in=total_in,
        total_out=total_out,
        total_unique_tracks=unique,
        total_frames=frames_processed,
        fps=info.frame_rate,
        db_path=db_path,
    )

    return {
        "sequence_name": info.name,
        "total_frames": frames_processed,
        "total_in": total_in,
        "total_out": total_out,
        "occupancy": max(total_in - total_out, 0),
        "unique_tracks": unique,
        "output_video": str(resolved_output) if resolved_output else None,
    }


def _resolve_output_path(output_path, sequence_name: str) -> Path:
    """Default the annotated-video path to outputs/annotated_videos/<name>_output.mp4."""
    if output_path is not None:
        return Path(output_path)
    return config.ANNOTATED_VIDEO_DIR / f"{sequence_name}_output.mp4"
