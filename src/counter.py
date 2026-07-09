"""
Line-crossing in/out counter built on ``supervision.LineZone``.

A virtual counting line is placed across the frame. As tracked people cross it,
Supervision's ``LineZone.trigger`` reports which detections crossed "in" and
which crossed "out" on this frame — with per-track de-duplication handled
internally. This module turns those crossings into structured event records.
"""

from __future__ import annotations

from dataclasses import dataclass

import supervision as sv

from . import config


@dataclass
class CrossingEvent:
    """A single validated line-crossing by one tracked person."""

    frame_index: int
    track_id: int
    direction: str  # "in" or "out"


def build_line_zone(
    frame_width: int,
    frame_height: int,
    orientation: str = "horizontal",
    position: float = 0.5,
) -> sv.LineZone:
    """
    Create a ``LineZone`` spanning the frame.

    * ``horizontal`` — a horizontal line at ``height * position`` (people cross
      it moving up/down).
    * ``vertical`` — a vertical line at ``width * position`` (people cross it
      moving left/right).
    """
    if orientation == "horizontal":
        y = int(frame_height * position)
        start = sv.Point(0, y)
        end = sv.Point(frame_width, y)
    elif orientation == "vertical":
        x = int(frame_width * position)
        start = sv.Point(x, 0)
        end = sv.Point(x, frame_height)
    else:
        raise ValueError(
            f"Unknown line orientation '{orientation}' (use 'horizontal' or 'vertical')."
        )

    # By default LineZone requires ALL FOUR box corners to cross before counting,
    # which is far too strict for people-counting (a person straddling the line
    # never counts). Trigger on a single anchor — the box centre — so a crossing
    # registers as soon as the person's midpoint passes the line, which is the
    # standard behaviour for footfall counters.
    return sv.LineZone(
        start=start,
        end=end,
        triggering_anchors=(sv.Position.CENTER,),
    )


class LineCrossingCounter:
    """
    Count people crossing a virtual line and classify each crossing as in/out.

    Parameters
    ----------
    frame_width, frame_height:
        Frame dimensions used to place the line.
    orientation, position:
        Line geometry (see :func:`build_line_zone`).
    swap_in_out:
        If True, swap the "in"/"out" labels. LineZone derives direction from the
        line's geometry; flip this when the mapping is reversed for a camera.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        orientation: str | None = None,
        position: float | None = None,
        swap_in_out: bool | None = None,
    ) -> None:
        self.orientation = orientation or config.LINE_ORIENTATION
        self.position = position if position is not None else config.LINE_POSITION
        self.swap_in_out = (
            swap_in_out if swap_in_out is not None else config.SWAP_IN_OUT
        )
        self.line_zone = build_line_zone(
            frame_width, frame_height, self.orientation, self.position
        )

    def update(self, detections: sv.Detections, frame_index: int) -> list[CrossingEvent]:
        """
        Feed the current frame's tracked detections through the line zone.

        Returns a list of :class:`CrossingEvent` for crossings that happened on
        this frame (usually empty). ``LineZone.trigger`` prevents a given track
        from being counted twice for the same crossing.
        """
        crossed_in, crossed_out = self.line_zone.trigger(detections)

        events: list[CrossingEvent] = []
        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            return events

        for i, track_id in enumerate(tracker_ids):
            if crossed_in[i]:
                events.append(
                    CrossingEvent(frame_index, int(track_id), self._label("in"))
                )
            if crossed_out[i]:
                events.append(
                    CrossingEvent(frame_index, int(track_id), self._label("out"))
                )
        return events

    def _label(self, raw_direction: str) -> str:
        """Apply the swap flag to a raw LineZone direction."""
        if not self.swap_in_out:
            return raw_direction
        return "out" if raw_direction == "in" else "in"

    # Convenience accessors for the running totals kept by LineZone. -------- #
    @property
    def total_in(self) -> int:
        raw_in = self.line_zone.in_count
        raw_out = self.line_zone.out_count
        return raw_out if self.swap_in_out else raw_in

    @property
    def total_out(self) -> int:
        raw_in = self.line_zone.in_count
        raw_out = self.line_zone.out_count
        return raw_in if self.swap_in_out else raw_out
