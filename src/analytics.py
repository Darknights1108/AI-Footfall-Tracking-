"""
Footfall analytics.

Pure functions that turn the raw ``footfall_events`` table into the metrics the
dashboard displays: totals, occupancy, real-timeline trends, busiest moment, and a
natural-language business-insight summary. Everything operates on a pandas
DataFrame so the same functions serve both the CLI and Streamlit.
"""

from __future__ import annotations

import pandas as pd

from . import config, database


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_events_df(
    sequence_name: str | None = None, db_path=None
) -> pd.DataFrame:
    """
    Load footfall events into a DataFrame with a parsed ``timestamp`` column.
    Returns an empty (but correctly-typed) frame when there are no events.
    """
    columns = [
        "id",
        "sequence_name",
        "frame_index",
        "timestamp",
        "track_id",
        "direction",
        "line_name",
        "created_at",
    ]
    if not database.table_exists("footfall_events", db_path):
        return pd.DataFrame(columns=columns)

    rows = database.fetch_events(sequence_name, db_path)
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame([dict(row) for row in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def total_in(df: pd.DataFrame) -> int:
    """Total number of "in" crossings."""
    return int((df["direction"] == "in").sum()) if not df.empty else 0


def total_out(df: pd.DataFrame) -> int:
    """Total number of "out" crossings."""
    return int((df["direction"] == "out").sum()) if not df.empty else 0


def current_occupancy(df: pd.DataFrame) -> int:
    """People currently inside = total in − total out (never below zero)."""
    return max(total_in(df) - total_out(df), 0)


def unique_tracks(df: pd.DataFrame) -> int:
    """Number of distinct people (track ids) that crossed the line."""
    return int(df["track_id"].nunique()) if not df.empty else 0


# --------------------------------------------------------------------------- #
# Real video-timeline analytics
#
# The x-axis here is the TRUE elapsed time within the clip (frame_index / fps),
# not a faked "hour of day". Nothing is misrepresented as real business hours.
# --------------------------------------------------------------------------- #
def elapsed_seconds(df: pd.DataFrame, fps: float) -> pd.Series:
    """Real seconds since the first frame for each event, from its frame index."""
    fps = fps or config.DEFAULT_FPS
    return (df["frame_index"] - 1) / fps


def crossings_over_time(
    df: pd.DataFrame, fps: float, bin_seconds: float | None = None
) -> pd.DataFrame:
    """
    In/out crossings bucketed along the real video timeline.

    Returns a DataFrame with columns ``t`` (bucket start, in seconds), ``in``,
    ``out``, ``net`` and ``total`` — one row per non-empty time bucket.
    """
    cols = ["t", "in", "out", "net", "total"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    bin_seconds = bin_seconds or config.TIMELINE_BIN_SECONDS
    work = df.copy()
    secs = elapsed_seconds(work, fps)
    work["t"] = (secs // bin_seconds) * bin_seconds  # bucket start in seconds

    grouped = (
        work.groupby(["t", "direction"]).size().unstack(fill_value=0).reset_index()
    )
    for col in ("in", "out"):
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["net"] = grouped["in"] - grouped["out"]
    grouped["total"] = grouped["in"] + grouped["out"]
    return grouped[cols].sort_values("t")


def occupancy_over_time(
    df: pd.DataFrame, fps: float, clip: bool = True
) -> pd.DataFrame:
    """
    Running occupancy after each crossing, along the real video timeline.

    Returns a DataFrame with columns ``t`` (seconds) and ``occupancy``.

    ``clip`` clamps occupancy at zero (a real space can't hold fewer than zero
    people). Pass ``clip=False`` to show the *raw* net (in − out), which reveals
    a systematic out > in bias as a downward drift instead of a flat line — used
    for unreliable sources so the chart is diagnostic rather than misleading.
    """
    cols = ["t", "occupancy"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    work = df.sort_values("frame_index").copy()
    work["t"] = elapsed_seconds(work, fps)
    step = work["direction"].map({"in": 1, "out": -1}).fillna(0)
    net = step.cumsum()
    work["occupancy"] = net.clip(lower=0) if clip else net
    return work[cols]


def counting_reliability(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Sanity-check the counting for a source.

    A real entrance that starts empty can never have cumulative "out" exceed
    cumulative "in". If the running net (in − out) drops well below zero, the
    counter is misbehaving for this source (usually track-ID fragmentation in a
    dense or overhead scene). Returns ``(is_reliable, message)``.
    """
    if df.empty:
        return True, ""

    work = df.sort_values("frame_index")
    net = work["direction"].map({"in": 1, "out": -1}).fillna(0).cumsum()
    min_net = int(net.min()) if len(net) else 0

    if min_net <= -config.COUNTING_UNRELIABLE_NET:
        return False, (
            f"Counting looks unreliable for this source: cumulative *out* exceeds "
            f"*in* by up to {-min_net} people, which is impossible for a real "
            f"entrance. This is typically track-ID fragmentation in a dense or "
            f"overhead scene. The occupancy chart below shows the raw net "
            f"(in − out) for diagnosis rather than a clamped, misleading flat line."
        )
    return True, ""


def peak_flow(
    df: pd.DataFrame, fps: float, window_seconds: float = 3.0
) -> tuple[float | None, int]:
    """
    Busiest moment: the time bucket with the most total crossings.

    Returns ``(bucket_start_seconds, total_crossings)`` or ``(None, 0)``.
    """
    buckets = crossings_over_time(df, fps, bin_seconds=window_seconds)
    if buckets.empty or buckets["total"].sum() == 0:
        return None, 0
    row = buckets.loc[buckets["total"].idxmax()]
    return float(row["t"]), int(row["total"])


def clip_duration_seconds(total_frames: int, fps: float) -> float:
    """Real length of the processed clip in seconds."""
    fps = fps or config.DEFAULT_FPS
    return total_frames / fps if fps else 0.0


def recent_events(
    df: pd.DataFrame, fps: float = config.DEFAULT_FPS, limit: int = 20
) -> pd.DataFrame:
    """
    Return the most recent crossing events (by frame index), annotated with the
    real elapsed video time in seconds rather than a faked wall clock.
    """
    if df.empty:
        return df
    work = df.copy()
    work["video_time_s"] = elapsed_seconds(work, fps).round(2)
    cols = ["video_time_s", "frame_index", "track_id", "direction"]
    return work.sort_values("frame_index", ascending=False).head(limit)[cols]


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
def overcrowding_alert(
    df: pd.DataFrame, threshold: int | None = None
) -> tuple[bool, str]:
    """Flag when current occupancy exceeds the configured threshold."""
    limit = threshold if threshold is not None else config.OCCUPANCY_ALERT_THRESHOLD
    occ = current_occupancy(df)
    if occ > limit:
        return True, (
            f"Overcrowding: current occupancy is {occ}, above the safe limit of {limit}."
        )
    return False, f"Occupancy normal ({occ} / {limit})."


def low_traffic_alert(
    df: pd.DataFrame, threshold: int | None = None
) -> tuple[bool, str]:
    """Flag when the number of unique visitors is below the configured floor."""
    limit = threshold if threshold is not None else config.LOW_TRAFFIC_THRESHOLD
    visitors = unique_tracks(df)
    if visitors < limit:
        return True, (
            f"Low traffic: only {visitors} unique visitors (below {limit})."
        )
    return False, f"Traffic healthy ({visitors} unique visitors)."


# --------------------------------------------------------------------------- #
# Business insight
# --------------------------------------------------------------------------- #
def business_insight_summary(df: pd.DataFrame, fps: float = config.DEFAULT_FPS) -> str:
    """Produce a short, human-readable summary sentence for the dashboard."""
    if df.empty:
        return "No footfall data available yet. Process a MOT17 sequence to begin."

    ti, to = total_in(df), total_out(df)
    occ = current_occupancy(df)
    visitors = unique_tracks(df)
    t_peak, peak_total = peak_flow(df, fps)

    base = (
        f"{visitors} people crossed the counting line "
        f"({ti} in, {to} out), leaving {occ} inside at the end of the clip."
    )
    if t_peak is not None:
        base += (
            f" Footfall was busiest around {t_peak:.0f}s into the video "
            f"with {peak_total} crossings in a 3s window."
        )
    return base
