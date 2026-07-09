"""
Streamlit dashboard: AI Footfall Tracking & Retail Analytics.

Reads footfall events from SQLite and presents KPIs, real-timeline trends, an in/out
breakdown, a recent-events table, alerts, and a business-insight summary.

Run with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import analytics, config, database

st.set_page_config(
    page_title="AI Footfall Tracking & Retail Analytics",
    page_icon="🚶",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Data access (cached briefly so the UI stays responsive)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=10)
def get_sequences() -> list[str]:
    """Return sequences ordered by activity (most crossing events first) so the
    dashboard opens on the most interesting one by default."""
    if not database.table_exists("footfall_events"):
        return []
    names = database.fetch_sequences()
    counts = {n: len(database.fetch_events(n)) for n in names}
    return sorted(names, key=lambda n: counts[n], reverse=True)


@st.cache_data(ttl=10)
def get_events(sequence_name: str | None):
    return analytics.load_events_df(sequence_name)


@st.cache_data(ttl=10)
def get_run(sequence_name: str):
    row = database.fetch_latest_run(sequence_name)
    return dict(row) if row is not None else None


def annotated_video_path(sequence_name: str) -> Path:
    """Location of the annotated demo video for a sequence (may not exist)."""
    return config.ANNOTATED_VIDEO_DIR / f"{sequence_name}_output.mp4"


@st.cache_data(ttl=60)
def get_preview_frames(video_path: str, n: int = 3):
    """
    Grab ``n`` evenly-spaced frames from the annotated video for display.

    Frames are read with OpenCV (which decodes the mp4v codec fine) and returned
    as RGB arrays, so they render reliably via ``st.image`` regardless of whether
    the browser can play the raw video inline.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        # Sample from the middle band where foot traffic is usually busiest.
        positions = [int(total * f) for f in (0.35, 0.55, 0.75)][:n]
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok:
                frames.append((pos, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🚶 AI Footfall Tracking & Retail Analytics Dashboard")
st.caption(
    "People tracking and footfall analytics on the MOT17 dataset — "
    "YOLO11 + ByteTrack + Supervision LineZone."
)

sequences = get_sequences()

# Empty-database guard: guide the user instead of crashing.
if not sequences:
    st.info(
        "No footfall data found yet.\n\n"
        "Process a MOT17 sequence first, for example:\n\n"
        "```\npython scripts/process_mot17.py "
        "--sequence MOT17/train/MOT17-09-FRCNN --line horizontal --export-video\n```"
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Dataset / sequence", options=sequences)
    st.markdown("---")
    st.markdown("**Alert thresholds**")
    occ_threshold = st.number_input(
        "Overcrowding occupancy >",
        min_value=1,
        value=int(config.OCCUPANCY_ALERT_THRESHOLD),
    )
    low_threshold = st.number_input(
        "Low traffic visitors <",
        min_value=1,
        value=int(config.LOW_TRAFFIC_THRESHOLD),
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

df = get_events(selected)

if df.empty:
    st.warning("No events recorded for this sequence.")
    st.stop()


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
st.subheader(f"Sequence: {selected}")

total_in = analytics.total_in(df)
total_out = analytics.total_out(df)
occupancy = analytics.current_occupancy(df)

# Run summary gives us the real fps and total tracked count.
run = get_run(selected)
fps = run["fps"] if run and run.get("fps") else config.DEFAULT_FPS
unique_tracked = run["total_unique_tracks"] if run else analytics.unique_tracks(df)
unique_crossers = analytics.unique_tracks(df)

# Real clip length (seconds) — an honest, factual figure, not a faked clock.
duration_s = (
    analytics.clip_duration_seconds(run["total_frames"], fps) if run else 0.0
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total In", total_in)
c2.metric("Total Out", total_out)
c3.metric("Current Occupancy", occupancy)
c4.metric(
    "Unique Tracked Persons",
    unique_tracked,
    help=f"{unique_crossers} of them crossed the counting line",
)
c5.metric(
    "Video Length",
    f"{duration_s:.1f}s",
    help="Real duration of the processed clip (frames ÷ fps)",
)


# --------------------------------------------------------------------------- #
# Detection preview — shows HOW the model detects & counts people
# --------------------------------------------------------------------------- #
video_path = annotated_video_path(selected)
with st.expander("🎥 Detection Preview — how the model detects & counts people", expanded=True):
    if not video_path.exists():
        st.info(
            "No annotated video for this sequence yet. Re-run processing with "
            "`--export-video` to generate one, e.g.\n\n"
            f"```\npython scripts/process_mot17.py "
            f"--sequence MOT17/train/{selected} --line horizontal --export-video\n```"
        )
    else:
        st.caption(
            "Purple boxes = detected people (YOLO11) · #N = track ID (ByteTrack) · "
            "white line = virtual counting line · in/out overlay = live count."
        )
        frames = get_preview_frames(str(video_path))
        if frames:
            cols = st.columns(len(frames))
            for col, (pos, img) in zip(cols, frames):
                col.image(img, caption=f"Frame {pos}", use_container_width=True)
        # Offer the full annotated clip for download (mp4v may not play inline).
        with open(video_path, "rb") as fh:
            st.download_button(
                "⬇️ Download full annotated video",
                data=fh,
                file_name=video_path.name,
                mime="video/mp4",
            )


# --------------------------------------------------------------------------- #
# Charts — plotted against the REAL video timeline (elapsed seconds), never a
# faked "hour of day". The x-axis is the true clip time (frame_index ÷ fps).
# --------------------------------------------------------------------------- #
st.caption(
    "⏱ Charts use the **real video timeline** (elapsed seconds in the clip), "
    "not simulated business hours. In a production CCTV deployment this axis "
    "would be the camera's real timestamps."
)

timeline = analytics.crossings_over_time(df, fps)
occ_curve = analytics.occupancy_over_time(df, fps)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Crossings Over Video Time")
    bar = px.bar(
        timeline,
        x="t",
        y=["in", "out"],
        barmode="group",
        labels={"t": "Elapsed video time (s)", "value": "People", "variable": "Direction"},
    )
    bar.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, use_container_width=True)

with col_right:
    st.markdown("#### Occupancy Over Video Time")
    occ_fig = go.Figure()
    occ_fig.add_trace(
        go.Scatter(
            x=occ_curve["t"],
            y=occ_curve["occupancy"],
            mode="lines",
            name="Occupancy",
            line=dict(width=3, shape="hv"),
            fill="tozeroy",
        )
    )
    occ_fig.update_layout(
        xaxis_title="Elapsed video time (s)",
        yaxis_title="People inside",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(occ_fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
st.markdown("#### Alerts")
over_flag, over_msg = analytics.overcrowding_alert(df, occ_threshold)
low_flag, low_msg = analytics.low_traffic_alert(df, low_threshold)

a1, a2 = st.columns(2)
with a1:
    (st.error if over_flag else st.success)(over_msg)
with a2:
    (st.warning if low_flag else st.success)(low_msg)


# --------------------------------------------------------------------------- #
# Business insight
# --------------------------------------------------------------------------- #
st.markdown("#### Business Insight")
st.info(analytics.business_insight_summary(df, fps))


# --------------------------------------------------------------------------- #
# Recent events
# --------------------------------------------------------------------------- #
st.markdown("#### Recent Crossing Events")
recent = analytics.recent_events(df, fps, limit=20)
st.dataframe(recent, use_container_width=True, hide_index=True)
