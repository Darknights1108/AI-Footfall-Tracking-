# 🚶 AI Footfall Tracking & Retail Analytics Dashboard

> CCTV-style people tracking and footfall analytics built on the public **MOT17**
> pedestrian dataset — detect people, track each with a unique ID, count in/out
> movement across a virtual line, store events in SQLite, and explore the results
> in an interactive Streamlit dashboard.

Built as a portfolio project demonstrating an end-to-end computer-vision
analytics pipeline relevant to retail people-counting systems.

---

## 📌 Project Overview

Physical retailers need to understand **how many people enter and leave a store**,
**when foot traffic peaks**, and **how busy the space is at any moment**. This
project reproduces that pipeline end-to-end using off-the-shelf, pretrained
computer-vision models — no training required:

**Detect → Track → Count → Store → Analyse → Visualise**

| Stage | Technology |
|-------|-----------|
| Detection | Ultralytics **YOLO11** (person class only) |
| Tracking | **ByteTrack** (stable per-person IDs) |
| Counting | **Roboflow Supervision** `LineZone` (in/out across a virtual line) |
| Storage | **SQLite** (footfall events + run summaries) |
| Analytics | **Pandas** (totals, occupancy, real-timeline trends, alerts) |
| Dashboard | **Streamlit** + **Plotly** |

---

## ❓ Problem Statement

Retail teams make staffing, layout, and marketing decisions based on footfall.
Manual counting is expensive and error-prone. An automated system should:

- Count visitors entering and leaving through an entrance.
- Maintain a live **occupancy** figure (people currently inside).
- Reveal **peak hours** and traffic trends.
- Raise **operational alerts** (overcrowding, unusually low traffic).

This project delivers exactly that from ordinary camera footage.

---

## 🏢 Why This Project Is Relevant to FootfallCam

[FootfallCam](https://www.footfallcam.com/) builds people-counting and retail
analytics systems. This project mirrors the core of that domain:

- **People counting via computer vision** — the fundamental FootfallCam capability.
- **In/out direction counting and live occupancy** — standard footfall metrics.
- **Business-facing analytics** — peak hours, trends, and occupancy alerts, not
  just raw detections.
- **A practical engineering pipeline** — modular detection/tracking/counting,
  a persistence layer, and a dashboard — the shape of a real product.

It demonstrates the ability to take a CV model from research into an analytics
product that a retail stakeholder could actually read.

---

## 🎥 Why MOT17 Is Used

> **This project uses the public MOT17 pedestrian tracking dataset for demo
> purposes. Real retail CCTV and sales data are usually private, so MOT17 is used
> to simulate CCTV-style people tracking and footfall counting.**

[MOT17](https://motchallenge.net/data/MOT17/) is a standard multi-object-tracking
benchmark containing CCTV-style pedestrian video sequences. It provides realistic
crowd movement for validating tracking and counting without needing private store
footage. The clips are short (~15–35 s), so the dashboard plots activity against
the **real video timeline** (elapsed seconds) — it does **not** fabricate a
"business day" clock. In a production CCTV deployment this axis would simply be
the camera's real timestamps.

---

## 🏗️ System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              process_mot17.py (CLI)          │
                    └─────────────────────────────────────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   ┌───────────────────┐   ┌───────────────────┐    ┌────────────────────┐
   │  mot17_loader.py  │   │    detector.py    │    │     tracker.py     │
   │  read img1 frames │──▶│  YOLO11 (person)  │──▶ │  ByteTrack (IDs)   │
   └───────────────────┘   └───────────────────┘    └────────────────────┘
                                                                │
                                                                ▼
   ┌───────────────────┐   ┌───────────────────┐    ┌────────────────────┐
   │   dashboard.py    │◀──│    database.py    │◀── │     counter.py     │
   │    (Streamlit)    │   │      (SQLite)     │    │ Supervision LineZone│
   └───────────────────┘   └───────────────────┘    └────────────────────┘
             ▲                          ▲
             │                          │
      ┌──────────────┐         ┌────────────────────┐
      │ analytics.py │         │ video_processor.py │
      │  (Pandas)    │         │  (orchestration +  │
      └──────────────┘         │  annotated video)  │
                               └────────────────────┘
```

---

## 🔄 Data Flow

```
MOT17 image sequence (img1/*.jpg)
        │
        ▼
YOLO11 detects person class only
        │
        ▼
ByteTrack assigns a stable track_id to each person
        │
        ▼
Supervision LineZone detects track_id crossing the virtual line
        │
        ▼
Each valid crossing → footfall_events row in SQLite (in / out)
        │
        ▼
Streamlit dashboard reads SQLite
        │
        ▼
Total In · Total Out · Occupancy · Crossings-over-time · Alerts · Insight
```

---

## 🧰 Tech Stack

- **Python 3.13**
- **Ultralytics YOLO11** — person detection (`yolo11n.pt`)
- **ByteTrack** (via Supervision) — multi-object tracking
- **Roboflow Supervision** — `LineZone` counting + annotators
- **OpenCV** — image/video processing
- **SQLite** — event storage
- **Pandas** — analytics aggregation
- **Streamlit + Plotly** — dashboard and charts
- **PyTorch (CUDA 12.8)** — GPU inference (RTX 50-series / Blackwell)

---

## ✨ Features

- **MOT17 sequence loader** — reads `img1/` frames directly (no mp4 conversion),
  parses `seqinfo.ini` for FPS/resolution, processes frames in order, and
  attaches a simulated timestamp to each.
- **Person-only detection** — YOLO11 filtered to the COCO person class with a
  configurable confidence threshold.
- **Stable multi-object tracking** — ByteTrack unique IDs feeding the counter.
- **Line-crossing in/out counting** — horizontal or vertical virtual line, with
  direction classification and built-in per-track de-duplication.
- **SQLite persistence** — `footfall_events` and `processing_runs` tables.
- **Real video timeline** — every temporal chart uses the true elapsed time in
  the clip (`frame_index ÷ fps`); no faked business-day clock.
- **Analytics** — totals, occupancy, unique visitors, crossings-over-time,
  running occupancy curve, busiest moment, overcrowding and low-traffic alerts,
  and a plain-English insight summary.
- **Streamlit dashboard** — KPI cards, charts, recent-events table, and alerts.
- **Annotated video export (optional)** — boxes, track IDs, counting line, and a
  live In/Out/Occupancy overlay.

---

## 📁 Project Structure

```
ai-footfall-analytics/
├── app/
│   └── dashboard.py          # Streamlit dashboard
├── src/
│   ├── config.py             # paths, thresholds, defaults
│   ├── mot17_loader.py       # MOT17 sequence loader
│   ├── detector.py           # YOLO11 person detection
│   ├── tracker.py            # ByteTrack tracking
│   ├── counter.py            # Supervision LineZone in/out counting
│   ├── database.py           # SQLite persistence
│   ├── analytics.py          # Pandas analytics functions
│   └── video_processor.py    # end-to-end orchestration + video export
├── scripts/
│   ├── process_mot17.py      # CLI: process a sequence
│   └── reset_db.py           # CLI: reset the database
├── data/                     # footfall.db (created at runtime)
├── outputs/annotated_videos/ # exported annotated videos
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📦 Dataset Setup Guide

1. Download the **MOT17** dataset from
   [motchallenge.net/data/MOT17](https://motchallenge.net/data/MOT17/).
2. Place it so the sequences are reachable as `MOT17/train/<SEQUENCE>/`. The
   loader expects this layout:

   ```
   MOT17/
   └── train/
       ├── MOT17-02-FRCNN/
       │   ├── img1/
       │   │   ├── 000001.jpg
       │   │   └── ...
       │   ├── seqinfo.ini
       │   ├── det/
       │   └── gt/
       ├── MOT17-04-FRCNN/
       ├── MOT17-09-FRCNN/
       └── MOT17-11-FRCNN/
   ```

3. If your dataset lives elsewhere, update `MOT17_ROOT` in `src/config.py` or
   pass an absolute `--sequence` path.

> The dataset is large (~5.7 GB) and is **not** committed to the repo (see
> `.gitignore`).

---

## ⚙️ Installation

```bash
# 1. Create and activate a virtual environment (Python 3.13 recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install PyTorch first (choose ONE)
#    GPU — NVIDIA RTX 50-series (Blackwell) needs CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#    CPU-only:
# pip install torch torchvision

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

The YOLO11 weights (`yolo11n.pt`) download automatically on first run.

---

## ▶️ How to Run MOT17 Processing

```bash
python scripts/process_mot17.py \
  --sequence MOT17/train/MOT17-09-FRCNN \
  --line horizontal \
  --export-video
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `--sequence` | Path to a MOT17 sequence | *(required)* |
| `--line` | `horizontal` or `vertical` counting line | `horizontal` |
| `--line-position` | Line position as a fraction of the frame (0–1) | `0.5` |
| `--swap-in-out` | Swap the in/out direction mapping | off |
| `--start-time` | Clock anchor for timestamps (real elapsed time is added on top) | `2026-07-09 10:00:00` |
| `--conf` | Detection confidence threshold | `0.30` |
| `--export-video` | Also write an annotated mp4 | off |

The script prints a final summary (sequence, frames, total in/out, occupancy,
unique tracks, output video path) and writes events to `data/footfall.db`.

### Choosing a sequence and line orientation

Counting only works when the **line orientation matches the pedestrian flow** —
people must move *across* the line, not along it. Recommended demo combinations
(all static-camera sequences):

| Sequence | Camera | Best line | Notes |
|----------|--------|-----------|-------|
| `MOT17-09-FRCNN` | Static, street-level | `horizontal` | People walk toward/away — richest demo |
| `MOT17-02-FRCNN` | Static, street-level | `horizontal` | Moderate flow |
| `MOT17-04-FRCNN` | Static, elevated | `vertical` | People move left↔right across the street |

Moving-camera sequences (`MOT17-05/10/11/13`) are **not suitable** for a fixed
counting line and will under-count.

To wipe the database and start fresh:

```bash
python scripts/reset_db.py
```

---

## 📊 How to Run the Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard shows, per sequence:

- **KPI cards** — Total In, Total Out, Current Occupancy, Unique Tracked Persons,
  Video Length (real clip duration).
- **Detection preview** — annotated frames showing boxes, track IDs and the line.
- **Crossings over video time** (bar chart) and **Occupancy over video time**
  (area chart), both on the real elapsed-seconds timeline.
- **Alerts** — overcrowding and low-traffic warnings (thresholds adjustable in
  the sidebar).
- **Business insight** — a plain-English summary of the clip's traffic.
- **Recent crossing events** — timestamp, frame index, track ID, direction.

If the database is empty, the dashboard explains how to process a sequence first.

---

## 🖼️ Example Screenshots

*(Add screenshots after your first run.)*

| Annotated tracking | Dashboard |
|--------------------|-----------|
| `outputs/annotated_videos/MOT17-09-FRCNN_output.mp4` | `streamlit run app/dashboard.py` |

Suggested captures:

- Annotated frame showing boxes, track IDs, the counting line, and the
  In/Out/Occupancy overlay.
- Dashboard with KPI cards and charts populated.

---

## ⚠️ Limitations

- **MOT17 is not real retail store data** — it is used to simulate CCTV-style
  people tracking; results are illustrative, not commercial ground truth.
- **Counting accuracy depends on camera angle and line placement** — a poorly
  positioned line or steep angle reduces reliability.
- **A pretrained YOLO model is used** instead of custom training, so detection is
  not tuned to any specific store environment.
- **Short MOT17 clips (~15–35 s)** — charts therefore span seconds, not hours.
  Time is shown as the **real elapsed video timeline**; the project deliberately
  does not fabricate a "business day" clock, so trends are demonstrative of the
  method rather than a full retail day.
- **Line jitter can over-count** — a person lingering exactly on the line may be
  counted several times (centre-anchor crossings); a production system would add
  a dwell/debounce guard.
- **No cross-camera re-identification** — a person leaving one camera and
  entering another is treated as a new track.

---

## 🚀 Future Improvements

- **Real-time CCTV camera input** (RTSP/webcam streaming).
- **Multi-camera support** with a unified store view.
- **Person re-identification** across cameras.
- **Sales conversion analytics** (footfall vs transactions).
- **Staff allocation recommendations** from predicted peak hours.
- **Occupancy heatmaps** and zone dwell-time analysis.
- **Model fine-tuning** with CrowdHuman or WiderPerson for denser scenes.

---

## 📝 Notes

- **In/Out direction:** `LineZone` derives direction from the line's geometry. If
  in/out comes out reversed for a given view, use `--swap-in-out` (or set
  `SWAP_IN_OUT` in `config.py`) — no need to redraw the line.
- All tunables (paths, thresholds, line defaults, alert limits) live in
  `src/config.py`.
