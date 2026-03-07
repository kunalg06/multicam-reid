# 🎯 Multi-Camera Person Tracking & Re-Identification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-purple)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![SQLite](https://img.shields.io/badge/DB-SQLite-003b57?logo=sqlite)

> A complete multi-camera surveillance system with persistent person identities, cross-camera re-identification, and an operator dashboard with a **Lost Person Registry** that never auto-deletes.

**No torchreid. No Cython. No build errors. Pure pip install.**

---

## What It Does

```
Person walks into Cam0       → assigned GID-0001
Person walks to Cam1         → recognised as GID-0001  ✅ (same ID)
Person returns to Cam0       → recognised as GID-0001  ✅ (same ID)
Person doesn't appear for 2min → status = LOST  ⚠️
Operator opens dashboard     → sees GID-0001 in Lost Registry
                               → can add notes, resolve, or reactivate
GID-0001 is never deleted    → permanent record until operator resolves ✅
```

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                     pipeline.py                         │
│  (video files OR live RTSP/webcam — handles both)       │
└──────────────────┬─────────────────────────────────────┘
                   │
          GlobalTracker  ─────────────────────┐
          (orchestrates all cameras)           │
                   │                           │
    ┌──────────────┴──────────────┐            │
    │                             │            │
CameraWorker 0              CameraWorker 1    ...
    │                             │
    ▼                             ▼
YOLOv8 detect            YOLOv8 detect
ByteTrack track          ByteTrack track
MobileNetV3 embed        MobileNetV3 embed
    │                             │
    └──────────────┬──────────────┘
                   │
                   ▼
         IdentityStore  (SQLite)
         ┌─────────────────────────────────────────┐
         │  match_or_create(embedding)              │
         │    1. compare against ALL known persons  │
         │    2. cosine similarity > threshold?     │
         │       YES → return existing global_id    │
         │       NO  → create new GID-XXXX          │
         │  promote_lost()                          │
         │    absent > threshold? → status=LOST     │
         │    NEVER deleted unless operator says so │
         └─────────────────────────────────────────┘
                   │
                   ▼
         Streamlit Dashboard  (dashboard/app.py)
         ┌─────────────────────────────────────────┐
         │  📊 Overview   — live stats + alerts     │
         │  🟢 Active     — currently tracked       │
         │  🔴 Lost       — never-delete registry   │
         │  🔍 Search     — by ID / time / camera   │
         │  👤 Detail     — full history + crops    │
         └─────────────────────────────────────────┘
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/multicam-reid.git
cd multicam-reid

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. PyTorch (CPU)
pip install torch torchvision

# 4. Everything else
pip install -r requirements.txt

# Done. No build steps. No git clones of external repos.
```

---

## Usage

### Video files (offline)

```bash
python pipeline.py \
  --videos cam0.mp4 cam1.mp4 \
  --output results/ \
  --lost-threshold 120
```

### Live RTSP streams

```bash
python pipeline.py --live \
  --sources rtsp://192.168.1.10/stream1 rtsp://192.168.1.11/stream2 \
  --output results/ \
  --lost-threshold 60
```

### Webcam

```bash
python pipeline.py --live --sources 0 1 --output results/
```

### Mixed (file + webcam)

```bash
python pipeline.py --live --sources cam0.mp4 0 --output results/
```

### Open operator dashboard (any time, separate terminal)

```bash
streamlit run dashboard/app.py
# Opens in browser at http://localhost:8501
```

---

## All Options

```
--videos           Video file paths (offline mode)
--live             Live mode (use --sources)
--sources          RTSP URLs or webcam indices
--output           Output directory           [results]
--db               SQLite database path       [database/identities.db]
--yolo-model       yolov8n/s/m/l/x.pt        [yolov8n.pt]
--reid-weights     Fine-tuned ReID .pth       [optional]
--conf             Detection confidence       [0.35]
--sim-threshold    Identity match threshold   [0.60]
--lost-threshold   Seconds absent → LOST      [120]
--device           auto / cuda / cpu          [auto]
```

---

## Identity Match Threshold Guide

| `--sim-threshold` | Behaviour |
|-------------------|-----------|
| `0.50` | Permissive — more re-IDs, possible false matches |
| `0.60` | Balanced ← **recommended** |
| `0.70` | Strict — fewer re-IDs, very reliable |
| `0.75` | Very strict — only match highly similar appearances |

---

## Lost Person Registry

The registry is the core operator feature:

- Person not seen for `--lost-threshold` seconds → status changes `active → lost`
- They **never disappear** from the registry
- Dashboard shows last known crop, all cameras visited, full sighting log
- Operator actions:
  - **Add notes** — "Confirmed found at Gate 3"
  - **Resolve** — close the case (status → resolved)
  - **Reactivate** — if person reappears, re-open the case

Even `resolved` entries are kept in the database permanently for audit trails.

---

## Database Schema

```sql
persons
  global_id        TEXT  UNIQUE  -- GID-0001
  status           TEXT          -- active | lost | resolved
  first_seen_at    REAL          -- unix timestamp
  last_seen_at     REAL          -- unix timestamp
  last_camera_id   INTEGER
  embedding        BLOB          -- float32 (512,) serialised
  best_crop_path   TEXT          -- path to best crop image
  notes            TEXT          -- operator notes
  resolved_at      REAL          -- when resolved

sightings
  global_id        TEXT  FK
  camera_id        INTEGER
  frame_idx        INTEGER
  seen_at          REAL
  bbox             TEXT  JSON
  conf             REAL
  crop_path        TEXT
```

---

## Project Structure

```
multicam-reid/
├── pipeline.py                  ← entry point
├── tracker/
│   ├── global_tracker.py        ← multi-camera orchestrator
│   └── mot_tracker.py           ← (per-camera, legacy single-cam mode)
├── reid/
│   └── feature_extractor.py     ← MobileNetV3 (pure PyTorch)
├── database/
│   └── identity_store.py        ← SQLite global identity store
├── dashboard/
│   └── app.py                   ← Streamlit operator dashboard
├── eval/
│   └── metrics.py               ← MOTA / IDF1 / MOTP
├── videos/input/                ← put your videos here
└── requirements.txt
```

---

## References

- **YOLOv8**: Jocher et al. (2023) — Ultralytics
- **ByteTrack**: Zhang et al. (2022) — ECCV
- **MobileNetV3**: Howard et al. (2019) — ICCV
- **Market-1501 / DukeMTMC**: person ReID benchmark datasets

---

## License

MIT — see LICENSE.
