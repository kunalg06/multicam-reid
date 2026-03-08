 # 🎯 Multi-Camera Person Tracking & Re-Identification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-purple)
![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)
![SQLite](https://img.shields.io/badge/DB-SQLite-003b57?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A complete multi-camera surveillance system with **persistent cross-camera person identities**, **automatic re-entry detection**, a **Lost Person Registry** that never auto-deletes, and a live **Streamlit operator dashboard**.

This project is a ground-up reimplementation of research originally conducted as part of an **MSc dissertation in Computer Vision** at **Sheffield Hallam University** (supervisor: Dr. Jing Wang). The original dissertation explored Multi-Camera Multi-People Tracking and Re-Identification using YOLOv4, DeepSORT, and torchreid. This repository modernises that work with a fully pip-installable stack — no Cython, no build steps, no torchreid dependency.

---

## 📌 Relation to Original Dissertation Work

The academic foundation and problem framing of this project draws from:

> **Multi-Camera Person Tracking and Re-Identification**  
> MSc Dissertation, Sheffield Hallam University  
> Supervisor: Dr. Jing Wang  
> Reference implementation: [samihormi/Multi-Camera-Person-Tracking-and-Re-Identification](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification)

**What the original dissertation used:**

| Component | Original (Dissertation)   |
|-----------|---------------------------|
| Detection | YOLOv4 (Keras / Darknet)  |
| Tracking  | DeepSORT                  |
| Re-ID     | torchreid (OSNet / ResNet)|
| Dataset   | DukeMTMC-ReID, Market-1501|
| Results   | IDF1 = 50.7%, MOTA = 99.8%|

**What this reimplementation uses — and why it differs:**

| Component | This Repo | Why Changed |
|-----------|-----------|-------------|
| Detection | YOLOv8 (ultralytics) | Single pip install, significantly faster, better accuracy |
| Tracking | ByteTrack (built into ultralytics) | Handles occlusion better than DeepSORT |
| Re-ID backbone | MobileNetV3 (torchvision) | torchreid requires Cython compilation which fails on Python 3.10+; torchvision ships with PyTorch — zero build step |
| Identity store | SQLite (stdlib) | Enables persistent cross-session identity, lost person registry, audit trail |
| Dashboard | Streamlit | Operator-facing UI for lost person management |
| Dataset | Any MOTChallenge-format video | More general than fixed benchmark datasets |

**None of the code from the samihormi reference repository is used here.** The architectural concepts (detect → track → extract features → match across cameras) are the same as in the dissertation and are standard in the multi-camera ReID literature. The implementation is written entirely from scratch.

---

## What This System Does

```
Person detected on Cam0        →  assigned GID-0001
Person walks to Cam1           →  recognised as GID-0001  ✅ same ID
Person returns to Cam0         →  recognised as GID-0001  ✅ same ID
Person not seen for 2 minutes  →  status = LOST  ⚠️
Person reappears on any camera →  automatically restored to GID-0001  ✅
                                   reappearance event logged in DB
Operator opens Streamlit       →  sees Lost Registry, full event timeline,
                                   can resolve / add notes / reactivate
GID-0001 never deleted         →  permanent record until operator closes case
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      pipeline.py                        │
│   Accepts: video files  OR  live RTSP  OR  webcam       │
└──────────────────────┬──────────────────────────────────┘
                       │
              GlobalTracker
              (one CameraWorker per source,
               all sharing the same DB)
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   CameraWorker 0  CameraWorker 1  ...N
        │
   ① YOLOv8 detect persons
   ② ByteTrack assign track_id
   ③ Crop person from frame
   ④ MobileNetV3 → 512-d embedding
        │
        └──────────────────────► IdentityStore (SQLite)
                                  ┌───────────────────────────────────┐
                                  │ match_or_create(embedding)        │
                                  │                                   │
                                  │  Search: active + LOST persons    │
                                  │    (lost persons always included) │
                                  │                                   │
                                  │  sim > threshold?                 │
                                  │    YES → return existing GID      │
                                  │           if was LOST → log       │
                                  │           reappearance event      │
                                  │    NO  → create new GID-XXXX      │
                                  │                                   │
                                  │  promote_lost() every 30 frames   │
                                  │    absent > threshold → LOST      │
                                  │    NEVER deleted                  │
                                  └───────────────────────────────────┘
                                            │
                                  Streamlit Dashboard
                                  ┌──────────────────────────────────┐
                                  │  📊 Overview   live stats        │ 
                                  │               + reappearance     │
                                  │                 alerts           │
                                  │  🟢 Active    currently tracked  │
                                  │  🔴 Lost      never-delete       │
                                  │               registry           │
                                  │  🔍 Search    by ID/time/camera  │
                                  │  👤 Detail    full event         │
                                  │               timeline           │
                                  └──────────────────────────────────┘
```

---

## Re-ID Model: Built From Scratch

Since torchreid cannot be installed on Python 3.10+ without a working C/Cython build environment, the ReID backbone is implemented entirely using `torchvision`, which ships with PyTorch.

```
Person crop  (H × W × 3 BGR)
        │
        ▼  Resize 256×128  +  ImageNet normalise
        │
        ▼  MobileNetV3-Small backbone  (torchvision)
        │    ├── 16 InvertedResidual blocks
        │    ├── Squeeze-and-Excitation channel attention
        │    └── AdaptiveAvgPool2d  →  (576,)
        │
        ▼  Linear(576 → 512)  +  BatchNorm1d
        │
        ▼  L2-normalise
        │
   512-d unit vector
   (cosine similarity = dot product — no FAISS needed)
```

**Embedding update strategy:** Exponential Moving Average (EMA) per identity — `0.7 × old + 0.3 × new` — so the stored embedding adapts to lighting and pose changes across the session without drifting away from the original appearance.

---

## Identity Lifecycle & Event Log

Every person ever detected gets one row in the `persons` table. Status transitions are recorded in a separate `events` table — the full lifecycle is auditable:

```
🆕 FIRST_SEEN  →  Camera 0, frame 42
🔴 LOST        →  Not seen for 120s, last seen Camera 0
🔄 REAPPEARED  →  Camera 1, frame 891 — ID automatically restored
📝 NOTE        →  "Confirmed ID at Gate 3"
✅ RESOLVED    →  Operator closed case
```

**Key guarantee:** Lost persons are **always included in the embedding search**. If `GID-0003` was marked LOST and then reappears on any camera — whether the same video or a different one — the system matches their embedding and restores `GID-0003` automatically. The reappearance is logged as an event and shown as an alert on the dashboard.

**Resolved persons are excluded** from future matching — once an operator closes a case, that identity will not be re-assigned.

---

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/multicam-reid.git
cd multicam-reid

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# 3. PyTorch — choose your platform:
# CPU only (any OS):
pip install torch torchvision
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Everything else (no build steps, no Cython):
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run.

---

## Usage

### Single video (testing)
```bash
python pipeline.py --videos your_video.mp4 --output results/

# Shorter lost threshold for quick testing (30s instead of 2 min)
python pipeline.py --videos your_video.mp4 --output results/ --lost-threshold 30
```

### Two cameras
```bash
python pipeline.py --videos cam0.mp4 cam1.mp4 --output results/
```

### Live RTSP streams
```bash
python pipeline.py --live \
  --sources rtsp://192.168.1.10/stream1 rtsp://192.168.1.11/stream2 \
  --output results/
```

### Webcam
```bash
python pipeline.py --live --sources 0 1 --output results/
```

### Open operator dashboard (separate terminal, any time)
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

---

## All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--videos` | — | Video file paths (offline mode) |
| `--live` | — | Live mode (use with `--sources`) |
| `--sources` | — | RTSP URLs or webcam indices |
| `--output` | `results` | Output directory |
| `--db` | `database/identities.db` | SQLite DB path |
| `--yolo-model` | `yolov8n.pt` | YOLOv8 variant: n/s/m/l/x |
| `--reid-weights` | None | Fine-tuned ReID checkpoint (optional) |
| `--conf` | `0.35` | Detection confidence threshold |
| `--sim-threshold` | `0.60` | Identity match cosine threshold |
| `--lost-threshold` | `120` | Seconds absent before → LOST |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

---

## Tuning Guide

### Identity match threshold (`--sim-threshold`)

| Value | Behaviour |
|-------|-----------|
| `0.50` | Permissive — more re-IDs, may have false matches |
| `0.60` | Balanced ← **recommended** |
| `0.70` | Strict — fewer re-IDs, very reliable |

### Lost threshold (`--lost-threshold`)

| Scenario | Recommended value |
|----------|------------------|
| Quick testing | `30` seconds |
| Indoor retail / office | `120` seconds (2 min) |
| Large building / campus | `300` seconds (5 min) |

### YOLOv8 model size vs speed (CPU)

| Model | Speed (CPU, 720p) | Use when |
|-------|-------------------|----------|
| `yolov8n.pt` | ~8 FPS | Testing, low-power hardware |
| `yolov8s.pt` | ~5 FPS | Better accuracy needed |
| `yolov8m.pt` | ~2 FPS | High accuracy, GPU recommended |

---

## Project Structure

```
multicam-reid/
│
├── pipeline.py                  ←  main entry point (file or live)
│
├── tracker/
│   └── global_tracker.py        ←  CameraWorker + GlobalTracker orchestrator
│
├── reid/
│   └── feature_extractor.py     ←  MobileNetV3 + 512-d embedding head
│
├── database/
│   └── identity_store.py        ←  SQLite identity store + event log
│                                    (match_or_create, promote_lost,
│                                     resolve, reactivate, add_note)
│
├── dashboard/
│   └── app.py                   ←  Streamlit operator dashboard
│                                    (Overview, Active, Lost Registry,
│                                     Search, Person Detail + event timeline)
│
├── eval/
│   └── metrics.py               ←  MOTA, IDF1, MOTP (pure numpy)
│
├── videos/input/                ←  put your .mp4 files here
├── requirements.txt
└── README.md
```

---

## Database Schema

```sql
-- One row per person, ever
persons (
    global_id       TEXT  UNIQUE,   -- GID-0001
    status          TEXT,           -- active | lost | resolved
    first_seen_at   REAL,
    last_seen_at    REAL,
    last_camera_id  INTEGER,
    embedding       BLOB,           -- float32 (512,) EMA-updated
    best_crop_path  TEXT,
    notes           TEXT,
    resolved_at     REAL
)

-- Every detection frame logged
sightings (
    global_id   TEXT,
    camera_id   INTEGER,
    frame_idx   INTEGER,
    seen_at     REAL,
    bbox        TEXT,    -- JSON [x1,y1,x2,y2]
    conf        REAL,
    crop_path   TEXT
)

-- Full lifecycle audit trail
events (
    global_id   TEXT,
    event_type  TEXT,    -- first_seen | lost | reappeared | resolved | note
    camera_id   INTEGER,
    occurred_at REAL,
    detail      TEXT
)
```

---

## References

- **YOLOv8**: Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- **ByteTrack**: Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- **MobileNetV3**: Howard, A. et al. (2019). *Searching for MobileNetV3*. ICCV 2019.
- **DeepSORT** *(original dissertation)*: Wojke, N. et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. ICIP 2017.
- **torchreid** *(original dissertation)*: Zhou, K. et al. (2019). *Omni-Scale Feature Learning for Person Re-Identification*. ICCV 2019.
- **DukeMTMC-ReID** *(original dissertation dataset)*: Zheng, Z. et al. (2017). *Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline In Vitro*.
- **Original reference repository**: Hormi, S. (2021). *Multi-Camera Person Tracking and Re-Identification*. https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification
- **Dissertation**: Gaikwad, K. (2022). *Multi-Camera Multi-People Tracking and Re-Identification*. MSc Dissertation, Sheffield Hallam University. Supervisor: Dr. Jing Wang.

---

## Academic Use & Attribution

This repository is a **clean reimplementation** of the author's own MSc dissertation research, updated with a modern, fully pip-installable stack. The original dissertation results (IDF1=50.7%, MOTA=99.8%) were obtained using a different codebase (YOLOv4 + DeepSORT + torchreid). No code from the [samihormi reference repository](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification) is included here — the architectural pipeline concept (detect → track → ReID feature extraction → cross-camera matching) is standard methodology in the multi-camera tracking literature, not proprietary to any single implementation.

If you use this work in academic research, please cite the relevant upstream papers listed above.

---

## License

MIT License — see [LICENSE](LICENSE).  
Copyright © 2025 Kunal Gaikwad