"""
tracker/global_tracker.py
--------------------------
GlobalTracker coordinates multiple camera streams (file or RTSP/webcam)
and resolves every detected person against the IdentityStore.

Flow per frame
--------------
1.  YOLOv8 + ByteTrack  →  raw track_id + bbox
2.  Extract MobileNetV3 embedding from crop
3.  IdentityStore.match_or_create()
      → compare embedding against ALL known persons in DB
      → return existing global_id  (same person seen before, any camera)
        OR create new global_id    (genuinely new person)
4.  Write sighting to DB
5.  Periodically call store.promote_lost() to flag absent persons

Key guarantee
-------------
  - A person leaving Cam0 and appearing on Cam1 gets the SAME global_id
  - A person leaving Cam1 and returning to Cam1 gets the SAME global_id
  - Both above scenarios checked from the SAME embedding database
  - Lost persons are never deleted — only status changes
"""

import cv2
import time
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import torch
from ultralytics import YOLO

from reid.feature_extractor import FeatureExtractor
from database.identity_store import IdentityStore


PERSON_CLASS = 0


def _color(global_id: str) -> tuple:
    """Deterministic BGR colour from a GID string."""
    h = abs(hash(global_id)) % (2**31)
    np.random.seed(h % (2**32 - 1))
    r, g, b = np.random.randint(80, 230, 3)
    return (int(b), int(g), int(r))


class CameraWorker:
    """
    Processes a single camera stream (file or RTSP/webcam URL).
    Runs detection + tracking + embedding extraction each frame.
    Writes results to the shared IdentityStore.
    Can run in its own thread for live multi-camera setups.

    Parameters
    ----------
    source      : video file path, RTSP URL, or webcam index (int)
    camera_id   : integer label for this camera
    store       : shared IdentityStore instance
    extractor   : shared FeatureExtractor instance
    yolo_model  : YOLO instance (shared across workers)
    output_dir  : where to write output video + crops
    conf        : detection confidence threshold
    promote_every_n : call store.promote_lost() every N frames
    save_video  : write annotated output .mp4
    save_crops  : save person crop images to disk
    """

    def __init__(
        self,
        source,
        camera_id: int,
        store: IdentityStore,
        extractor: FeatureExtractor,
        yolo_model: YOLO,
        output_dir: str,
        conf: float         = 0.35,
        promote_every_n: int = 30,
        save_video: bool    = True,
        save_crops: bool    = True,
        device: str         = "cpu",
    ):
        self.source      = source
        self.camera_id   = camera_id
        self.store       = store
        self.extractor   = extractor
        self.model       = yolo_model
        self.output_dir  = Path(output_dir)
        self.conf        = conf
        self.promote_every = promote_every_n
        self.save_video  = save_video
        self.save_crops  = save_crops
        self.device      = device

        self.crops_dir = self.output_dir / f"cam{camera_id}_crops"
        if save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Live stats
        self.frame_idx    = 0
        self.running      = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_sync(self) -> Dict[int, List[dict]]:
        """Process video synchronously (for file-based offline use)."""
        self.running = True
        all_tracks = {}
        cap, writer = self._open_source()

        print(f"[Cam {self.camera_id}] Starting  source={self.source}")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            result = self._process_frame(frame, writer)
            all_tracks[self.frame_idx - 1] = result

        cap.release()
        if writer:
            writer.release()
        self.running = False
        print(f"[Cam {self.camera_id}] Done  ({self.frame_idx} frames)")
        return all_tracks

    def run_threaded(self):
        """Start processing in a background thread (for live streams)."""
        self._thread = threading.Thread(
            target=self.run_sync, daemon=True, name=f"cam{self.camera_id}"
        )
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, writer) -> List[dict]:
        h, w = frame.shape[:2]

        # ── Step 1: YOLOv8 + ByteTrack ────────────────────────────────
        results = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=self.conf,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )

        frame_dets = []
        r = results[0]

        if r.boxes is not None and r.boxes.id is not None:
            boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            confs     = r.boxes.conf.cpu().numpy()

            # ── Step 2: batch-extract embeddings ──────────────────────
            crops = []
            valid = []
            for box, tid, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = (
                    max(0, int(box[0])), max(0, int(box[1])),
                    min(w, int(box[2])), min(h, int(box[3]))
                )
                crop = frame[y1:y2, x1:x2]
                if crop.size > 100:
                    crops.append(crop)
                    valid.append((tid, conf, [x1, y1, x2, y2], crop))

            if crops:
                embeddings = self.extractor.extract(crops)  # (N, 512)
            else:
                embeddings = []

            # ── Step 3: resolve against global identity store ─────────
            for i, (tid, conf, bbox, crop) in enumerate(valid):
                emb = embeddings[i] if i < len(embeddings) else None
                if emb is None:
                    continue

                # Save crop
                crop_path = None
                if self.save_crops:
                    crop_path = str(
                        self.crops_dir /
                        f"f{self.frame_idx:06d}_t{int(tid):04d}.jpg"
                    )
                    cv2.imwrite(crop_path, crop)

                # Match or create in global DB
                global_id, is_new, was_lost = self.store.match_or_create(
                    embedding=emb,
                    camera_id=self.camera_id,
                    frame_idx=self.frame_idx,
                    bbox=bbox,
                    conf=float(conf),
                    crop_path=crop_path,
                )

                if is_new:
                    print(f"  [Cam {self.camera_id}] 🆕 New:        {global_id}"
                          f"  f={self.frame_idx}")
                elif was_lost:
                    print(f"  [Cam {self.camera_id}] 🔄 Reappeared: {global_id}"
                          f"  f={self.frame_idx}  (was LOST → now ACTIVE)")

                frame_dets.append({
                    "track_id"  : int(tid),
                    "global_id" : global_id,
                    "bbox"      : bbox,
                    "conf"      : float(conf),
                    "crop_path" : crop_path,
                    "camera_id" : self.camera_id,
                    "frame_idx" : self.frame_idx,
                    "is_new"    : is_new,
                    "was_lost"  : was_lost,
                })

        # ── Step 4: annotate + write frame ────────────────────────────
        if writer:
            ann = self._annotate(frame.copy(), frame_dets)
            writer.write(ann)

        # ── Step 5: periodic lost-person promotion ────────────────────
        if self.frame_idx % self.promote_every == 0:
            self.store.promote_lost()

        self.frame_idx += 1
        return frame_dets

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, dets: List[dict]) -> np.ndarray:
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            gid   = det["global_id"]
            conf  = det["conf"]
            color = _color(gid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{gid}  {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        stats = self.store.stats()
        bar   = (f"CAM {self.camera_id}  |  f{self.frame_idx:05d}"
                 f"  |  active={stats['active']}"
                 f"  lost={stats['lost']}"
                 f"  resolved={stats['resolved']}")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0,0,0), -1)
        cv2.putText(frame, bar, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 2)
        return frame

    # ------------------------------------------------------------------
    # Open source (file or RTSP)
    # ------------------------------------------------------------------

    def _open_source(self):
        cap = cv2.VideoCapture(
            int(self.source) if str(self.source).isdigit() else str(self.source)
        )
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"cam{self.camera_id}_tracked.mp4"
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        return cap, writer


# ---------------------------------------------------------------------------
# GlobalTracker: orchestrates all cameras
# ---------------------------------------------------------------------------

class GlobalTracker:
    """
    Top-level orchestrator for multi-camera tracking.

    Parameters
    ----------
    sources         : list of video paths / RTSP URLs / webcam indices
    output_dir      : root output directory
    db_path         : SQLite database path
    reid_weights    : optional fine-tuned ReID checkpoint
    yolo_model      : YOLOv8 model name
    conf            : detection confidence
    sim_threshold   : identity match threshold (global DB lookup)
    lost_threshold  : seconds before marking person as lost
    device          : 'cpu', 'cuda', or 'auto'
    """

    def __init__(
        self,
        sources: List,
        output_dir: str          = "results",
        db_path: str             = "database/identities.db",
        reid_weights: str        = None,
        yolo_model: str          = "yolov8n.pt",
        conf: float              = 0.35,
        sim_threshold: float     = 0.60,
        lost_threshold: float    = 120.0,
        device: str              = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device  = device
        self.sources = sources

        print(f"\n[GlobalTracker] Initialising  cameras={len(sources)}"
              f"  device={device}")

        # Shared components
        self.store = IdentityStore(
            db_path=db_path,
            lost_threshold_secs=lost_threshold,
            similarity_threshold=sim_threshold,
        )
        self.extractor = FeatureExtractor(
            weights_path=reid_weights,
            device=device,
            batch_size=8,
        )
        print(f"[GlobalTracker] Loading {yolo_model} ...")
        self.yolo = YOLO(yolo_model)

        # One CameraWorker per source
        self.workers: List[CameraWorker] = []
        for cam_id, src in enumerate(sources):
            self.workers.append(CameraWorker(
                source=src,
                camera_id=cam_id,
                store=self.store,
                extractor=self.extractor,
                yolo_model=self.yolo,
                output_dir=output_dir,
                conf=conf,
                device=device,
            ))

    def run_files(self) -> Dict[int, dict]:
        """
        Process all file-based sources sequentially.
        Returns {cam_id: {frame_idx: [dets]}}
        """
        all_tracks = {}
        for worker in self.workers:
            all_tracks[worker.camera_id] = worker.run_sync()
        # Final lost check
        self.store.promote_lost()
        return all_tracks

    def run_live(self):
        """
        Launch all cameras in parallel threads (for live RTSP/webcam).
        Block until all stopped or KeyboardInterrupt.
        """
        print(f"[GlobalTracker] Starting {len(self.workers)} live camera threads")
        for w in self.workers:
            w.run_threaded()
        try:
            while any(w.running for w in self.workers):
                time.sleep(1)
                self.store.promote_lost()
        except KeyboardInterrupt:
            print("\n[GlobalTracker] Stopping ...")
            for w in self.workers:
                w.stop()
