"""
pipeline.py
-----------
Main entry point for the Multi-Camera Tracking & Re-ID system.

Usage — video files
-------------------
  python pipeline.py --videos cam0.mp4 cam1.mp4 --output results/

Usage — live RTSP streams
-------------------------
  python pipeline.py --live \
    --sources rtsp://192.168.1.10/stream1 rtsp://192.168.1.11/stream2 \
    --output results/

Usage — mixed (file + webcam)
------------------------------
  python pipeline.py --live --sources cam0.mp4 0 --output results/

Then open the dashboard in a separate terminal:
  streamlit run dashboard/app.py

Key flags
---------
  --lost-threshold   seconds absent before person → LOST   [120]
  --sim-threshold    cosine similarity for identity match  [0.60]
  --reentry-timeout  alias for lost-threshold (same value)
"""

import argparse
import time
from pathlib import Path

from tracker.global_tracker import GlobalTracker


def main():
    p = argparse.ArgumentParser(
        description="Multi-Camera Person Tracking & Re-Identification"
    )

    # Sources
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--videos", nargs="+",
        help="Video file paths (offline processing)"
    )
    source_group.add_argument(
        "--live", action="store_true",
        help="Live mode — use --sources for RTSP URLs or webcam indices"
    )
    p.add_argument(
        "--sources", nargs="+",
        help="For --live mode: RTSP URLs or webcam indices (e.g. 0 1)"
    )

    # Output
    p.add_argument("--output",          default="results",
                   help="Output directory  [results]")
    p.add_argument("--db",              default="database/identities.db",
                   help="SQLite database path")

    # Model
    p.add_argument("--yolo-model",      default="yolov8n.pt",
                   help="YOLOv8 variant: n/s/m/l/x.pt  [yolov8n.pt]")
    p.add_argument("--reid-weights",    default=None,
                   help="Optional fine-tuned ReID .pth")

    # Thresholds
    p.add_argument("--conf",            type=float, default=0.35,
                   help="Detection confidence  [0.35]")
    p.add_argument("--sim-threshold",   type=float, default=0.60,
                   help="Identity match cosine threshold  [0.60]")
    p.add_argument("--lost-threshold",  type=float, default=120.0,
                   help="Seconds absent → LOST status  [120]")

    # System
    p.add_argument("--device",          default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-video",        action="store_true",
                   help="Don't save output videos (faster)")

    args = p.parse_args()

    # Resolve sources
    if args.videos:
        sources = args.videos
        live = False
    else:
        if not args.sources:
            p.error("--live requires --sources")
        # Convert numeric strings to ints (webcam indices)
        sources = [int(s) if s.isdigit() else s for s in args.sources]
        live = True

    _banner(sources, args.output, args.device,
            args.sim_threshold, args.lost_threshold)

    tracker = GlobalTracker(
        sources=sources,
        output_dir=args.output,
        db_path=args.db,
        reid_weights=args.reid_weights,
        yolo_model=args.yolo_model,
        conf=args.conf,
        sim_threshold=args.sim_threshold,
        lost_threshold=args.lost_threshold,
        device=args.device,
    )

    if live:
        print("\n🎥  Live mode — press Ctrl+C to stop\n")
        tracker.run_live()
    else:
        print("\n📹  Processing video files ...\n")
        t0 = time.time()
        all_tracks = tracker.run_files()
        elapsed = time.time() - t0

        # Print summary
        stats = tracker.store.stats()
        print("\n" + "═" * 55)
        print("  PIPELINE COMPLETE")
        print("═" * 55)
        for cam_id, cam_data in all_tracks.items():
            n_frames  = len(cam_data)
            n_gids    = len({d["global_id"]
                             for fd in cam_data.values() for d in fd})
            print(f"  Camera {cam_id}: {n_frames} frames | {n_gids} unique persons")
        print(f"\n  DB stats:")
        print(f"    Active   : {stats['active']}")
        print(f"    Lost     : {stats['lost']}")
        print(f"    Resolved : {stats['resolved']}")
        print(f"    Sightings: {stats['sightings']}")
        print(f"\n  Time: {elapsed:.1f}s")
        print("═" * 55)
        print(f"\n  Results  → {args.output}/")
        print(f"  Database → {args.db}")
        print(f"\n  Open dashboard:  streamlit run dashboard/app.py\n")


def _banner(sources, out, device, sim_thr, lost_thr):
    print("\n" + "═" * 55)
    print("  Multi-Camera Tracking & Re-ID")
    print("═" * 55)
    print(f"  Sources          : {len(sources)} camera(s)")
    for i, s in enumerate(sources):
        print(f"    [{i}] {s}")
    print(f"  Output           : {out}")
    print(f"  Device           : {device}")
    print(f"  Sim threshold    : {sim_thr}  (identity match)")
    print(f"  Lost threshold   : {lost_thr}s  (absent → LOST)")
    print("═" * 55)


if __name__ == "__main__":
    main()
