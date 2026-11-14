#!/usr/bin/env python3
"""Record the RealSense colour stream (or any webcam) to an MP4 plus periodic JPG snapshots.

    python deploy/camera_record.py --seconds 110 --out runs/real_hand/cam_<time>
Writes <out>.mp4 (15 fps) and <out>_NNN.jpg every --snap-every seconds. Falls back to
OpenCV VideoCapture(--index) when pyrealsense2 is unavailable or no RealSense is found.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def open_realsense(width, height, fps):
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    prof = pipe.start(cfg)
    dev = prof.get_device()
    print(f"[cam] RealSense {dev.get_info(rs.camera_info.name)} sn={dev.get_info(rs.camera_info.serial_number)} {width}x{height}@{fps}")

    def grab():
        fr = pipe.wait_for_frames(2000)
        c = fr.get_color_frame()
        return np.asanyarray(c.get_data()) if c else None

    return grab, pipe.stop


def open_webcam(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"webcam {index} not available")
    print(f"[cam] webcam {index}")

    def grab():
        ok, f = cap.read()
        return f if ok else None

    return grab, cap.release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--out", default=None)
    ap.add_argument("--snap-every", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--index", type=int, default=0, help="webcam index for the fallback")
    args = ap.parse_args()
    out = Path(args.out or f"runs/real_hand/cam_{datetime.now():%Y-%m-%d_%H%M%S}")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        grab, close = open_realsense(args.width, args.height, max(args.fps, 15))
    except Exception as e:
        print(f"[cam] RealSense not used ({type(e).__name__}: {e}); trying a webcam")
        grab, close = open_webcam(args.index, args.width, args.height)
    writer = None
    t0 = time.time(); last_snap = -1e9; n = 0; snaps = 0
    try:
        while time.time() - t0 < args.seconds:
            f = grab()
            if f is None:
                continue
            if writer is None:
                h, w = f.shape[:2]
                writer = cv2.VideoWriter(str(out) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            stamp = f"{time.time() - t0:6.1f}s  {datetime.now():%H:%M:%S}"
            cv2.putText(f, stamp, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(f, stamp, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(f); n += 1
            if time.time() - t0 - last_snap >= args.snap_every:
                cv2.imwrite(f"{out}_{snaps:03d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85]); snaps += 1
                last_snap = time.time() - t0
    finally:
        if writer is not None:
            writer.release()
        close()
    print(f"[cam] wrote {out}.mp4 ({n} frames) and {snaps} snapshots")


if __name__ == "__main__":
    main()
