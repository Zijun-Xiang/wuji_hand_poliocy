#!/usr/bin/env python3
"""Run Wuji's official pen observer on an Intel RealSense colour stream.

Wuji's `pen_world_observer.py` grabs frames from a Hikrobot camera (MVS SDK)
and reads the intrinsics from `config/camera.yaml`.  This wrapper replaces
exactly those two things:

  * frames come from the RealSense colour stream (pyrealsense2, factory
    intrinsics and distortion from the device, no calibration step);
  * the camera config handed to the observer is built from those intrinsics
    (ROI = the full frame).

Detection, PnP, the wrist-tag world frame, the ZMQ publishing and the preview
window are Wuji's code, untouched.  The pen still has to carry Wuji's 18 tag
patterns and the wrist its AprilTag 36h11 ID 0 (deploy/markers/).

    python pen_observer_realsense.py --wuji-root <wuji-mjlab> [--width 1280 --height 720 --fps 30] [--preview] [observer options]

Caveat: the Hikrobot setup sees the 13.5 mm pen tags at ~2000 px focal length;
a RealSense lens is much wider (colour ~1380 px at 1920x1080, IR ~640 px at
1280x720), so the tags are 2-3x smaller in pixels.  Put the camera close
(25-35 cm from the pen).  Two usable modes on a D435i:
  --stream color    1920x1080@30: most pixels per tag, but rolling shutter and 30 fps
  --stream infrared 1280x720@90 or 848x480@90: global shutter, 90 fps like Wuji's rig,
                    fewer pixels per tag, projector off

Not yet run on hardware: written against the pyrealsense2 2.5x API.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wuji-root", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stream", choices=("color", "infrared"), default="color",
                    help="color: rolling-shutter RGB (sharper tags, 30 fps at 1280x720); "
                         "infrared: left IR imager, GLOBAL shutter, mono, up to 90 fps, projector switched off "
                         "(better for fast motion; try 848x480 or 1280x720 at 90)")
    ap.add_argument("--serial", default=None, help="RealSense serial when several are plugged in")
    ap.add_argument("--exposure", type=float, default=None, help="fixed colour exposure (device units); default: auto-exposure")
    args, observer_argv = ap.parse_known_args()

    root = Path(args.wuji_root).expanduser().resolve() / "deploy" / "pen_spin"
    if not (root / "scripts" / "pen_world_observer.py").is_file():
        sys.exit(f"{root} is not wuji-mjlab/deploy/pen_spin")
    import os

    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    if args.stream == "color":
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    else:
        cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    profile = pipe.start(cfg)
    if args.stream == "infrared":
        try:  # the dot projector would cover the tags with speckles
            profile.get_device().first_depth_sensor().set_option(rs.option.emitter_enabled, 0)
        except Exception as exc:  # noqa: BLE001
            print(f"[realsense] could not switch the IR projector off: {exc}")
    if args.exposure is not None:
        # a fixed, short exposure keeps fast-moving tags sharp (auto-exposure
        # tends to pick long exposures indoors); the unit is device-specific,
        # read the current value from the RealSense Viewer first
        try:
            dev = profile.get_device()
            sensor = dev.first_color_sensor() if args.stream == "color" else dev.first_depth_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, float(args.exposure))
        except Exception as exc:  # noqa: BLE001 - exposure control is a nicety
            print(f"[realsense] could not set the exposure: {exc}")
    stream_sel = profile.get_stream(rs.stream.color) if args.stream == "color" else profile.get_stream(rs.stream.infrared, 1)
    intr = stream_sel.as_video_stream_profile().get_intrinsics()
    k = list(intr.coeffs) + [0.0] * 5
    camera_cfg = {
        "sensor": {"full_width": intr.width, "full_height": intr.height},
        "roi": {"offset_x": 0, "offset_y": 0, "width": intr.width, "height": intr.height},
        "fast_roi": {"offset_x": 0, "offset_y": 0, "width": intr.width, "height": intr.height},
        "intrinsics": {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(intr.ppx), "cy": float(intr.ppy)},
        "distortion": {"k1": k[0], "k2": k[1], "p1": k[2], "p2": k[3], "k3": k[4]},
        "capture": {"exposure_time": 6000, "gain": 0, "frame_rate": args.fps},
    }
    print(f"[realsense] {args.stream} {intr.width}x{intr.height}@{args.fps} fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f} model={intr.model} coeffs={list(intr.coeffs)}")

    import cv2
    import pen_world_observer as obs
    from lib import camera_config

    def frames(_cfg):
        n = 0
        t0 = time.perf_counter()
        while True:
            fs = pipe.wait_for_frames()
            c = fs.get_color_frame() if args.stream == "color" else fs.get_infrared_frame(1)
            if not c:
                continue
            img = np.asanyarray(c.get_data())
            n += 1
            if n % 300 == 0:
                print(f"[realsense] {n / (time.perf_counter() - t0):.1f} fps delivered", flush=True)
            yield (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img), None

    # the observer calls these by module-global name
    obs.load_camera_config = lambda config_file=None: camera_cfg
    camera_config.load_camera_config = lambda config_file=None: camera_cfg
    obs._camera_frames = frames
    sys.argv = [sys.argv[0], "--source", "camera", *observer_argv]
    try:
        return obs.main()
    finally:
        pipe.stop()


if __name__ == "__main__":
    sys.exit(main())
