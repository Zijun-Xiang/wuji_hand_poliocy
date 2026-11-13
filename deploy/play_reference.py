#!/usr/bin/env python3
"""Play a stitched loop's reference joint motion on the real Wuji Hand 2 (no camera).

This is the camera-less test: the hand follows the loop's reference joint angles
(`reference_qpos`, 50 Hz) with Wuji's deploy-side driver, gains and easing, and
reports how well the real joints track the stitched trajectory, including at
the blends between demos.  It does NOT spin a pen: without the camera the
tracking policy cannot see the pen (12 of 13 official clips fail within 0.7-7 s
in Wuji's simulator when the pen pose is missing), so this script moves the
hand in the air, or with a pen placed loosely for a look.

Hardware access goes through wuji-mjlab's own deploy code
(`deploy/pen_spin/runtime/hand.py` -> `lib/hand2_adapter.py` -> wuji_sdk);
only the terminal handling is replaced so the script also runs on Windows
(`runtime/safety.py` needs termios).

    python play_reference.py --wuji-root <wuji-mjlab> --motion clips/graph_loop_1lap.npz --no-hardware   # dry run
    python play_reference.py --wuji-root <wuji-mjlab> --motion clips/graph_loop_1lap.npz --speed 0.5     # real hand, half speed

Keys: Enter = start (after the hand eased to frame 0), Ctrl+C = ease the hand open and stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EASE_S = 3.0


def ease(ctrl, current, target, duration=EASE_S, rate=50.0):
    """Smoothstep from current to target (same shape as runtime/safety.py::_ease)."""
    steps = max(1, int(duration * rate))
    cur = np.asarray(current, dtype=np.float64).reshape(5, 4)
    tgt = np.asarray(target, dtype=np.float64).reshape(5, 4)
    for i in range(steps):
        t = (i + 1) / steps
        ctrl.set_joint_target_position(cur + (t * t * (3 - 2 * t)) * (tgt - cur))
        time.sleep(duration / steps)


def pace(deadline: float) -> None:
    while True:
        slack = deadline - time.perf_counter()
        if slack <= 0:
            return
        if slack > 0.001:
            time.sleep(slack - 0.0005)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wuji-root", required=True, help="wuji-mjlab checkout (its deploy/pen_spin is imported)")
    ap.add_argument("--motion", required=True, help="clip .npz in the official schema (deploy/clips/*.npz)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor; 0.5 = half speed (default 1.0)")
    ap.add_argument("--laps", type=int, default=1, help="repeat the clip this many times")
    ap.add_argument("--hold-s", type=float, default=1.0, help="hold frame 0 before moving (official: 1 s)")
    ap.add_argument("--no-hardware", action="store_true", help="dry run with Wuji's stub hand")
    ap.add_argument("--hand-sn", default=None)
    ap.add_argument("--log", default=None, help="save commanded/actual joints to this .npz")
    args = ap.parse_args()
    motion = Path(args.motion).expanduser().resolve()
    side = motion.with_suffix(".groups.json")
    log_path = Path(args.log).expanduser().resolve() if args.log else None

    root = Path(args.wuji_root).expanduser().resolve() / "deploy" / "pen_spin"
    if not (root / "runtime" / "hand.py").is_file():
        sys.exit(f"{root} is not wuji-mjlab/deploy/pen_spin")
    sys.path.insert(0, str(root))
    import os

    os.chdir(root)  # lib.config_loader reads config/*.yaml relative to the deploy dir
    from lib.config_loader import get_gains, get_timing
    from lib.motion_clip import load_motion_clip
    from runtime import hand as hand_mod

    clip = load_motion_clip(motion, fps=1.0 / float(get_timing()["ctrl_dt"]))
    q = clip.reference_qpos.astype(np.float64)
    dt = clip.frame_dt / args.speed
    lower, upper = hand_mod.joint_limits()
    lower, upper = np.asarray(lower).reshape(20), np.asarray(upper).reshape(20)
    excess = max(float((lower - q).max()), float((q - upper).max()), 0.0)
    if excess > np.deg2rad(1.0):
        sys.exit(f"clip leaves the hand's joint limits (control.yaml) by {np.rad2deg(excess):.1f} deg — refusing to play it")
    if excess > 0:
        # Wuji's own clips exceed the deploy limits by up to 0.2 deg; the official runtime clamps, so do we
        print(f"  note   clip exceeds the joint limits by {np.rad2deg(excess):.2f} deg at most; targets are clamped like pen-policy does")
    q = np.clip(q, lower, upper)
    speed = np.abs(np.diff(q, axis=0)).max(axis=1) / dt
    groups = None
    if side.is_file():
        groups = json.loads(side.read_text())["frame_groups"]
    print(f"  clip   {motion.name}: {len(q)} frames = {len(q) * clip.frame_dt:.1f} s, "
          f"x{args.laps} laps at speed {args.speed} -> {len(q) * dt * args.laps:.1f} s of motion")
    print(f"  joints max step {np.rad2deg(speed.max()) * dt:.2f} deg per 20 ms, "
          f"max speed {speed.max():.2f} rad/s (data P99.9 x 1.5 = 5.81)")
    if groups:
        from collections import Counter

        print(f"  source groups per frame: {dict(Counter(groups))}")

    gains = get_gains()
    if args.no_hardware:
        seed = q[0].reshape(5, 4)
        hand = hand_mod.StubHand(lower.reshape(5, 4), upper.reshape(5, 4), seed)
        print("  hand   stub (--no-hardware)")
    else:
        hand = hand_mod.connect_real(serial_number=args.hand_sn, kp=gains["kp"], kd=gains["kd"],
                                     servo_hz=get_timing()["servo_hz"], interp_s=clip.frame_dt,
                                     kp_overrides=gains["kp_joint"], kd_overrides=gains["kd_joint"],
                                     effort_overrides=gains["effort_joint"])
        hand.write_joint_effort_limit(gains["effort"])
        print(f"  hand   Wuji Hand 2 via wuji_sdk, kp={gains['kp']} kd={gains['kd']} effort limit {gains['effort']} A")

    logged_cmd, logged_act, logged_t = [], [], []
    try:
        with hand.realtime_controller() as ctrl:
            hand.write_joint_enabled(True)
            cur = np.asarray(hand.read_joint_actual_position(), dtype=np.float64)
            ease(ctrl, cur, q[0])
            err = np.rad2deg(np.abs(np.asarray(hand.read_joint_actual_position()).reshape(20) - q[0]).max())
            print(f"  eased to frame 0 (max joint error {err:.2f} deg)")
            print("\n  >>> Hand is at the loop's start pose. CLEAR THE AREA. Press Enter to play, Ctrl+C to open and stop.")
            input()
            t0 = time.perf_counter()
            total = len(q) * args.laps
            hold = int(round(args.hold_s / dt))
            for i in range(-hold, total):
                k = max(i, 0) % len(q)
                ctrl.set_joint_target_position(q[k].reshape(5, 4))
                act = np.asarray(ctrl.get_joint_actual_position(), dtype=np.float64).reshape(20)
                logged_cmd.append(q[k]); logged_act.append(act); logged_t.append(time.perf_counter() - t0)
                if i >= 0 and i % int(round(1.0 / dt)) == 0:
                    e = np.rad2deg(np.abs(act - q[k]).max())
                    g = f"  group {groups[k]}" if groups else ""
                    print(f"  [{i * dt:6.1f} s] frame {k:5d}/{len(q)}  lap {i // len(q) + 1}  max joint err {e:5.1f} deg{g}", flush=True)
                pace(t0 + (i + hold + 1) * dt)
            print("\n  clip finished; easing the hand open.")
            ease(ctrl, hand.read_joint_actual_position(), np.zeros((5, 4)))
    except KeyboardInterrupt:
        print("\n  [Ctrl+C] easing hand open before shutdown...")
        try:
            with hand.realtime_controller() as ctrl:
                ease(ctrl, hand.read_joint_actual_position(), np.zeros((5, 4)))
            print("  hand open.")
        except KeyboardInterrupt:
            print("  [Ctrl+C] ease abandoned — disabling joints now.")
    finally:
        try:
            hand.write_joint_enabled(False)
        finally:
            close = getattr(hand, "close", None)
            if close:
                close()
        print("  joints disabled, hand closed.")
    if logged_cmd:
        cmd, act = np.asarray(logged_cmd), np.asarray(logged_act)
        e = np.rad2deg(np.abs(cmd - act))
        print(f"  tracking: mean joint error {e.mean():.2f} deg, worst {e.max():.2f} deg over {len(cmd)} ticks")
        if log_path:
            np.savez(log_path, t=np.asarray(logged_t), commanded=cmd, actual=act)
            print(f"  log saved to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
