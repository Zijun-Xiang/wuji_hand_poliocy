#!/usr/bin/env python3
"""Smoothest-motion showcase on the real Wuji Hand 2 (right), straight on wuji_sdk.

What makes it smooth (vs. the Nov-12 demo that jittered):
  * minimum-jerk keyframe trajectories (deploy/smooth_demo_traj.py): continuous position,
    velocity and acceleration, zero velocity at every keyframe;
  * commands at --hz (default 250) scheduled on time.perf_counter() with a 1 ms Windows timer
    (timeBeginPeriod) and sleep+spin waiting, so the send interval is even to ~0.02 ms;
  * velocity feedforward: every JointCommand carries the planned velocity, so the MIT damping
    term does not brake the planned motion;
  * firmware v3.0.0 default gains kp 5.0 / kd 0.1 (stiffer spring than the 3.0 that let friction
    stick the joints), 1.5 A effort limit (Wuji's deploy value);
  * the firmware's own friction feed-forward (constant + velocity table, position table) when
    the firmware accepts it (--ff on, default).
Safety: joints must all be online and fault-free; abort (ease back to the measured pose, disable)
if any joint lags its target by > 0.35 rad for 0.4 s, if a joint reports a fault, if feedback
stops, or on Ctrl+C. The whole showcase was checked for self-collision in MuJoCo.

    python deploy/smooth_demo.py                 # 48 s showcase, 8 s countdown
    python deploy/smooth_demo.py --speed 0.7     # slower
    python deploy/smooth_demo.py --no-hardware   # fake hand, exercises the loop and the log
Logs: runs/real_hand/smooth_demo_<time>.npz + .json (per-tick commanded/actual pos+vel, effort).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from smooth_demo_traj import LABELS, LIMITS, build_showcase  # noqa: E402

RUNS = HERE.parent / "runs" / "real_hand"
ABORT_ERR_RAD, ABORT_ERR_S, STALE_S = 0.35, 0.4, 0.5
_STOP = False


def _sigint(*_):
    global _STOP
    _STOP = True


# ----------------------------------------------------------------------------- precise timing
def enable_fine_timer():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
            return lambda: ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
    return lambda: None


def calibrate_sleep():
    worst = 0.0
    for _ in range(30):
        t0 = time.perf_counter(); time.sleep(0.0005); worst = max(worst, time.perf_counter() - t0 - 0.0005)
    margin = worst + 0.0002
    return margin if margin < 0.0025 else None


def wait_until(deadline, margin):
    while True:
        rem = deadline - time.perf_counter()
        if rem <= 0:
            return
        if margin is not None and rem > margin + 0.0003:
            time.sleep(rem - margin)


# ----------------------------------------------------------------------------- hand backends
class RealHand:
    def __init__(self, sn, kp, kd, effort, ff):
        import wuji_sdk
        self.sdk = wuji_sdk
        self.mgr = wuji_sdk.SdkManager.instance()
        if not sn:
            devs = self.mgr.scan()
            if not devs:
                raise SystemExit("no Wuji hand found")
            sn = devs[0].sn
        self.hand = self.mgr.connect(sn=sn, device_name="wuji_hand_2", options=wuji_sdk.ConnectOptions(enable_bridge=False))
        self.fw = str(self.hand.get("firmware_version"))
        n_online = int(self.hand.online_joints_count().get())
        print(f"[hand] {self.hand.serial_number} fw={self.fw} joints online={n_online}/20")
        if n_online != 20:
            raise SystemExit("not all 20 joints are online — refusing to move")
        self.hand.mit_params().set((float(kp), float(kd)))          # all 20 joints
        self.hand.effort_limit().set(float(effort))
        print(f"[hand] mit_params set kp={kp} kd={kd}; effort limit {effort} A")
        self.ff_state = "off"
        if ff:
            try:
                self.hand.set_constant(wuji_sdk.FeedForwardMode.ConstantAndVelocityTable)
                self.hand.set_position_table(True)
                self.ff_state = "constant+velocity table+position table"
            except Exception as e:
                self.ff_state = f"not available ({type(e).__name__}: {str(e)[:80]})"
                try:
                    self.hand.set_constant(wuji_sdk.FeedForwardMode.Off)
                except Exception:
                    pass
        print(f"[hand] firmware feed-forward: {self.ff_state}")
        self.sub_states = self.hand.joint_states().subscribe()
        self.sub_diag = self.hand.joint_diagnostics().subscribe()
        self.pub = self.hand.joint_command().publish()
        self.q = np.zeros(20); self.dq = np.zeros(20); self.tau = np.zeros(20)
        self.last_state_t = None
        self.fault = None
        self.warn = None
        self.enabled = False

    def poll(self):
        for _ in range(64):
            fr = self.sub_states.recv()
            if fr is None:
                break
            for e in fr.joints:
                f, jw = divmod(e.nid - 1, 5)
                if jw < 4 and 0 <= f < 5:
                    i = f * 4 + jw
                    self.q[i], self.dq[i], self.tau[i] = e.position, e.velocity, e.effort
            self.last_state_t = time.perf_counter()
        for _ in range(16):
            fr = self.sub_diag.recv()
            if fr is None:
                break
            for e in fr.joints:
                if e.error_code_current:
                    f, jw = divmod(e.nid - 1, 5)
                    code = e.error_code_current
                    try:
                        info = self.hand.describe_error(code) or {}
                    except Exception:
                        info = {}
                    sev = str(info.get("severity", "")).lower()
                    name = info.get("name", "")
                    label = LABELS[f * 4 + jw] if jw < 4 else "tactile"
                    # warnings (encoder data quality, soft-limit hit, ...) are logged; faults abort
                    if sev in ("warning", "info", "notice") or (not sev and code < 0x1000):
                        self.warn = f"{label} 0x{code:04X} {name}"
                    else:
                        self.fault = f"{label} 0x{code:04X} {name} ({sev}) {info.get('resolution', '')}"

    def wait_feedback(self, timeout=3.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            self.poll()
            if self.last_state_t is not None:
                return True
            time.sleep(0.01)
        return False

    def enable(self):
        try:
            self.hand.clear_fault()
        except Exception:
            pass
        self.hand.enable()
        self.enabled = True
        time.sleep(0.3)
        self.poll()

    def send(self, q, dq):
        JC = self.sdk.JointCommand
        self.pub.send([JC(float(p), float(v), 0.0) for p, v in zip(q, dq)])

    def disable(self):
        try:
            self.hand.disable()
        finally:
            self.enabled = False

    def close(self):
        try:
            if self.ff_state.startswith("constant"):
                self.hand.set_constant(self.sdk.FeedForwardMode.Off)
                self.hand.set_position_table(False)
        except Exception:
            pass
        for s in (self.sub_states, self.sub_diag):
            try:
                s.close()
            except Exception:
                pass
        try:
            self.pub.close()
        except Exception:
            pass
        try:
            self.mgr.disconnect_all()
        except Exception:
            pass


class FakeHand:
    """Second-order lag model with a little friction, for --no-hardware."""

    def __init__(self, *a, **k):
        self.fw = "fake"; self.ff_state = "fake"; self.q = np.zeros(20) + 0.05; self.dq = np.zeros(20); self.tau = np.zeros(20)
        self.last_state_t = time.perf_counter(); self.fault = None; self.warn = None; self.enabled = False
        self._t = None; self._target = None

    def poll(self):
        now = time.perf_counter()
        if self._t is not None and self._target is not None:
            dt = min(0.02, now - self._t)
            qd, vd = self._target
            acc = 400 * (qd - self.q) + 20 * (vd - self.dq) - 2.0 * np.sign(self.dq) * (np.abs(self.dq) > 0.02)
            self.dq += acc * dt; self.q += self.dq * dt; self.tau = 0.1 * (qd - self.q)
        self._t = now; self.last_state_t = now

    def wait_feedback(self, timeout=3.0):
        return True

    def enable(self):
        self.enabled = True

    def send(self, q, dq):
        self._target = (np.asarray(q, float), np.asarray(dq, float))

    def disable(self):
        self.enabled = False

    def close(self):
        pass


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand-sn", default=None)
    ap.add_argument("--hz", type=float, default=250.0, help="command rate")
    ap.add_argument("--kp", type=float, default=5.0)
    ap.add_argument("--kd", type=float, default=0.1)
    ap.add_argument("--effort", type=float, default=1.5, help="current limit, A")
    ap.add_argument("--ff", choices=["on", "off"], default="on", help="firmware friction feed-forward tables")
    ap.add_argument("--speed", type=float, default=1.0, help="time scale of the showcase (>1 faster)")
    ap.add_argument("--countdown", type=float, default=8.0)
    ap.add_argument("--no-hardware", action="store_true")
    ap.add_argument("--tag", default="smooth_demo")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, _sigint)
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    hand = FakeHand() if args.no_hardware else RealHand(args.hand_sn, args.kp, args.kd, args.effort, args.ff == "on")
    restore_timer = enable_fine_timer()
    margin = calibrate_sleep()
    print(f"[timing] {args.hz:.0f} Hz on perf_counter; sleep margin {margin*1e3:.2f} ms" if margin else "[timing] coarse sleep, spinning")
    if not hand.wait_feedback():
        raise SystemExit("no joint feedback")
    q_meas = hand.q.copy()
    print("[pose] measured (deg):", np.round(np.rad2deg(q_meas), 1).tolist())
    if np.any(q_meas < LIMITS[:, 0] - 0.05) or np.any(q_meas > LIMITS[:, 1] + 0.05):
        raise SystemExit("measured pose outside the joint limits — check the hand before moving")
    tl = build_showcase(q_meas, speed=args.speed)
    T = tl.t_end()
    print(f"[traj] showcase {T:.1f} s, {len(LABELS)} joints, minimum-jerk keyframes, velocity feedforward on")

    for s in range(int(args.countdown), 0, -1):
        if _STOP:
            print("stopped before start"); hand.close(); return
        print(f"[start] moving in {s} s — keep clear of the hand", flush=True)
        hand.poll(); time.sleep(1.0)

    hand.enable()
    print("[hand] joints enabled")
    dt = 1.0 / args.hz
    log_t, log_qc, log_dqc, log_qa, log_dqa, log_tau = [], [], [], [], [], []
    intervals = []
    over_since = np.full(20, np.nan)
    reason = "completed"
    t0 = time.perf_counter(); next_due = t0; last_send = None
    hand.warn = None
    warns = set()
    try:
        while True:
            t = time.perf_counter() - t0
            if t > T + 0.5:
                break
            if _STOP:
                reason = "stopped by user"; break
            q, dq = tl.sample(t)
            wait_until(next_due, margin)
            hand.send(q, dq)
            now = time.perf_counter()
            if last_send is not None:
                intervals.append(now - last_send)
            last_send = now
            next_due += dt
            if next_due < now - dt:           # fell behind: resync instead of bursting
                next_due = now
            hand.poll()
            if hand.last_state_t is None or now - hand.last_state_t > STALE_S:
                reason = "no joint feedback"; break
            if hand.fault:
                reason = f"fault: {hand.fault}"; break
            if getattr(hand, "warn", None):
                warns.add(hand.warn); hand.warn = None
            err = np.abs(q - hand.q)
            bad = err > ABORT_ERR_RAD
            over_since[~bad] = np.nan
            newly = bad & np.isnan(over_since)
            over_since[newly] = now
            lagging = bad & ((now - over_since) > ABORT_ERR_S)
            if np.any(lagging):
                j = int(np.argmax(lagging)); reason = f"{LABELS[j]} lags its target by {err[j]:.2f} rad (blocked?)"; break
            log_t.append(t); log_qc.append(q); log_dqc.append(dq); log_qa.append(hand.q.copy()); log_dqa.append(hand.dq.copy()); log_tau.append(hand.tau.copy())
    finally:
        # ease to the measured-start pose if we stopped mid-way, then disable
        try:
            if reason != "completed":
                qs = hand.q.copy(); ts = time.perf_counter()
                while time.perf_counter() - ts < 1.5:
                    s = min(1.0, (time.perf_counter() - ts) / 1.5); s = s * s * (3 - 2 * s)
                    hand.send(qs + s * (np.zeros(20) - qs), np.zeros(20)); time.sleep(0.004); hand.poll()
            hand.disable()
            print("[hand] joints disabled")
        except Exception as e:
            print(f"[hand] disable error: {e}")
        restore_timer()

    # ------------------------------------------------------------------ analysis
    qc, qa = np.array(log_qc), np.array(log_qa); dqc, dqa = np.array(log_dqc), np.array(log_dqa)
    summary = {"reason": reason, "firmware": hand.fw, "hz": args.hz, "kp": args.kp, "kd": args.kd, "effort_a": args.effort,
               "firmware_feedforward": hand.ff_state, "speed": args.speed, "showcase_s": round(T, 1), "ticks": len(log_t),
               "warnings_seen": sorted(warns)}
    if intervals:
        iv = np.array(intervals) * 1e3
        summary["send_interval_ms"] = {"target": round(dt * 1e3, 3), "mean": round(float(iv.mean()), 3), "stdev": round(float(iv.std()), 3),
                                       "p99": round(float(np.percentile(iv, 99)), 3), "max": round(float(iv.max()), 3)}
    if len(log_t) > 10:
        err = np.rad2deg(np.abs(qc - qa))
        moving = np.abs(dqc) > 0.15
        resid = dqa - dqc
        # stuck share: command moving but the joint moves < 25 % of the commanded speed
        stuck = (np.abs(dqa) < 0.25 * np.abs(dqc)) & moving
        summary["tracking_deg"] = {"mean": round(float(err.mean()), 2), "p95": round(float(np.percentile(err, 95)), 2), "max": round(float(err.max()), 2),
                                   "max_joint": LABELS[int(np.unravel_index(err.argmax(), err.shape)[1])]}
        summary["stuck_share_pct"] = round(100 * float(stuck.sum() / max(1, moving.sum())), 1)
        summary["velocity_residual_rms_rad_s"] = round(float(np.sqrt((resid[moving] ** 2).mean())), 3) if moving.any() else None
        per = {}
        for j in range(20):
            mv = moving[:, j]
            if mv.sum() > 20:
                per[LABELS[j]] = {"stuck_pct": round(100 * float(stuck[:, j].sum() / mv.sum()), 1),
                                  "vel_resid_rms": round(float(np.sqrt((resid[mv, j] ** 2).mean())), 3),
                                  "err_mean_deg": round(float(err[:, j].mean()), 2)}
        summary["per_joint"] = per
        np.savez_compressed(RUNS / f"{args.tag}_{stamp}.npz", t=np.array(log_t), q_cmd=qc, dq_cmd=dqc, q_act=qa, dq_act=dqa, effort=np.array(log_tau), labels=np.array(LABELS))
    (RUNS / f"{args.tag}_{stamp}.json").write_text(json.dumps(summary, indent=1))
    print("[result] " + json.dumps({k: v for k, v in summary.items() if k != "per_joint"}))
    print(f"[log] {RUNS / (args.tag + '_' + stamp)}.npz/.json")
    hand.close()


if __name__ == "__main__":
    main()
