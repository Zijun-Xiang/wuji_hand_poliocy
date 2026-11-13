#!/usr/bin/env python3
"""Probe the Wuji Hand 2 for fingertip tactile modules and write everything to a text file.

Used when tactile_viewer.py reports the fingertips offline: dumps hardware/firmware versions,
the device's node and topic lists, fingertip_online(), each finger's status/info calls (with
the exact error if any), the frame-level communication summary (tactile_online_mask,
per-finger tactile response rates) and a 3 s count of frames on each fingertip data stream.

    python deploy/tactile_probe.py [--hand-sn SN] [--out runs/tactile/probe.txt]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HERE = Path(__file__).resolve().parent


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line)
        self.f.write(line + "\n")
        self.f.flush()


def safe(log, label, fn):
    try:
        v = fn()
        log(f"{label}: {v!r}")
        return v
    except Exception as e:
        log(f"{label}: ERROR {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sn", default=None)
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "tactile" / f"probe_{datetime.now():%Y-%m-%d_%H%M%S}.txt"))
    args = ap.parse_args()
    log = Tee(Path(args.out))
    log(f"# tactile probe {datetime.now().isoformat(timespec='seconds')}")
    import wuji_sdk
    log("wuji_sdk", getattr(wuji_sdk, "__version__", "?"))
    mgr = wuji_sdk.SdkManager.instance()
    devs = mgr.scan()
    for d in devs:
        log(f"scan: sn={d.sn} ip={getattr(d, 'ip', '?')} transport={getattr(d, 'transport_type', '?')} type={getattr(d, 'device_type', '?')}")
    sn = args.hand_sn or (devs[0].sn if devs else None)
    if not sn:
        log("no hand found")
        return
    hand = mgr.connect(sn=sn, device_name="wuji_hand_2", options=wuji_sdk.ConnectOptions(enable_bridge=False))
    log(f"connected: {hand.serial_number} ({type(hand).__name__})")
    try:
        safe(log, "hw_version", lambda: hand.hw_version().get())
        safe(log, "firmware_version (get)", lambda: hand.get("firmware_version"))
        safe(log, "online_joints_count", lambda: hand.online_joints_count().get())
        nodes = safe(log, "nodes", lambda: [n.path for n in hand.nodes()])
        topics = safe(log, "topics", lambda: [getattr(t, "path", getattr(t, "name", repr(t))) for t in hand.topics()])
        params = safe(log, "params", lambda: [getattr(p, "path", getattr(p, "name", repr(p))) for p in hand.params()])
        resources = safe(log, "resources", lambda: [getattr(r, "path", repr(r)) for r in hand.resources()])
        online = safe(log, "fingertip_online", lambda: list(hand.fingertip_online()))
        for f in FINGERS:
            safe(log, f"fingertip_{f}_status.get", lambda f=f: getattr(hand, f"fingertip_{f}_status")().get())
            info = safe(log, f"fingertip_{f}_info.get", lambda f=f: getattr(hand, f"fingertip_{f}_info")().get())
            if info is not None:
                for k in ("model", "device_type", "rate_hz", "digest", "format"):
                    log(f"   {f}.{k} = {getattr(info, k, '?')!r}")
        # communication summary from one joint-diagnostics frame
        try:
            sub = hand.joint_diagnostics().subscribe()
            t0 = time.time(); frame = None
            while time.time() - t0 < 3 and frame is None:
                frame = sub.recv(); time.sleep(0.01)
            sub.close()
            if frame is None:
                log("joint_diagnostics: no frame in 3 s")
            else:
                cs = getattr(frame, "comm", None) or getattr(frame, "summary", None) or getattr(frame, "comm_summary", None)
                log(f"joint_diagnostics frame attrs: {[a for a in dir(frame) if not a.startswith('_')]}")
                for a in dir(frame):
                    if "comm" in a.lower() or "summary" in a.lower() or "tactile" in a.lower():
                        log(f"   frame.{a} = {getattr(frame, a)!r}")
                if cs is not None:
                    for a in ("tactile_online_mask", "tactile_response_rate_pct", "tactile_timeout_total", "age_ms"):
                        log(f"   comm.{a} = {getattr(cs, a, '?')!r}")
        except Exception as e:
            log(f"joint_diagnostics: ERROR {e}")
        safe(log, "comm_diag.get", lambda: hand.comm_diag().get())
        # count fingertip frames for 3 s regardless of the online flag
        subs = {}
        for f in FINGERS:
            try:
                subs[f] = getattr(hand, f"fingertip_{f}_data")().subscribe()
                log(f"fingertip_{f}_data: subscribed (dtype={getattr(subs[f], 'dtype', '?')})")
            except Exception as e:
                log(f"fingertip_{f}_data: subscribe ERROR {e}")
        counts = {f: 0 for f in subs}; first = {}
        t0 = time.time()
        while time.time() - t0 < 3:
            for f, s in subs.items():
                try:
                    m = s.recv()
                except Exception as e:
                    log(f"fingertip_{f}_data: recv ERROR {e}"); subs[f] = None; continue
                if m is not None:
                    counts[f] += 1
                    if f not in first:
                        first[f] = (len(m.data), list(m.data)[:64], getattr(m, "info_digest", None))
            subs = {f: s for f, s in subs.items() if s is not None}
            time.sleep(0.002)
        for f in FINGERS:
            log(f"fingertip_{f}_data: {counts.get(f, 0)} frames in 3 s; first={first.get(f)}")
        for s in subs.values():
            try:
                s.close()
            except Exception:
                pass
        # generic subscribe by path, in case the semantic accessors hide a node
        for path in ("thumb/fingertip/data", "fingertip/thumb/data", "sensor/fingertip_thumb"):
            safe(log, f"subscribe({path!r})", lambda p=path: hand.subscribe(p))
    finally:
        try:
            mgr.disconnect_all()
        except Exception:
            pass
    log("# done; file:", args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
