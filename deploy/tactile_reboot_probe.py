#!/usr/bin/env python3
"""Reboot the Wuji Hand 2 so its master re-enumerates the finger buses, then re-check the
fingertip tactile modules. Writes runs/tactile/reboot_probe_<time>.txt.

The joints must be disabled (nothing else connected to the hand). The hand goes limp during
the reboot (~10-20 s) and comes back with the motors disabled.

    python deploy/tactile_reboot_probe.py [--hand-sn SN] [--wait 60]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sn", default=None)
    ap.add_argument("--wait", type=float, default=90.0, help="seconds to wait for the hand to come back")
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "tactile" / f"reboot_probe_{datetime.now():%Y-%m-%d_%H%M%S}.txt"))
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", encoding="utf-8")

    def log(*a):
        line = " ".join(str(x) for x in a); print(line, flush=True); fh.write(line + "\n"); fh.flush()

    import wuji_sdk
    mgr = wuji_sdk.SdkManager.instance()
    opts = wuji_sdk.ConnectOptions(enable_bridge=False)

    def tactile_report(hand, tag):
        try:
            no = int(hand.get("node_online"))
            bits = [(no >> (nid - 1)) & 1 for nid in range(1, 26)]
            tact = {f: bits[b * 5 + 4] for b, f in enumerate(FINGERS)}
            joints = sum(bits[b * 5 + j] for b in range(5) for j in range(4))
            log(f"[{tag}] node_online=0x{no:X}  joints online={joints}/20  tactile nodes={tact}")
        except Exception as e:
            log(f"[{tag}] node_online: ERROR {e}")
        try:
            log(f"[{tag}] fingertip_online={list(hand.fingertip_online())}")
        except Exception as e:
            log(f"[{tag}] fingertip_online: ERROR {e}")
        try:
            sub = hand.joint_diagnostics().subscribe(); t0 = time.time(); fr = None
            while fr is None and time.time() - t0 < 3:
                fr = sub.recv(); time.sleep(0.01)
            sub.close()
            if fr is not None:
                log(f"[{tag}] comm.tactile_online_mask={fr.comm.tactile_online_mask} timeouts={list(fr.comm.tactile_timeout_total)}")
        except Exception as e:
            log(f"[{tag}] joint_diagnostics: ERROR {e}")
        for f in FINGERS:
            try:
                st = getattr(hand, f"fingertip_{f}_status")().get()
                log(f"[{tag}] {f}: status model={st.model} state={st.state}")
            except Exception as e:
                log(f"[{tag}] {f}: status ERROR {type(e).__name__}: {str(e)[:80]}")
        subs = {f: getattr(hand, f"fingertip_{f}_data")().subscribe() for f in FINGERS}
        counts = {f: 0 for f in FINGERS}; t0 = time.time()
        while time.time() - t0 < 3:
            for f, s in subs.items():
                m = s.recv()
                if m is not None:
                    counts[f] += 1
            time.sleep(0.002)
        for s in subs.values():
            s.close()
        log(f"[{tag}] fingertip data frames in 3 s: {counts}")

    log(f"# reboot probe {datetime.now().isoformat(timespec='seconds')}")
    devs = mgr.scan()
    sn = args.hand_sn or (devs[0].sn if devs else None)
    if not sn:
        log("no hand found"); return 1
    hand = mgr.connect(sn=sn, device_name="wuji_hand_2", options=opts)
    log(f"connected {hand.serial_number}  fw={hand.get('firmware_version')}  hw={hand.hw_version().get()}")
    tactile_report(hand, "before")
    log("[reboot] sending reboot…")
    try:
        hand.reboot()
    except Exception as e:
        log(f"[reboot] reboot() raised {type(e).__name__}: {e} (the device usually drops the link while rebooting)")
    try:
        mgr.disconnect_all()
    except Exception:
        pass
    t0 = time.time(); hand = None
    time.sleep(8)
    while time.time() - t0 < args.wait:
        try:
            devs = [d for d in mgr.scan() if d.sn == sn]
            if devs:
                hand = mgr.connect(sn=sn, device_name="wuji_hand_2", options=opts)
                break
        except Exception as e:
            log(f"[reboot] waiting… ({type(e).__name__})")
        time.sleep(2)
    if hand is None:
        log(f"[reboot] hand did not come back within {args.wait:.0f} s — power-cycle it and run tactile_probe.bat")
        return 1
    log(f"[reboot] back after {time.time() - t0:.0f} s: {hand.serial_number}  fw={hand.get('firmware_version')}")
    time.sleep(3)
    tactile_report(hand, "after")
    try:
        mgr.disconnect_all()
    except Exception:
        pass
    log("# done:", out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc(); sys.exit(1)
