"""Joint/bus health probe for the Wuji Hand 2: which joints are online, per-bus node mask,
fault codes, and (optionally) a clear_fault + soft reboot to try to bring a dropped bus back.
Writes runs/real_hand/joint_probe_<time>.txt. Motors stay disabled throughout.

    python deploy/joint_probe.py                 # read-only
    python deploy/joint_probe.py --clear-fault   # also try hand.clear_fault()
    python deploy/joint_probe.py --reboot        # also soft-reboot the hand and re-check (hand goes limp ~15 s)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def safe(log, label, fn):
    try:
        v = fn()
        log(f"{label}: {v!r}")
        return v
    except Exception as e:  # noqa: BLE001
        log(f"{label}: ERROR {type(e).__name__}: {e}")
        return None


def decode_mask(log, mask):
    if not isinstance(mask, int):
        return
    log(f"node_online = 0x{mask:X}")
    for b, f in enumerate(FINGERS):
        bits = [(mask >> (b * 5 + n)) & 1 for n in range(5)]
        joints = "".join("S%d%s " % (n + 1, "" if bits[n] else "(OFF)") for n in range(4))
        log(f"  bus {b} {f:6s}: {joints} tactile={'on' if bits[4] else 'off'}")


def snapshot(log, hand, tag):
    log(f"--- {tag}")
    safe(log, "online_joints_count", lambda: hand.online_joints_count().get())
    mask = safe(log, "get('node_online')", lambda: hand.get("node_online"))
    decode_mask(log, mask)
    for p in ("status", "health_status", "input_voltage", "board_temperature", "error_status"):
        safe(log, f"get({p!r})", lambda p=p: hand.get(p))
    for f in FINGERS:
        for leaf in ("S1.node_info", "S1.error_status", "S1.hw_version", "S1.echo"):
            safe(log, f"get('{f}.{leaf}')", lambda p=f"{f}.{leaf}": hand.get(p))
    safe(log, "comm_diag.get", lambda: hand.comm_diag().get())
    # one joint_diagnostics frame: per-joint errors / comm counters
    try:
        sub = hand.joint_diagnostics().subscribe()
        t0 = time.time()
        frame = None
        while time.time() - t0 < 3 and frame is None:
            frame = sub.recv()
            time.sleep(0.01)
        sub.close()
        if frame is None:
            log("joint_diagnostics: no frame in 3 s")
        else:
            for a in dir(frame):
                if a.startswith("_"):
                    continue
                v = getattr(frame, a)
                if callable(v):
                    continue
                log(f"  diag.{a} = {v!r}"[:600])
    except Exception as e:  # noqa: BLE001
        log(f"joint_diagnostics: ERROR {e}")
    # one joint_states frame: which joints report
    try:
        sub = hand.joint_states().subscribe()
        t0 = time.time()
        js = None
        while time.time() - t0 < 2 and js is None:
            js = sub.recv()
            time.sleep(0.01)
        sub.close()
        if js is None:
            log("joint_states: no frame in 2 s")
        else:
            for a in ("position", "velocity", "effort", "online", "error", "error_code", "temperature"):
                if hasattr(js, a):
                    log(f"  joint_states.{a} = {list(getattr(js, a))!r}"[:600])
    except Exception as e:  # noqa: BLE001
        log(f"joint_states: ERROR {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sn", default=None)
    ap.add_argument("--clear-fault", action="store_true")
    ap.add_argument("--reboot", action="store_true")
    ap.add_argument("--wait", type=float, default=60.0)
    ap.add_argument("--out", default=str(HERE.parent / "runs" / "real_hand" / f"joint_probe_{datetime.now():%Y-%m-%d_%H%M%S}.txt"))
    args = ap.parse_args()
    log = Tee(Path(args.out))
    log(f"# joint probe {datetime.now().isoformat(timespec='seconds')}")
    import wuji_sdk

    log("wuji_sdk", getattr(wuji_sdk, "__version__", "?"))
    mgr = wuji_sdk.SdkManager.instance()
    opts = wuji_sdk.ConnectOptions(enable_bridge=False)
    devs = mgr.scan()
    for d in devs:
        log(f"scan: sn={d.sn} ip={getattr(d, 'ip', '?')} transport={getattr(d, 'transport_type', '?')}")
    sn = args.hand_sn or (devs[0].sn if devs else None)
    if not sn:
        log("no hand found")
        return 1
    hand = mgr.connect(sn=sn, device_name="wuji_hand_2", options=opts)
    log(f"connected {hand.serial_number}  fw={hand.get('firmware_version')}  hw={hand.hw_version().get()}")
    snapshot(log, hand, "initial")
    if args.clear_fault:
        safe(log, "clear_fault()", lambda: hand.clear_fault())
        time.sleep(1.0)
        snapshot(log, hand, "after clear_fault")
    if args.reboot:
        log("[reboot] sending reboot… (hand goes limp for ~15 s)")
        try:
            hand.reboot()
        except Exception as e:  # noqa: BLE001
            log(f"[reboot] reboot() raised {type(e).__name__}: {e}")
        try:
            mgr.disconnect_all()
        except Exception:  # noqa: BLE001
            pass
        t0 = time.time()
        time.sleep(8)
        hand = None
        while time.time() - t0 < args.wait:
            try:
                if [d for d in mgr.scan() if d.sn == sn]:
                    hand = mgr.connect(sn=sn, device_name="wuji_hand_2", options=opts)
                    break
            except Exception as e:  # noqa: BLE001
                log(f"[reboot] waiting… ({type(e).__name__})")
            time.sleep(2)
        if hand is None:
            log(f"[reboot] hand did not come back within {args.wait:.0f} s — power-cycle it")
            return 2
        log(f"[reboot] back after {time.time() - t0:.0f} s")
        time.sleep(3)
        snapshot(log, hand, "after reboot")
    try:
        mgr.disconnect_all()
    except Exception:  # noqa: BLE001
        pass
    log(f"# written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
