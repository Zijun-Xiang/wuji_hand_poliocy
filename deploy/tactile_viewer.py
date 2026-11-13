#!/usr/bin/env python3
"""Live viewer for the Wuji Hand 2 fingertip tactile sensors.

Connects to the hand over the Wuji SDK (`wuji_sdk`, Ethernet/USB), subscribes to the five
`fingertip_<finger>_data` streams and shows them live:

  top row     one panel per finger (thumb .. pinky): the taxel array as a heat map
              (grid when the sensor's format describes one, otherwise a bar chart), and
              the 3-axis force vector when the format contains one
  bottom row  the last 10 s of the per-finger total contact signal

The payload layout is NOT hard-coded: each finger's `fingertip_<finger>_info().get().format`
(a JSON string from the sensor) is parsed at start-up, printed, and saved together with a
few raw frames to runs/tactile/discovery_*.json, so a different sensor model/firmware can
be supported by extending `FormatDecoder` without guessing.

Keys in the window:  c = zero-calibrate all fingertips (no contact!)   r = reset colour scale
                     q = quit (also writes the log)

    python deploy/tactile_viewer.py                 # live, auto-detect the hand
    python deploy/tactile_viewer.py --hand-sn XXXX  # pin one hand
    python deploy/tactile_viewer.py --no-hardware   # synthetic data, to try the GUI
    python deploy/tactile_viewer.py --dump-only     # print formats + 5 raw frames, no GUI
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "tactile"


# --------------------------------------------------------------------------- format decoding
_DTYPES = {
    "u8": "u1", "uint8": "u1", "i8": "i1", "int8": "i1",
    "u16": "<u2", "uint16": "<u2", "i16": "<i2", "int16": "<i2",
    "u32": "<u4", "uint32": "<u4", "i32": "<i4", "int32": "<i4",
    "f32": "<f4", "float": "<f4", "float32": "<f4", "f64": "<f8", "float64": "<f8",
}


def _first(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


class FormatDecoder:
    """Turns a FingertipSensorData.data payload (list[int]) into named numpy arrays.

    Understood format JSON shapes (any of):
      {"dtype": "i16", "count": 24, "rows": 6, "cols": 4}
      {"fields": [{"name": "taxel", "dtype": "u16", "count": 40, "rows": 8, "cols": 5},
                  {"name": "force", "dtype": "f32", "count": 3}]}
      {"channels": [...]}, {"layout": [...]}, {"items": [...]}   (same as "fields")
      {"points": [[x, y], ...]}  or  {"taxels": [{"x":..,"y":..}, ...]}  -> scatter layout
    Anything else -> the raw integers are shown as one bar chart ("raw").
    """

    def __init__(self, format_str: str | None, model: str = ""):
        self.model = model
        self.raw_format = format_str or ""
        try:
            self.fmt = json.loads(format_str) if format_str else {}
        except Exception:
            self.fmt = {}
        if not isinstance(self.fmt, dict):
            self.fmt = {"fields": self.fmt} if isinstance(self.fmt, list) else {}
        self.endian = "<" if str(_first(self.fmt, "endian", "byte_order", default="little")).lower().startswith(("l", "<")) else ">"
        self.scale = float(_first(self.fmt, "scale", "lsb", default=1.0) or 1.0)
        self.unit = str(_first(self.fmt, "unit", "units", default=""))
        fields = _first(self.fmt, "fields", "channels", "layout", "items", "signals")
        self.fields: list[dict] = []
        if isinstance(fields, list) and fields and isinstance(fields[0], dict):
            off = 0
            for f in fields:
                dt = _DTYPES.get(str(_first(f, "dtype", "type", "format", default="i16")).lower(), "<i2")
                dt = self.endian + dt.lstrip("<>") if len(dt) > 2 else dt
                shape = _first(f, "shape")
                count = int(_first(f, "count", "len", "length", "n", "size", default=(int(np.prod(shape)) if shape else 1)))
                rows, cols = _first(f, "rows", "h", "height"), _first(f, "cols", "w", "width")
                if shape and len(shape) == 2 and rows is None:
                    rows, cols = shape
                self.fields.append(dict(name=str(_first(f, "name", "id", default=f"f{len(self.fields)}")), dtype=dt,
                                        count=count, offset=int(_first(f, "offset", default=off)),
                                        rows=rows, cols=cols, points=_first(f, "points", "coords", "xy"),
                                        scale=float(_first(f, "scale", default=1.0) or 1.0),
                                        unit=str(_first(f, "unit", default=""))))
                off = self.fields[-1]["offset"] + count * np.dtype(dt).itemsize
        elif self.fmt:
            dt = _DTYPES.get(str(_first(self.fmt, "dtype", "type", "value_type", "element_type", default="i16")).lower(), "<i2")
            shape = _first(self.fmt, "shape")
            rows, cols = _first(self.fmt, "rows", "h", "height"), _first(self.fmt, "cols", "w", "width")
            if shape and len(shape) == 2 and rows is None:
                rows, cols = shape
            points = _first(self.fmt, "points", "coords", "taxels", "xy")
            count = _first(self.fmt, "count", "len", "length", "n", "size", "num_points", "taxel_count")
            if count is None:
                count = (int(rows) * int(cols) if rows and cols else (len(points) if points else None))
            if count:
                self.fields.append(dict(name=str(_first(self.fmt, "name", default="taxel")), dtype=dt, count=int(count),
                                        offset=int(_first(self.fmt, "offset", default=0)), rows=rows, cols=cols,
                                        points=points, scale=self.scale, unit=self.unit))
                # a trailing 3-axis force block if the format names one
                fk = _first(self.fmt, "force", "force_offset")
                if isinstance(fk, dict):
                    fdt = _DTYPES.get(str(_first(fk, "dtype", "type", default="f32")).lower(), "<f4")
                    self.fields.append(dict(name="force", dtype=fdt, count=3,
                                            offset=int(_first(fk, "offset", default=self.fields[0]["offset"] + int(count) * np.dtype(dt).itemsize)),
                                            rows=None, cols=None, points=None, scale=float(_first(fk, "scale", default=1.0) or 1.0), unit=str(_first(fk, "unit", default=""))))
        self.bytes_needed = max((f["offset"] + f["count"] * np.dtype(f["dtype"]).itemsize for f in self.fields), default=0)
        self.mode = "fields" if self.fields else "raw"

    # which field is the taxel map, which is the force vector
    def taxel_field(self) -> dict | None:
        for f in self.fields:
            n = f["name"].lower()
            if any(k in n for k in ("taxel", "pressure", "normal", "array", "map", "grid", "cell", "matrix", "value")):
                return f
        best = max(self.fields, key=lambda f: f["count"], default=None)
        return best if best and best["count"] >= 4 else None

    def force_field(self) -> dict | None:
        for f in self.fields:
            n = f["name"].lower()
            if f["count"] == 3 and any(k in n for k in ("force", "fxyz", "f_", "wrench", "vector")):
                return f
        for f in self.fields:
            if f["count"] == 3 and f is not self.taxel_field():
                return f
        return None

    def decode(self, data) -> dict[str, np.ndarray]:
        raw = np.asarray(data)
        out: dict[str, np.ndarray] = {}
        if self.mode == "fields" and raw.size:
            as_bytes = raw.dtype.kind in "iu" and raw.size >= self.bytes_needed and raw.min() >= 0 and raw.max() <= 255 and self.bytes_needed > 0
            if as_bytes:
                buf = bytes(raw.astype(np.uint8).tolist())
                for f in self.fields:
                    n = f["count"]
                    arr = np.frombuffer(buf, dtype=f["dtype"], count=n, offset=f["offset"]).astype(np.float64)
                    out[f["name"]] = arr * f["scale"]
            else:
                # the SDK already delivered values (one int per element)
                pos = 0
                for f in self.fields:
                    n = f["count"]
                    seg = raw[pos:pos + n].astype(np.float64)
                    if seg.size < n:
                        seg = np.pad(seg, (0, n - seg.size))
                    out[f["name"]] = seg * f["scale"]
                    pos += n
        if not out:
            out["raw"] = raw.astype(np.float64)
        return out


def guess_grid(n: int) -> tuple[int, int] | None:
    """A plausible rows x cols for n taxels (prefers near-square, cols <= rows)."""
    best = None
    for c in range(1, int(math.sqrt(n)) + 1):
        if n % c == 0:
            r = n // c
            if best is None or abs(r - c) < abs(best[0] - best[1]):
                best = (r, c)
    return best if best and best[1] > 1 else None


# --------------------------------------------------------------------------- data sources
class FingerStream:
    def __init__(self, name: str):
        self.name = name
        self.online = False
        self.model = ""
        self.state = ""
        self.rate_hz = 0.0
        self.decoder = FormatDecoder(None)
        self.latest: dict[str, np.ndarray] | None = None
        self.latest_t = 0.0
        self.count = 0
        self.log_t: list[float] = []
        self.log_raw: list[np.ndarray] = []
        self.samples_raw: list[list[int]] = []     # first few raw payloads, for discovery.json
        self.hz_window: deque[float] = deque(maxlen=50)

    def push(self, data, t: float, keep_log: bool):
        raw = np.asarray(data)
        self.latest = self.decoder.decode(raw)
        self.latest_t = t
        self.count += 1
        self.hz_window.append(t)
        if len(self.samples_raw) < 5:
            self.samples_raw.append([int(v) for v in raw.tolist()])
        if keep_log:
            self.log_t.append(t)
            self.log_raw.append(raw.astype(np.int32))

    def hz(self) -> float:
        if len(self.hz_window) < 2:
            return 0.0
        return (len(self.hz_window) - 1) / max(1e-6, self.hz_window[-1] - self.hz_window[0])


class RealHand:
    """Wuji Hand 2 over wuji_sdk: discovery, per-finger info/status, subscriptions."""

    def __init__(self, serial_number: str | None, rate_hz: int):
        import wuji_sdk
        self.sdk = wuji_sdk
        self.mgr = wuji_sdk.SdkManager.instance()
        sn = serial_number
        if not sn:
            devs = self.mgr.scan()
            if not devs:
                raise RuntimeError("no Wuji hand found (USB/UDP) — is it powered and on this network?")
            if len(devs) > 1:
                raise RuntimeError("several hands found: " + ", ".join(f"sn={d.sn} ip={d.ip}" for d in devs) + " — pass --hand-sn")
            sn = devs[0].sn
            print(f"[hand] found sn={sn} ip={devs[0].ip} transport={devs[0].transport_type}")
        self.hand = self.mgr.connect(sn=sn, device_name="wuji_hand_2", options=wuji_sdk.ConnectOptions(enable_bridge=False))
        print(f"[hand] connected: {self.hand.serial_number}")
        self.streams = {f: FingerStream(f) for f in FINGERS}
        try:
            online = list(self.hand.fingertip_online())
        except Exception as e:  # older firmware
            print(f"[hand] fingertip_online() failed ({e}); assuming all five")
            online = [True] * 5
        self.subs = {}
        for f, on in zip(FINGERS, online):
            s = self.streams[f]
            s.online = bool(on)
            if not on:
                print(f"[{f:6s}] tactile module offline")
                continue
            try:
                st = getattr(self.hand, f"fingertip_{f}_status")().get()
                s.model, s.state = str(st.model), str(st.state)
            except Exception as e:
                s.model, s.state = "?", f"status error: {e}"
            try:
                info = getattr(self.hand, f"fingertip_{f}_info")().get()
                s.rate_hz = float(info.rate_hz)
                s.decoder = FormatDecoder(info.format, model=str(info.model))
                print(f"[{f:6s}] model={info.model} type={info.device_type} rate={info.rate_hz} Hz digest={info.digest}")
                print(f"[{f:6s}] format={info.format}")
            except Exception as e:
                print(f"[{f:6s}] info read failed: {e}")
                s.decoder = FormatDecoder(None)
            sub = getattr(self.hand, f"fingertip_{f}_data")().subscribe()
            if rate_hz > 0:
                try:
                    applied = sub.set_rate(int(rate_hz))
                    print(f"[{f:6s}] stream rate set to {applied} Hz")
                except Exception as e:
                    print(f"[{f:6s}] rate control not available ({e}); device default")
            self.subs[f] = sub

    def poll(self, keep_log: bool) -> int:
        n = 0
        now = time.time()
        for f, sub in self.subs.items():
            for _ in range(64):              # drain what has arrived since the last poll
                msg = sub.recv()
                if msg is None:
                    break
                self.streams[f].push(msg.data, now, keep_log)
                n += 1
        return n

    def calibrate(self):
        self.hand.calibrate_fingertip()

    def close(self):
        for sub in self.subs.values():
            try:
                sub.close()
            except Exception:
                pass
        try:
            self.mgr.disconnect_all()
        except Exception:
            pass


class FakeHand:
    """Synthetic fingertips for --no-hardware: a 6x4 (thumb 8x5) i16 taxel grid + 3-axis force."""

    def __init__(self, rate_hz: int):
        self.streams = {f: FingerStream(f) for f in FINGERS}
        self.rate = rate_hz or 100
        self.t0 = time.time()
        self.next_t = self.t0
        for f in FINGERS:
            rows, cols = (8, 5) if f == "thumb" else (6, 4)
            fmt = json.dumps({"fields": [{"name": "taxel", "dtype": "i16", "count": rows * cols, "rows": rows, "cols": cols, "unit": "raw"},
                                         {"name": "force", "dtype": "f32", "count": 3, "unit": "N"}]})
            s = self.streams[f]
            s.online, s.model, s.state, s.rate_hz = True, "Thumb" if f == "thumb" else "Standard", "Ready", float(self.rate)
            s.decoder = FormatDecoder(fmt, model=s.model)

    def poll(self, keep_log: bool) -> int:
        n = 0
        now = time.time()
        while self.next_t <= now:
            t = self.next_t - self.t0
            for i, f in enumerate(FINGERS):
                s = self.streams[f]
                rows, cols = s.decoder.fields[0]["rows"], s.decoder.fields[0]["cols"]
                yy, xx = np.mgrid[0:rows, 0:cols]
                cx, cy = cols / 2 + 1.2 * math.sin(0.7 * t + i), rows / 2 + 1.5 * math.cos(0.5 * t + i)
                amp = 2000 * max(0.0, math.sin(0.9 * t + 1.3 * i))
                tax = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 2.5) + np.random.randn(rows, cols) * 15
                force = np.array([0.3 * math.sin(t + i), 0.2 * math.cos(1.3 * t + i), amp / 1000.0], dtype=np.float32)
                payload = struct.pack(f"<{rows * cols}h", *np.clip(tax, -32768, 32767).astype(np.int16).ravel()) + force.tobytes()
                s.push(list(payload), self.next_t, keep_log)
                n += 1
            self.next_t += 1.0 / self.rate
        return n

    def calibrate(self):
        print("[fake] calibrate (no-op)")

    def close(self):
        pass


# --------------------------------------------------------------------------- GUI
def run_gui(hand, args):
    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams["toolbar"] = "None"
    fig = plt.figure(figsize=(15, 7.2), facecolor="#111318")
    fig.canvas.manager.set_window_title("Wuji Hand 2 — fingertip tactile sensors") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None
    gs = fig.add_gridspec(2, 5, height_ratios=[3.2, 1.4], hspace=0.42, wspace=0.28, left=0.04, right=0.985, top=0.9, bottom=0.08)
    axes_map = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax_ts = fig.add_subplot(gs[1, :])
    for ax in axes_map + [ax_ts]:
        ax.set_facecolor("#1b1e26")
        for sp in ax.spines.values():
            sp.set_color("#3a3f4b")
        ax.tick_params(colors="#9aa0ad", labelsize=8)
    title = fig.text(0.5, 0.955, "Wuji Hand 2 fingertip tactile — connecting…", ha="center", va="center", color="#e8e8ec", fontsize=13, weight="bold")
    help_txt = fig.text(0.985, 0.012, "keys:  c = zero-calibrate (no contact!)   r = reset colour scale   q = quit", ha="right", va="bottom", color="#9aa0ad", fontsize=9)

    hist_len = int(10 * 30)
    hist_t: deque[float] = deque(maxlen=hist_len)
    hist = {f: deque(maxlen=hist_len) for f in FINGERS}
    colors = ["#ff6b6b", "#ffb347", "#8ce99a", "#66d9e8", "#b197fc"]
    lines = {}
    for f, c in zip(FINGERS, colors):
        (lines[f],) = ax_ts.plot([], [], color=c, lw=1.6, label=f)
    ax_ts.legend(loc="upper left", ncol=5, fontsize=8, frameon=False, labelcolor="#e8e8ec")
    ax_ts.set_xlabel("time (s)", color="#9aa0ad", fontsize=9)
    ax_ts.set_ylabel("total contact (sum |taxel|)", color="#9aa0ad", fontsize=9)
    ax_ts.set_xlim(-10, 0)

    artists: dict[str, dict] = {}
    vmax = {f: 1.0 for f in FINGERS}
    t_start = time.time()
    state = {"quit": False, "last_cal": 0.0}

    def on_key(ev):
        if ev.key == "q":
            state["quit"] = True
            plt.close(fig)
        elif ev.key == "c":
            try:
                hand.calibrate()
                state["last_cal"] = time.time()
                print("[cal] fingertip zero-calibration requested (sensors must be unloaded)")
            except Exception as e:
                print(f"[cal] failed: {e}")
        elif ev.key == "r":
            for f in FINGERS:
                vmax[f] = 1.0

    fig.canvas.mpl_connect("key_press_event", on_key)

    def build_panel(f: str, ax, s: FingerStream):
        ax.clear()
        ax.set_facecolor("#1b1e26")
        ax.set_title(f, color="#e8e8ec", fontsize=11, pad=6)
        d = {"kind": "none"}
        if not s.online:
            ax.text(0.5, 0.5, "offline", ha="center", va="center", color="#777", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            return d
        dec = s.decoder
        tf = dec.taxel_field()
        if tf is None:
            n = len(s.latest["raw"]) if s.latest and "raw" in s.latest else 1
            bars = ax.bar(np.arange(n), np.zeros(n), color="#66d9e8")
            ax.set_xlim(-0.5, n - 0.5)
            d = {"kind": "bar", "bars": bars, "key": "raw"}
        else:
            n = tf["count"]
            if tf["points"]:
                pts = np.array([[p["x"], p["y"]] if isinstance(p, dict) else p[:2] for p in tf["points"]], dtype=float)
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.zeros(n), s=260, cmap="inferno", vmin=0, vmax=1, edgecolors="#333")
                ax.set_aspect("equal"); ax.invert_yaxis()
                d = {"kind": "scatter", "sc": sc, "key": tf["name"]}
            else:
                rc = (int(tf["rows"]), int(tf["cols"])) if tf["rows"] and tf["cols"] else guess_grid(n)
                if rc:
                    im = ax.imshow(np.zeros(rc), cmap="inferno", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
                    ax.set_xticks([]); ax.set_yticks([])
                    d = {"kind": "grid", "im": im, "shape": rc, "key": tf["name"]}
                else:
                    bars = ax.bar(np.arange(n), np.zeros(n), color="#66d9e8")
                    ax.set_xlim(-0.5, n - 0.5)
                    d = {"kind": "bar", "bars": bars, "key": tf["name"]}
        ff = dec.force_field()
        if ff is not None:
            d["force_key"] = ff["name"]
            d["force_txt"] = ax.text(0.02, 0.02, "", transform=ax.transAxes, color="#ffd166", fontsize=8, va="bottom", family="monospace")
            d["force_unit"] = ff["unit"]
        d["info"] = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="#9aa0ad", fontsize=7.5, va="top", family="monospace")
        return d

    def update(_frame):
        hand.poll(keep_log=bool(args.log))
        now = time.time()
        hist_t.append(now - t_start)
        for f, ax in zip(FINGERS, axes_map):
            s = hand.streams[f]
            if f not in artists:
                if s.online and s.latest is None:
                    continue                  # wait for the first frame to know the array size
                artists[f] = build_panel(f, ax, s)
            a = artists[f]
            if s.latest is None:
                hist[f].append(0.0)
                continue
            key = a.get("key")
            vals = s.latest.get(key) if key else None
            total = float(np.abs(vals).sum()) if vals is not None else 0.0
            hist[f].append(total)
            if vals is not None:
                mag = np.abs(vals)
                # per-finger running scale, but never below 15 % of the strongest finger so idle
                # fingertips stay dark instead of amplifying their noise
                vmax[f] = max(vmax[f] * 0.995, float(np.percentile(mag, 99)) * 1.05, 1.0)
                floor = 0.15 * max(vmax.values())
                vmax[f] = max(vmax[f], floor)
                if a["kind"] == "grid":
                    r, c = a["shape"]
                    g = mag[: r * c]
                    if g.size < r * c:
                        g = np.pad(g, (0, r * c - g.size))
                    a["im"].set_data(g.reshape(r, c)); a["im"].set_clim(0, vmax[f])
                elif a["kind"] == "scatter":
                    a["sc"].set_array(mag); a["sc"].set_clim(0, vmax[f])
                elif a["kind"] == "bar":
                    for b, v in zip(a["bars"], vals):
                        b.set_height(v)
                    ax.set_ylim(-vmax[f] * 1.1, vmax[f] * 1.1)
            if "force_key" in a and a["force_key"] in s.latest:
                fx, fy, fz = s.latest[a["force_key"]][:3]
                a["force_txt"].set_text(f"F=({fx:+.2f},{fy:+.2f},{fz:+.2f}) {a['force_unit']}  |F|={math.sqrt(fx*fx+fy*fy+fz*fz):.2f}")
            a["info"].set_text(f"{s.model} {s.state}\n{s.hz():.0f} Hz  n={s.count}  max={vmax[f]:.0f}")
        # time series
        if hist_t:
            t_now = hist_t[-1]
            ts = np.array(hist_t) - t_now
            ymax = 1.0
            for f in FINGERS:
                h = np.array(hist[f])
                lines[f].set_data(ts[-len(h):], h)
                if h.size:
                    ymax = max(ymax, float(h.max()))
            ax_ts.set_ylim(0, ymax * 1.1)
        online = [hand.streams[f] for f in FINGERS if hand.streams[f].online]
        rates = ", ".join(f"{s.name} {s.hz():.0f}Hz" for s in online)
        cal = "  (calibrating…)" if now - state["last_cal"] < 3 else ""
        title.set_text(f"Wuji Hand 2 fingertip tactile — {len(online)}/5 modules online — {rates}{cal}")
        return []

    ani = FuncAnimation(fig, update, interval=int(1000 / args.fps), cache_frame_data=False)
    if args.headless:
        for _ in range(int(args.headless_frames)):
            update(0)
            time.sleep(1.0 / args.fps)
        out = RUNS / "viewer_headless.png"
        fig.savefig(out, dpi=110, facecolor=fig.get_facecolor())
        print(f"[headless] saved {out}")
    else:
        plt.show()
    return ani


# --------------------------------------------------------------------------- main
def write_discovery(hand, path: Path):
    doc = {"time": datetime.now().isoformat(timespec="seconds"), "fingers": {}}
    for f in FINGERS:
        s = hand.streams[f]
        doc["fingers"][f] = {"online": s.online, "model": s.model, "state": s.state, "rate_hz": s.rate_hz,
                             "format": s.decoder.raw_format, "parsed_fields": s.decoder.fields, "decode_mode": s.decoder.mode,
                             "sample_raw_payloads": s.samples_raw,
                             "sample_decoded": ({k: np.round(v, 3).tolist() for k, v in s.latest.items()} if s.latest else None)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    print(f"[discovery] wrote {path}")


def write_log(hand, path: Path):
    arrays = {}
    for f in FINGERS:
        s = hand.streams[f]
        if s.log_raw:
            n = min(len(a) for a in s.log_raw)
            arrays[f"{f}_t"] = np.array(s.log_t)
            arrays[f"{f}_raw"] = np.stack([a[:n] for a in s.log_raw])
            arrays[f"{f}_format"] = np.array(s.decoder.raw_format)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[log] wrote {path} ({sum(v.shape[0] for k, v in arrays.items() if k.endswith('_raw'))} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand-sn", default=None, help="serial number, when several hands are on the bus")
    ap.add_argument("--hz", type=int, default=0, help="requested device push rate per stream (0 = device default)")
    ap.add_argument("--fps", type=float, default=25.0, help="GUI refresh rate")
    ap.add_argument("--log", default="auto", help="npz to write at exit ('auto' = runs/tactile/tactile_<time>.npz, '' = none)")
    ap.add_argument("--no-hardware", action="store_true", help="synthetic fingertips (GUI check without the hand)")
    ap.add_argument("--dump-only", action="store_true", help="print formats and 5 raw frames per finger, then exit")
    ap.add_argument("--headless", action="store_true", help="render N frames to a PNG instead of a window (tests)")
    ap.add_argument("--headless-frames", type=int, default=60)
    args = ap.parse_args()
    if args.log == "auto":
        args.log = str(RUNS / f"tactile_{datetime.now():%Y-%m-%d_%H%M%S}.npz")

    hand = FakeHand(args.hz) if args.no_hardware else RealHand(args.hand_sn, args.hz)
    try:
        # collect a few frames before the GUI so the panels know the array sizes
        t0 = time.time()
        while time.time() - t0 < 2.0 and not all(hand.streams[f].latest is not None for f in FINGERS if hand.streams[f].online):
            hand.poll(keep_log=False)
            time.sleep(0.01)
        for f in FINGERS:
            s = hand.streams[f]
            if s.online:
                print(f"[{f:6s}] first frame: {len(s.samples_raw[0]) if s.samples_raw else 0} ints; decoded keys: "
                      f"{ {k: v.shape for k, v in s.latest.items()} if s.latest else 'none yet'}")
        write_discovery(hand, RUNS / f"discovery_{datetime.now():%Y-%m-%d_%H%M%S}.json")
        if args.dump_only:
            for f in FINGERS:
                s = hand.streams[f]
                for i, p in enumerate(s.samples_raw):
                    print(f"[{f:6s}] raw[{i}] ({len(p)}): {p[:48]}{' …' if len(p) > 48 else ''}")
            return
        run_gui(hand, args)
    finally:
        if args.log:
            try:
                write_log(hand, Path(args.log))
            except Exception as e:
                print(f"[log] not written: {e}")
        hand.close()


if __name__ == "__main__":
    main()
