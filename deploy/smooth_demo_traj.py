"""Choreography for the smooth-motion showcase on the Wuji Hand 2 (right hand).

Joint order is finger-major, 4 per finger (same as the SDK / wuji-mjlab):
  thumb  : cmc_flex, cmc_abd, mcp, ip
  fingers: mcp_flex, mcp_abd, pip, dip      (index, middle, ring, pinky)
Positive = flexion for flex/pip/dip/mcp/ip; thumb cmc_abd positive = toward the fingers
(opposition); finger mcp_abd positive = toward the +x side of the hand (see SPREAD below,
measured in the MJCF). All values in radians.

Every joint has its own list of keyframes (t, q); between keyframes the joint follows a
minimum-jerk (最小加加速度) polynomial, so position, velocity and acceleration are continuous
and velocity is zero at every keyframe. Overlapping finger motions are simply keyframes at
different times on different joints.
"""
from __future__ import annotations

import numpy as np

LIMITS = np.array([
    [-1.187, 1.291], [-1.484, 0.698], [-1.047, 1.570], [-1.047, 1.570],
    [-1.047, 1.570], [-0.698, 0.698], [-1.047, 2.094], [-1.047, 1.570],
    [-1.047, 1.570], [-0.698, 0.698], [-1.047, 2.094], [-1.047, 1.570],
    [-1.047, 1.570], [-0.698, 0.698], [-1.047, 2.094], [-1.047, 1.570],
    [-1.047, 1.570], [-0.698, 0.698], [-1.047, 2.094], [-1.047, 1.570],
])
LABELS = [f"{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky") for j in ("j1", "j2", "j3", "j4")]
FINGER = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}
# spread direction per finger for mcp_abd measured on the MJCF: index +abd moves toward the middle finger; ring and pinky
# +abd move away from it, the pinky needs more angle for the same gap.
SPREAD = {"index": -0.45, "middle": -0.15, "ring": +0.25, "pinky": +0.60}   # rad at full spread; even ~7 cm tip gaps
# flexion pose of a finger at "amount" a in [0, 1]
CURL = {"mcp_flex": 0.80, "pip": 1.25, "dip": 0.85}
THUMB_CURL = {"cmc_flex": 0.45, "mcp": 0.75, "ip": 0.75}
MARGIN = 0.05


def smoothstep5(s: np.ndarray | float):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s), 30 * s * s * (1 - s) * (1 - s)  # value, d/ds


class Timeline:
    def __init__(self, q0: np.ndarray):
        self.keys = [[(0.0, float(q0[j]))] for j in range(20)]

    def t_end(self) -> float:
        return max(k[-1][0] for k in self.keys)

    def last(self, j: int) -> float:
        return self.keys[j][-1][1]

    def move(self, j: int, t_start: float, duration: float, q: float):
        """Hold the joint at its last value until t_start, then reach q at t_start + duration."""
        q = float(np.clip(q, LIMITS[j, 0] + MARGIN, LIMITS[j, 1] - MARGIN))
        t_last = self.keys[j][-1][0]
        if t_start < t_last - 1e-9:
            raise ValueError(f"{LABELS[j]}: keyframe at {t_start:.2f}s precedes the last one at {t_last:.2f}s")
        if t_start > t_last + 1e-9:
            self.keys[j].append((t_start, self.keys[j][-1][1]))
        self.keys[j].append((t_start + duration, q))

    # finger-level helpers ---------------------------------------------------------------
    def finger_curl(self, finger: str, t_start: float, duration: float, amount: float):
        f = FINGER[finger]
        if finger == "thumb":
            self.move(f * 4 + 0, t_start, duration, THUMB_CURL["cmc_flex"] * amount)
            self.move(f * 4 + 2, t_start, duration, THUMB_CURL["mcp"] * amount)
            self.move(f * 4 + 3, t_start, duration, THUMB_CURL["ip"] * amount)
        else:
            self.move(f * 4 + 0, t_start, duration, CURL["mcp_flex"] * amount)
            self.move(f * 4 + 2, t_start, duration, CURL["pip"] * amount)
            self.move(f * 4 + 3, t_start, duration, CURL["dip"] * amount)

    def finger_spread(self, finger: str, t_start: float, duration: float, amount: float):
        f = FINGER[finger]
        self.move(f * 4 + 1, t_start, duration, SPREAD[finger] * amount)

    def thumb_pose(self, t_start: float, duration: float, cmc_flex=None, cmc_abd=None, mcp=None, ip=None):
        for j, v in zip(range(4), (cmc_flex, cmc_abd, mcp, ip)):
            if v is not None:
                self.move(j, t_start, duration, v)

    def hold_all(self, t: float):
        for j in range(20):
            if self.keys[j][-1][0] < t:
                self.keys[j].append((t, self.keys[j][-1][1]))

    # evaluation ----------------------------------------------------------------------------
    def sample(self, t: float):
        q = np.zeros(20); dq = np.zeros(20)
        for j in range(20):
            keys = self.keys[j]
            if t <= keys[0][0]:
                q[j] = keys[0][1]; continue
            if t >= keys[-1][0]:
                q[j] = keys[-1][1]; continue
            # find the segment
            lo, hi = 0, len(keys) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if keys[mid][0] <= t:
                    lo = mid
                else:
                    hi = mid
            t0, q0 = keys[lo]; t1, q1 = keys[hi]
            T = t1 - t0
            s = (t - t0) / T
            v, dv = smoothstep5(s)
            q[j] = q0 + (q1 - q0) * v
            dq[j] = (q1 - q0) * dv / T
        return q, dq


def build_showcase(q_start: np.ndarray, speed: float = 1.0) -> Timeline:
    """The full showcase. `speed` > 1 makes everything faster. Returns a Timeline from q_start."""
    tl = Timeline(q_start)
    k = 1.0 / speed
    fingers4 = ["index", "middle", "ring", "pinky"]

    # 0) go to the open home pose
    t = 0.0
    for j in range(20):
        tl.move(j, t, 2.0 * k, 0.0)
    t += 2.3 * k

    # 1) breathing: all four fingers gently curl and open twice (slow, tiny, perfectly smooth)
    for cyc in range(2):
        for f in fingers4:
            tl.finger_curl(f, t, 2.0 * k, 0.18)
        tl.finger_curl("thumb", t, 2.0 * k, 0.15)
        t += 2.0 * k
        for f in fingers4:
            tl.finger_curl(f, t, 2.0 * k, 0.0)
        tl.finger_curl("thumb", t, 2.0 * k, 0.0)
        t += 2.0 * k
    t += 0.3 * k

    # 2) ripple: curl index -> pinky one after another, then release pinky -> index; twice
    for rep in range(2):
        for i, f in enumerate(fingers4):
            tl.finger_curl(f, t + 0.30 * k * i, 0.9 * k, 0.85)
        t += 0.30 * k * 3 + 0.9 * k + 0.4 * k
        for i, f in enumerate(reversed(fingers4)):
            tl.finger_curl(f, t + 0.30 * k * i, 0.9 * k, 0.0)
        t += 0.30 * k * 3 + 0.9 * k + 0.3 * k

    # 3) fist (thumb stays outside the fingers), hold, open
    for f in fingers4:
        tl.finger_curl(f, t, 1.4 * k, 0.95)
    tl.thumb_pose(t, 1.4 * k, cmc_flex=0.25, cmc_abd=-0.15, mcp=0.55, ip=0.55)
    t += 1.4 * k + 1.0 * k
    for f in fingers4:
        tl.finger_curl(f, t, 1.4 * k, 0.0)
    tl.thumb_pose(t, 1.4 * k, cmc_flex=0.0, cmc_abd=0.0, mcp=0.0, ip=0.0)
    t += 1.4 * k + 0.4 * k

    # 4) spread the fingers wide, hold, close them together
    for f in fingers4:
        tl.finger_spread(f, t, 1.3 * k, 1.0)
    tl.thumb_pose(t, 1.3 * k, cmc_abd=-0.5)
    t += 1.3 * k + 0.8 * k
    for f in fingers4:
        tl.finger_spread(f, t, 1.3 * k, 0.0)
    tl.thumb_pose(t, 1.3 * k, cmc_abd=0.0)
    t += 1.3 * k + 0.4 * k

    # 5) piano: quick taps index, middle, ring, pinky, ring, middle, index
    for i, f in enumerate(["index", "middle", "ring", "pinky", "ring", "middle", "index"]):
        t0 = t + 0.38 * k * i
        fi = FINGER[f]
        tl.move(fi * 4 + 0, t0, 0.32 * k, 0.55)
        tl.move(fi * 4 + 2, t0, 0.32 * k, 0.55)
        tl.move(fi * 4 + 0, t0 + 0.32 * k, 0.32 * k, 0.0)
        tl.move(fi * 4 + 2, t0 + 0.32 * k, 0.32 * k, 0.0)
    t += 0.38 * k * 6 + 0.64 * k + 0.4 * k

    # 6) thumb: opposition sweep across the palm and back, then a thumb curl
    tl.thumb_pose(t, 1.6 * k, cmc_flex=0.35, cmc_abd=0.55)
    t += 1.6 * k + 0.5 * k
    tl.thumb_pose(t, 1.6 * k, cmc_flex=0.0, cmc_abd=0.0)
    t += 1.6 * k + 0.2 * k
    tl.finger_curl("thumb", t, 1.0 * k, 0.9)
    t += 1.0 * k + 0.4 * k
    tl.finger_curl("thumb", t, 1.0 * k, 0.0)
    t += 1.0 * k + 0.4 * k

    # 7) count to five: from a fist, open index, middle, ring, pinky, thumb
    for f in fingers4:
        tl.finger_curl(f, t, 1.2 * k, 0.95)
    tl.thumb_pose(t, 1.2 * k, cmc_flex=0.25, cmc_abd=-0.15, mcp=0.55, ip=0.55)
    t += 1.2 * k + 0.6 * k
    for i, f in enumerate(fingers4):
        tl.finger_curl(f, t + 0.7 * k * i, 0.6 * k, 0.0)
    t += 0.7 * k * 4
    tl.thumb_pose(t, 0.7 * k, cmc_flex=0.0, cmc_abd=0.0, mcp=0.0, ip=0.0)
    t += 0.7 * k + 0.6 * k

    # 8) goodbye wave: two fast ripples with small amplitude, then home
    for rep in range(2):
        for i, f in enumerate(fingers4):
            tl.finger_curl(f, t + 0.12 * k * i, 0.45 * k, 0.5)
        t += 0.12 * k * 3 + 0.45 * k
        for i, f in enumerate(fingers4):
            tl.finger_curl(f, t + 0.12 * k * i, 0.45 * k, 0.0)
        t += 0.12 * k * 3 + 0.45 * k
    t += 0.3 * k
    for j in range(20):
        tl.move(j, t, 1.0 * k, 0.0)
    tl.hold_all(t + 1.0 * k + 0.5 * k)
    return tl


if __name__ == "__main__":
    tl = build_showcase(np.zeros(20))
    print(f"showcase: {tl.t_end():.1f} s")
    ts = np.arange(0, tl.t_end(), 0.004)
    Q = np.array([tl.sample(t)[0] for t in ts]); DQ = np.array([tl.sample(t)[1] for t in ts])
    print("max |q| per joint:", np.round(np.abs(Q).max(0), 2).tolist())
    print("max |dq| rad/s:", round(float(np.abs(DQ).max()), 2), " max |ddq| rad/s^2:", round(float(np.abs(np.diff(DQ, axis=0) / 0.004).max()), 1))
