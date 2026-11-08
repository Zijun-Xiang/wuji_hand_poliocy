"""Physics-free feasibility checks for jumps: can the *real* hand do this?

These are cheap filters applied before the MuJoCo validator:

1. **Joint limits** – every blended frame between ``u`` and ``v`` must respect
   the Wuji Hand 2 joint ranges (taken from the MJCF).
2. **Joint velocity** – the fastest joint during the blend must stay below a
   velocity limit.  By default the limit is derived from the dataset itself
   (the hand demonstrably moved that fast while spinning), scaled by a margin.
3. **Pen continuity** – the pen cannot teleport.  The pen pose at ``u+1`` (where
   the data says the pen *will be*) and at ``v`` (where we pretend it is) must
   agree within a position and a shaft-direction tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.clips import PEN_SHAFT_AXIS_LOCAL
from ..data.transforms import quat_apply
from .motion_graph import JumpStatus, MotionGraph

# Joint ranges of wujihand2_2-right (rad), finger1..5 x joint1..4.
HAND2_JOINT_LOWER = np.array(
    [-1.187, -1.484, -1.047, -1.047] + [-1.047, -0.698, -1.047, -1.047] * 4
)
HAND2_JOINT_UPPER = np.array(
    [1.291, 0.698, 1.57, 1.57] + [1.57, 0.698, 2.094, 1.57] * 4
)


@dataclass
class KinematicLimits:
    joint_lower: np.ndarray = field(default_factory=lambda: HAND2_JOINT_LOWER.copy())
    joint_upper: np.ndarray = field(default_factory=lambda: HAND2_JOINT_UPPER.copy())
    max_joint_vel: float = 6.0  # rad/s, see ``from_dataset``
    max_pen_pos_jump_m: float = 0.02  # 2 cm, for a minimal blend
    max_pen_axis_jump_rad: float = np.deg2rad(20.0)
    blend_frames: int = 2  # minimum intermediate frames inserted between u and v
    max_blend_frames: int = 10  # longest blend allowed (0.2 s); longer jumps are rejected
    pen_tolerance_growth: float = 0.5  # pen tolerances grow by this fraction per extra blend frame
    max_pen_pos_jump_cap_m: float = 0.06
    max_pen_axis_jump_cap_rad: float = np.deg2rad(45.0)

    @classmethod
    def from_dataset(cls, graph: MotionGraph, *, percentile: float = 99.9, margin: float = 1.5, **kw) -> "KinematicLimits":
        qvel = np.concatenate([ep.qvel for ep in graph.episodes])
        vmax = float(np.percentile(np.abs(qvel), percentile) * margin)
        return cls(max_joint_vel=vmax, **kw)


@dataclass
class KinematicReport:
    ok: np.ndarray  # bool per jump
    max_joint_vel: np.ndarray  # with the minimum blend
    pen_pos_jump: np.ndarray
    pen_axis_jump: np.ndarray
    joint_limit_ok: np.ndarray
    blend: np.ndarray  # blend frames chosen per jump (so that joint speed <= limit)


def check_jumps(graph: MotionGraph, limits: KinematicLimits, *, indices: np.ndarray | None = None) -> KinematicReport:
    """Evaluate the kinematic filters for the given jump indices (default: all).

    The blend length is chosen per jump: the smallest number of intermediate
    frames (>= ``limits.blend_frames``) that keeps every joint below the speed
    limit.  Jumps needing more than ``limits.max_blend_frames`` fail.  Pen
    tolerances grow with the blend length (the tracking policy gets more time
    to move the pen) up to a cap."""
    J = graph.jumps
    idx = np.arange(len(J)) if indices is None else np.asarray(indices)
    src, dst = J.src[idx], J.dst[idx]
    dt = graph.episodes[0].frame_dt
    n = src.size
    if n == 0:
        z = np.zeros(0)
        return KinematicReport(z.astype(bool), z, z, z, z.astype(bool), z.astype(np.int16))

    q_u = np.stack([graph.node_qpos(k) for k in src])
    q_v = np.stack([graph.node_qpos(k) for k in dst])
    dq = np.abs(q_v - q_u).max(axis=1)
    max_vel_min_blend = dq / ((limits.blend_frames + 1) * dt)
    # frames needed so that dq / ((b+1) dt) <= vmax  ->  b >= dq/(vmax dt) - 1
    needed = np.ceil(dq / (limits.max_joint_vel * dt) - 1.0).astype(int)
    blend = np.clip(np.maximum(needed, limits.blend_frames), limits.blend_frames, limits.max_blend_frames)
    speed_ok = dq / ((blend + 1) * dt) <= limits.max_joint_vel + 1e-9

    def in_range(q):
        return np.all((q >= limits.joint_lower - 1e-6) & (q <= limits.joint_upper + 1e-6), axis=1)

    joint_ok = in_range(q_u) & in_range(q_v)

    p_next = np.stack([graph.node_pen_pose(k + 1) for k in src])
    p_v = np.stack([graph.node_pen_pose(k) for k in dst])
    pos_jump = np.linalg.norm(p_next[:, :3] - p_v[:, :3], axis=1)
    ax_next = quat_apply(p_next[:, 3:], np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (n, 3)))
    ax_v = quat_apply(p_v[:, 3:], np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (n, 3)))
    axis_jump = np.arccos(np.clip(np.sum(ax_next * ax_v, axis=1), -1.0, 1.0))

    extra = np.maximum(blend - limits.blend_frames, 0)
    pos_tol = np.minimum(limits.max_pen_pos_jump_m * (1.0 + limits.pen_tolerance_growth * extra), limits.max_pen_pos_jump_cap_m)
    axis_tol = np.minimum(limits.max_pen_axis_jump_rad * (1.0 + limits.pen_tolerance_growth * extra), limits.max_pen_axis_jump_cap_rad)

    ok = joint_ok & speed_ok & (pos_jump <= pos_tol) & (axis_jump <= axis_tol)
    return KinematicReport(ok=ok, max_joint_vel=max_vel_min_blend, pen_pos_jump=pos_jump, pen_axis_jump=axis_jump, joint_limit_ok=joint_ok, blend=blend.astype(np.int16))


def apply_kinematic_filter(graph: MotionGraph, limits: KinematicLimits) -> KinematicReport:
    """Mark candidate jumps that fail the kinematic checks as ``KINEMATIC_FAIL``."""
    idx = graph.jumps.with_status(JumpStatus.CANDIDATE)
    report = check_jumps(graph, limits, indices=idx)
    failed = idx[~report.ok]
    graph.jumps.status[failed] = int(JumpStatus.KINEMATIC_FAIL)
    graph.jumps.blend[idx] = report.blend
    return report
