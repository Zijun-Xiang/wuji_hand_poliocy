"""Low-level controllers that turn a reference frame into joint targets.

Two options:

* :class:`PDReferenceController` – send the reference joint angles straight to
  the hand's position actuators (what an open-loop replay on the real hand
  would do).
* :class:`OnnxTrackingController` – Wuji's released PPO tracking policies
  (``policy.onnx`` per motion group).  The policy observes the current
  reference frame, the live joint state and the live pen pose and outputs a
  residual on the reference joint angles, exactly as in
  ``wuji-mjlab/deploy/pen_spin`` (155-D observation, residual x0.5, clip +-0.6,
  EMA 0.5).  Because the policies were trained to *correct* imperfect
  retargeted references, they are the realistic yardstick for "can the hand
  actually execute this transition".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.clips import FRAME_DT, NUM_JOINTS
from ..data.transforms import quat_apply

POLICY_DIR = Path(__file__).resolve().parents[3] / "assets" / "wuji_policies"
OBS_DIM = 155
AXIS_POINT_RADIUS_M = 0.10
POSITION_TANH_SCALE_M = 0.15


@dataclass
class ReferenceFrame:
    """Everything a controller may want to know about the frame being tracked."""

    qpos: np.ndarray  # (20,)
    pen_pose: np.ndarray  # (7,) wrist-local
    link_pos: np.ndarray | None = None  # (21, 3) wrist-local
    group_id: str | None = None  # motion group the frame comes from (policy selection)


@dataclass
class LiveState:
    qpos: np.ndarray
    qvel: np.ndarray
    pen_pose: np.ndarray  # wrist-local
    link_pos: np.ndarray | None = None


def bounded_axis_points(pen_pose: np.ndarray, radius: float = AXIS_POINT_RADIUS_M, scale: float = POSITION_TANH_SCALE_M) -> np.ndarray:
    """The two tanh-bounded shaft end points (6-D) used by the Wuji policy."""
    pen_pose = np.asarray(pen_pose, dtype=np.float64)
    pts = np.array([[radius, 0, 0], [-radius, 0, 0]])
    world = pen_pose[:3] + quat_apply(np.broadcast_to(pen_pose[3:], (2, 4)), pts)
    return (scale * np.tanh(world / scale)).reshape(-1)


class Controller:
    def reset(self) -> None:  # pragma: no cover - interface
        pass

    def act(self, ref: ReferenceFrame, live: LiveState) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class PDReferenceController(Controller):
    """Target = reference joint angles (the actuators' own PD does the rest)."""

    name = "pd"

    def act(self, ref: ReferenceFrame, live: LiveState) -> np.ndarray:
        return np.asarray(ref.qpos, dtype=np.float64)


@dataclass
class OnnxTrackingController(Controller):
    """Wuji's residual tracking policy, one ONNX file per motion group."""

    policy_dir: Path = POLICY_DIR
    default_group: str = "03_continuous_multi_axis"
    action_scale: float = 0.5
    residual_clip: tuple[float, float] = (-0.6, 0.6)
    ema_alpha: float = 0.5
    name: str = "onnx"
    _sessions: dict[str, object] = field(default_factory=dict, repr=False)
    _last_raw: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, np.float32), repr=False)
    _prev_residual: np.ndarray = field(default_factory=lambda: np.zeros(NUM_JOINTS, np.float32), repr=False)

    def __post_init__(self) -> None:
        cfg_path = Path(self.policy_dir) / "pen_spin_policy.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text())
            self.action_scale = float(cfg.get("action_scale", self.action_scale))
            lo, hi = cfg.get("action_residual_clip", self.residual_clip)
            self.residual_clip = (float(lo), float(hi))
            self.ema_alpha = float(cfg.get("action_ema_alpha", self.ema_alpha))

    # ------------------------------------------------------------------ #
    def available_groups(self) -> list[str]:
        return sorted(p.parent.name for p in Path(self.policy_dir).glob("*/policy.onnx"))

    def _session(self, group: str | None):
        import onnxruntime as ort

        groups = self.available_groups()
        if not groups:
            raise FileNotFoundError(f"no policy.onnx found under {self.policy_dir}")
        key = group if group in groups else (self.default_group if self.default_group in groups else groups[0])
        if key not in self._sessions:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._sessions[key] = ort.InferenceSession(
                str(Path(self.policy_dir) / key / "policy.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def reset(self) -> None:
        self._last_raw = np.zeros(NUM_JOINTS, np.float32)
        self._prev_residual = np.zeros(NUM_JOINTS, np.float32)

    def observation(self, ref: ReferenceFrame, live: LiveState) -> np.ndarray:
        if ref.link_pos is None:
            raise ValueError("OnnxTrackingController needs the reference link positions")
        command = np.concatenate(
            [np.asarray(ref.qpos, np.float32), bounded_axis_points(ref.pen_pose).astype(np.float32), np.asarray(ref.link_pos, np.float32).reshape(-1)]
        )
        obs = np.concatenate(
            [
                command,
                np.asarray(live.qpos, np.float32),
                np.asarray(live.qvel, np.float32) * 0.25,
                bounded_axis_points(live.pen_pose).astype(np.float32),
                self._last_raw,
            ]
        ).astype(np.float32)
        if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
            raise RuntimeError("bad pen-spin observation")
        return obs

    def act(self, ref: ReferenceFrame, live: LiveState) -> np.ndarray:
        sess = self._session(ref.group_id)
        obs = self.observation(ref, live)
        raw = sess.run(["actions"], {"obs": obs.reshape(1, -1)})[0].reshape(NUM_JOINTS).astype(np.float32)
        self._last_raw = raw
        residual = np.clip(raw * self.action_scale, *self.residual_clip).astype(np.float32)
        processed = self.ema_alpha * residual + (1.0 - self.ema_alpha) * self._prev_residual
        self._prev_residual = processed
        return np.asarray(ref.qpos, np.float64) + processed
