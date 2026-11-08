"""Execute a reference trajectory in the MuJoCo scene and score how it went."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.clips import PEN_SHAFT_AXIS_LOCAL, Episode
from ..data.transforms import quat_apply
from ..features.featurizer import FrameState
from ..graph.planner import ReferenceTrajectory
from .controller import Controller, LiveState, OnnxTrackingController, PDReferenceController, ReferenceFrame
from .scene import PenSpinScene


@dataclass
class Thresholds:
    """Termination thresholds, matching wuji-mjlab's pen-spin terminations."""

    pos_fail_m: float = 0.05
    axis_fail_rad: float = np.deg2rad(60.0)


@dataclass
class RolloutResult:
    n_steps: int
    terminated: bool
    termination_step: int  # -1 if not terminated
    termination_reason: str
    pen_pos_err: np.ndarray  # (n_steps,) m, vs reference after each tick
    pen_axis_err: np.ndarray  # (n_steps,) rad
    joint_err: np.ndarray  # (n_steps,) max |q - q_ref| rad
    states: list[FrameState] = field(default_factory=list)
    targets: np.ndarray | None = None  # (n_steps, 20) commanded joint targets

    @property
    def completed(self) -> bool:
        return not self.terminated

    def mean_pos_err(self, last: int | None = None) -> float:
        x = self.pen_pos_err if last is None else self.pen_pos_err[-last:]
        return float(np.mean(x)) if x.size else float("nan")

    def mean_axis_err(self, last: int | None = None) -> float:
        x = self.pen_axis_err if last is None else self.pen_axis_err[-last:]
        return float(np.mean(x)) if x.size else float("nan")

    def summary(self) -> str:
        status = "ok" if self.completed else f"FAILED@{self.termination_step} ({self.termination_reason})"
        return (
            f"{status}: {self.n_steps} ticks, pen pos err mean {100 * self.mean_pos_err():.1f} cm "
            f"(final {100 * self.mean_pos_err(5):.1f}), axis err mean {np.rad2deg(self.mean_axis_err()):.0f} deg, "
            f"joint err mean {np.mean(self.joint_err):.3f} rad"
        )


def frame_state_from_episode(ep: Episode, t: int) -> FrameState:
    return FrameState(
        qpos=ep.qpos[t], qvel=ep.qvel[t], pen_pose=ep.pen_pose[t], pen_linvel=ep.pen_linvel[t],
        pen_angvel=ep.pen_angvel[t], link_pos=ep.link_pos[t],
    )


def live_state(scene: PenSpinScene) -> FrameState:
    lin, ang = scene.pen_velocity_local()
    return FrameState(
        qpos=scene.hand_qpos(), qvel=scene.hand_qvel(), pen_pose=scene.pen_pose_local(),
        pen_linvel=lin, pen_angvel=ang, link_pos=scene.link_pos_local(),
    )


def _axis_error(pen_pose: np.ndarray, ref_pose: np.ndarray) -> float:
    a = quat_apply(pen_pose[3:], PEN_SHAFT_AXIS_LOCAL)
    b = quat_apply(ref_pose[3:], PEN_SHAFT_AXIS_LOCAL)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def make_controller(name: str, **kwargs) -> Controller:
    if name == "pd":
        return PDReferenceController()
    if name == "onnx":
        return OnnxTrackingController(**kwargs)
    raise ValueError(f"unknown controller {name!r} (use 'pd' or 'onnx')")


def controller_lookahead(controller: Controller) -> int:
    """PD tracks the *next* frame (no lag); the Wuji policy was trained on the current frame."""
    return 1 if isinstance(controller, PDReferenceController) else 0


def rollout(
    scene: PenSpinScene,
    controller: Controller,
    ref: ReferenceTrajectory,
    start: FrameState,
    *,
    thresholds: Thresholds | None = None,
    record_states: bool = False,
    max_steps: int | None = None,
) -> RolloutResult:
    """Start at ``start`` (which should equal ``ref`` frame 0) and track the
    reference tick by tick.  Errors are measured after each tick against the
    reference frame the hand should have reached."""
    th = thresholds or Thresholds()
    T = len(ref)
    n = T - 1 if max_steps is None else min(T - 1, max_steps)
    scene.set_state(start.qpos, start.qvel, start.pen_pose, start.pen_linvel, start.pen_angvel)
    controller.reset()
    look = controller_lookahead(controller)

    pos_err = np.zeros(n)
    axis_err = np.zeros(n)
    joint_err = np.zeros(n)
    targets = np.zeros((n, start.qpos.shape[0]))
    states: list[FrameState] = []
    terminated, term_step, reason = False, -1, ""
    for k in range(n):
        live = LiveState(scene.hand_qpos(), scene.hand_qvel(), scene.pen_pose_local(), scene.link_pos_local())
        j = min(k + look, T - 1)
        frame = ReferenceFrame(
            qpos=ref.qpos[j], pen_pose=ref.pen_pose[j],
            link_pos=None if ref.link_pos is None else ref.link_pos[j],
            group_id=None if ref.group_ids is None else ref.group_ids[j],
        )
        target = controller.act(frame, live)
        targets[k] = target
        scene.control_step(target)
        if record_states:
            states.append(live_state(scene))
        ref_next = min(k + 1, T - 1)
        pen = scene.pen_pose_local()
        pos_err[k] = np.linalg.norm(pen[:3] - ref.pen_pose[ref_next, :3])
        axis_err[k] = _axis_error(pen, ref.pen_pose[ref_next])
        joint_err[k] = np.abs(scene.hand_qpos() - ref.qpos[ref_next]).max()
        if not np.isfinite(pen).all():
            terminated, term_step, reason = True, k, "numerical"
        elif pos_err[k] > th.pos_fail_m:
            terminated, term_step, reason = True, k, "pen_position"
        elif axis_err[k] > th.axis_fail_rad:
            terminated, term_step, reason = True, k, "pen_axis"
        if terminated:
            pos_err, axis_err, joint_err, targets = pos_err[: k + 1], axis_err[: k + 1], joint_err[: k + 1], targets[: k + 1]
            break
    return RolloutResult(
        n_steps=int(pos_err.size), terminated=terminated, termination_step=term_step, termination_reason=reason,
        pen_pos_err=pos_err, pen_axis_err=axis_err, joint_err=joint_err, states=states, targets=targets,
    )


def episode_reference(ep: Episode, start: int, n_frames: int) -> ReferenceTrajectory:
    stop = min(len(ep), start + n_frames)
    return ReferenceTrajectory(
        qpos=ep.qpos[start:stop], pen_pose=ep.pen_pose[start:stop],
        nodes=np.arange(start, stop), frame_dt=ep.frame_dt,
        link_pos=ep.link_pos[start:stop], group_ids=[ep.group_id] * (stop - start),
    )


def track_episode(
    scene: PenSpinScene, controller: Controller, ep: Episode, *, start: int = 0, n_frames: int | None = None, **kw
) -> RolloutResult:
    """Baseline: replay a recorded clip from frame ``start``."""
    n = len(ep) - start if n_frames is None else n_frames
    ref = episode_reference(ep, start, n)
    return rollout(scene, controller, ref, frame_state_from_episode(ep, start), **kw)
