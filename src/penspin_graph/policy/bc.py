"""Distil "motion graph + tracking policy" into one reactive policy (behaviour cloning).

Pipeline
--------
1. Sample random walks on the *validated* graph and execute them in MuJoCo with
   the low-level tracking controller (Wuji's ONNX policy by default).
2. Record ``(observation, commanded joint target)`` pairs along the way.  The
   observation is what the real robot can measure: joint angles, joint
   velocities and the pen's two shaft end points (camera) - no reference
   signal.
3. Fit an MLP that maps observation -> next joint target (as a residual on the
   current joint angles) with mean-squared error.
4. Evaluate closed loop: start from random dataset frames and run the BC
   policy alone; report how long the pen survives and how much it rotates.

This is the cheap, CPU-friendly way to get a reference-free policy.  For the
strong version, export stitched clips (:mod:`export_clips`) and run
wuji-mjlab's PPO on a GPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.clips import PEN_SHAFT_AXIS_LOCAL
from ..data.transforms import quat_angle, quat_apply
from ..features.featurizer import FrameState
from ..graph.motion_graph import JumpStatus, MotionGraph
from ..graph.planner import path_to_trajectory
from ..sim.controller import Controller, LiveState, bounded_axis_points
from ..sim.rollout import Thresholds, frame_state_from_episode, make_controller, rollout
from ..sim.scene import PenSpinScene
from .export_clips import random_walk

OBS_DIM = 20 + 20 + 6


def bc_observation(qpos: np.ndarray, qvel: np.ndarray, pen_pose_local: np.ndarray) -> np.ndarray:
    return np.concatenate([qpos, 0.25 * qvel, bounded_axis_points(pen_pose_local)]).astype(np.float32)


@dataclass
class BCDataset:
    obs: np.ndarray  # (N, 46)
    target_residual: np.ndarray  # (N, 20) commanded target - current qpos
    walks: int
    ticks: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, obs=self.obs, target_residual=self.target_residual, walks=self.walks, ticks=self.ticks)

    @classmethod
    def load(cls, path: Path) -> "BCDataset":
        with np.load(path) as z:
            return cls(z["obs"], z["target_residual"], int(z["walks"]), int(z["ticks"]))


def collect_bc_dataset(
    graph: MotionGraph,
    *,
    n_walks: int = 100,
    walk_frames: int = 150,
    blend_frames: int | None = None,
    controller: str = "onnx",
    seed: int = 0,
    scene: PenSpinScene | None = None,
    verbose: bool = True,
    statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,),
) -> BCDataset:
    rng = np.random.default_rng(seed)
    scene = scene or PenSpinScene()
    ctrl = make_controller(controller)
    obs_all, res_all = [], []
    walks, t0 = 0, time.time()
    attempts = 0
    while walks < n_walks and attempts < 10 * n_walks:
        attempts += 1
        path = random_walk(graph, rng, min_frames=walk_frames // 2, max_frames=walk_frames, statuses=statuses)
        if not path:
            continue
        traj = path_to_trajectory(graph, path, blend_frames=blend_frames)
        e, t = graph.node_episode_frame(path[0])
        start = frame_state_from_episode(graph.episodes[e], t)
        res = rollout(scene, ctrl, traj, start, record_states=True)
        # states[k] is the state *after* tick k; the observation for tick k is the state before it
        prev = start
        for k in range(res.n_steps):
            obs_all.append(bc_observation(prev.qpos, prev.qvel, prev.pen_pose))
            res_all.append(res.targets[k] - prev.qpos)
            prev = res.states[k]
        walks += 1
        if verbose and walks % 10 == 0:
            print(f"  collected {walks}/{n_walks} walks, {len(obs_all)} samples ({time.time() - t0:.0f}s)", end="\r")
    if verbose:
        print()
    return BCDataset(np.asarray(obs_all, np.float32), np.asarray(res_all, np.float32), walks, len(obs_all))


# ---------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------- #
def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("behaviour cloning needs PyTorch: pip install torch") from exc
    return torch


def build_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...] = (512, 512, 256)):
    torch = _torch()
    layers, d = [], in_dim
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ELU()]
        d = h
    layers.append(torch.nn.Linear(d, out_dim))
    return torch.nn.Sequential(*layers)


@dataclass
class BCPolicy(Controller):
    """Reference-free policy: joint target = qpos + MLP(obs)."""

    model: object
    obs_mean: np.ndarray
    obs_std: np.ndarray
    res_scale: float
    name: str = "bc"

    def reset(self) -> None:
        pass

    def predict(self, qpos: np.ndarray, qvel: np.ndarray, pen_pose_local: np.ndarray) -> np.ndarray:
        torch = _torch()
        obs = (bc_observation(qpos, qvel, pen_pose_local) - self.obs_mean) / self.obs_std
        with torch.no_grad():
            out = self.model(torch.as_tensor(obs[None], dtype=torch.float32)).numpy()[0]
        return np.asarray(qpos, np.float64) + out * self.res_scale

    def act(self, ref, live: LiveState) -> np.ndarray:  # ``ref`` is ignored: no reference needed
        return self.predict(live.qpos, live.qvel, live.pen_pose)

    def save(self, directory: Path) -> None:
        torch = _torch()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "bc_policy.pt")
        meta = {
            "obs_mean": self.obs_mean.tolist(), "obs_std": self.obs_std.tolist(), "res_scale": self.res_scale,
            "hidden": [m.out_features for m in self.model if isinstance(m, torch.nn.Linear)][:-1],
            "obs_dim": OBS_DIM,
        }
        (directory / "bc_policy.json").write_text(json.dumps(meta))
        try:
            dummy = torch.zeros(1, OBS_DIM)
            torch.onnx.export(self.model, dummy, str(directory / "bc_policy.onnx"), input_names=["obs"], output_names=["residual"], dynamo=False)
        except Exception as exc:  # pragma: no cover - onnx export is best effort
            (directory / "ONNX_EXPORT_FAILED.txt").write_text(str(exc))

    @classmethod
    def load(cls, directory: Path) -> "BCPolicy":
        torch = _torch()
        directory = Path(directory)
        meta = json.loads((directory / "bc_policy.json").read_text())
        model = build_mlp(OBS_DIM, 20, tuple(meta["hidden"]))
        model.load_state_dict(torch.load(directory / "bc_policy.pt", map_location="cpu"))
        model.eval()
        return cls(model, np.asarray(meta["obs_mean"], np.float32), np.asarray(meta["obs_std"], np.float32), float(meta["res_scale"]))


def train_bc(
    data: BCDataset,
    *,
    hidden: tuple[int, ...] = (512, 512, 256),
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    verbose: bool = True,
) -> BCPolicy:
    torch = _torch()
    torch.manual_seed(seed)
    obs_mean = data.obs.mean(0)
    obs_std = np.maximum(data.obs.std(0), 1e-3).astype(np.float32)
    res_scale = float(np.maximum(np.abs(data.target_residual).max(), 1e-3))
    X = torch.as_tensor((data.obs - obs_mean) / obs_std, dtype=torch.float32)
    Y = torch.as_tensor(data.target_residual / res_scale, dtype=torch.float32)
    n = X.shape[0]
    perm = torch.randperm(n)
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = build_mlp(OBS_DIM, 20, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        order = tr_idx[torch.randperm(tr_idx.numel())]
        total = 0.0
        for s in range(0, order.numel(), batch_size):
            b = order[s : s + batch_size]
            loss = torch.nn.functional.mse_loss(model(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b.numel()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = torch.nn.functional.mse_loss(model(X[val_idx]), Y[val_idx]).item()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1}/{epochs}: train mse {total / tr_idx.numel():.5f}  val mse {val:.5f}")
    model.eval()
    return BCPolicy(model, obs_mean.astype(np.float32), obs_std, res_scale)


# ---------------------------------------------------------------------- #
# evaluation
# ---------------------------------------------------------------------- #
@dataclass
class ClosedLoopEpisode:
    start_label: str
    ticks_survived: int
    pen_rotation_rad: float
    dropped: bool
    jumps_taken: int = 0
    episodes_visited: int = 1


@dataclass
class ClosedLoopReport:
    episodes: list[ClosedLoopEpisode] = field(default_factory=list)

    def describe(self) -> str:
        if not self.episodes:
            return "no episodes"
        surv = np.array([e.ticks_survived for e in self.episodes])
        rot = np.array([e.pen_rotation_rad for e in self.episodes])
        dropped = np.mean([e.dropped for e in self.episodes])
        jumps = np.array([e.jumps_taken for e in self.episodes])
        visited = np.array([e.episodes_visited for e in self.episodes])
        return (
            f"{len(self.episodes)} starts: survived mean {surv.mean() * 0.02:.2f}s (min {surv.min() * 0.02:.2f}s), "
            f"pen rotation mean {np.rad2deg(rot.mean()):.0f} deg, drop rate {100 * dropped:.0f}%, "
            f"jumps taken mean {jumps.mean():.1f}, demos visited mean {visited.mean():.1f}"
        )


def evaluate_closed_loop(
    graph: MotionGraph,
    policy: Controller,
    *,
    n_starts: int = 20,
    horizon: int = 250,
    seed: int = 0,
    scene: PenSpinScene | None = None,
    drop_height_m: float = 0.12,
    verbose: bool = True,
) -> ClosedLoopReport:
    """Run a reference-free policy from random recorded states.

    Survival = the pen stays within ``drop_height_m`` below its start height and
    the simulation stays finite.  Rotation is the accumulated pen quaternion
    angle, a proxy for "it keeps spinning"."""
    rng = np.random.default_rng(seed)
    scene = scene or PenSpinScene()
    report = ClosedLoopReport()
    for i in range(n_starts):
        node = int(rng.integers(graph.num_nodes))
        e, t = graph.node_episode_frame(node)
        start = frame_state_from_episode(graph.episodes[e], t)
        scene.set_state(start.qpos, start.qvel, start.pen_pose, start.pen_linvel, start.pen_angvel)
        policy.reset()
        z0 = scene.pen_pose_world()[2]
        prev_q = scene.pen_pose_local()[3:]
        rot, survived, dropped = 0.0, 0, False
        for k in range(horizon):
            live = LiveState(scene.hand_qpos(), scene.hand_qvel(), scene.pen_pose_local(), scene.link_pos_local())
            target = policy.act(None, live)
            scene.control_step(target)
            pen_w = scene.pen_pose_world()
            if not np.isfinite(pen_w).all() or pen_w[2] < z0 - drop_height_m:
                dropped = True
                break
            q = scene.pen_pose_local()[3:]
            rot += float(quat_angle(prev_q[None], q[None])[0])
            prev_q = q
            survived = k + 1
        matcher = getattr(policy, "matcher", None)
        n_jumps = int(getattr(matcher, "jumps_taken", 0)) if matcher is not None else 0
        visited = len(set(graph.episode_index[[h for h in matcher.history if h >= 0]].tolist())) if matcher is not None and matcher.history else 1
        report.episodes.append(ClosedLoopEpisode(graph.node_label(node), survived, rot, dropped, n_jumps, visited))
        if verbose:
            print(f"  start {i + 1}/{n_starts} {graph.node_label(node)}: survived {survived * 0.02:.2f}s, rotation {np.rad2deg(rot):.0f} deg, jumps {n_jumps}, demos {visited}{' DROPPED' if dropped else ''}")
    return report
