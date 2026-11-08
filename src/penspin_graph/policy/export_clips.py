"""Turn stitched graph walks / loops into a **wuji-mjlab compatible clip group**.

The output directory follows the ``wuji_pen_spin_clips:1`` catalog schema
(``index.json`` + ``clip_XXX.npz`` with SHA-256 digests) plus a recipe JSON, so
it can be dropped into ``wuji-mjlab/src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/``
and used directly as the reference set for PPO training::

    pixi run train --task WujiHand2_PenSpin \\
        env.commands.motion.recipe_path=<task>/data/recipe/06_graph_stitched.json ...

All world-frame arrays are expressed with the simulator's fixed wrist pose, so
the stitched clips are consistent with each other even though their sources
were captured with different wrist poses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.clips import FRAME_DT
from ..data.transforms import points_to_local, points_to_world, pose_to_world
from ..graph.loops import Loop
from ..graph.motion_graph import JumpStatus, MotionGraph
from ..graph.planner import ReferenceTrajectory, loop_to_trajectory, path_to_trajectory
from ..sim.scene import FIXED_WRIST_POSE_W

CLIP_SCHEMA = "wuji_pen_spin_clips:1"
RECIPE_SCHEMA = "wuji_pen_spin_recipe:1"


def _landmarks_local(graph: MotionGraph, node: int) -> np.ndarray | None:
    e, t = graph.node_episode_frame(node)
    ep = graph.episodes[e]
    lm = ep.extras.get("mediapipe_landmarks_w")
    if lm is None:
        return None
    return points_to_local(lm[t][None], ep.wrist_pose_w[t][None])[0]


def trajectory_to_clip_arrays(graph: MotionGraph, traj: ReferenceTrajectory, *, wrist_pose_w: np.ndarray = FIXED_WRIST_POSE_W) -> dict[str, np.ndarray]:
    """Reference trajectory -> arrays in the wuji-mjlab NPZ schema."""
    T = len(traj)
    if traj.link_pos is None:
        raise ValueError("trajectory needs link positions (build it with path_to_trajectory)")
    wrist = np.broadcast_to(wrist_pose_w, (T, 7)).astype(np.float32)
    object_pose_w = pose_to_world(traj.pen_pose, wrist_pose_w).astype(np.float32)
    link_pos_w = points_to_world(traj.link_pos, np.broadcast_to(wrist_pose_w, (T, 7))).astype(np.float32)

    # MediaPipe landmarks are not used by training but the deploy loader
    # requires the key; carry the source landmarks (interpolated at blends).
    lm_local = np.zeros((T, 21, 3))
    have = np.zeros(T, bool)
    for i, node in enumerate(traj.nodes):
        if node >= 0:
            lm = _landmarks_local(graph, int(node))
            if lm is not None:
                lm_local[i] = lm
                have[i] = True
    if have.any():
        idx = np.flatnonzero(have)
        for d in range(3):
            for k in range(21):
                lm_local[:, k, d] = np.interp(np.arange(T), idx, lm_local[idx, k, d])
    landmarks_w = points_to_world(lm_local, np.broadcast_to(wrist_pose_w, (T, 7))).astype(np.float32)
    return {
        "reference_qpos": traj.qpos.astype(np.float32),
        "wrist_pose_w": wrist,
        "object_pose_w": object_pose_w,
        "link_pos_w": link_pos_w,
        "mediapipe_landmarks_w": landmarks_w,
    }


def random_walk(
    graph: MotionGraph,
    rng: np.random.Generator,
    *,
    min_frames: int,
    max_frames: int,
    jump_prob: float = 0.08,
    min_dwell_frames: int = 25,
    statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,),
    start: int | None = None,
) -> list[int]:
    """Walk the stitched graph.

    Follow recorded frames; after at least ``min_dwell_frames`` on one episode,
    take a usable jump with probability ``jump_prob`` per tick, and *always*
    try to jump before the episode's trailing hold starts.  Destinations are
    sampled with probability proportional to their remaining active frames, so
    the walk prefers continuations with a long future.  Stops at ``max_frames``
    or when the motion ends with no way out (holds are not recorded)."""
    J = graph.jumps
    mask = np.isin(J.status, [int(s) for s in statuses])
    out: dict[int, list[int]] = {}
    for s_, d_ in zip(J.src[mask], J.dst[mask]):
        out.setdefault(int(s_), []).append(int(d_))
    active = graph.active_remaining(np.arange(graph.num_nodes))
    if start is None:
        sources = np.fromiter(out.keys(), dtype=np.int64) if out else np.arange(graph.num_nodes)
        sources = sources[active[sources] > 0] if (active[sources] > 0).any() else sources
        node = int(rng.choice(sources))
    else:
        node = int(start)
    path = [node]
    dwell = 0
    while len(path) < max_frames:
        options = [d for d in out.get(node, []) if active[d] > 0]
        remaining = int(active[node])
        nxt = graph.successor(node)
        want_jump = options and dwell >= min_dwell_frames and (remaining <= 1 or rng.random() < jump_prob)
        if options and (nxt is None or remaining <= 1) and dwell < min_dwell_frames:
            want_jump = True  # better to jump early than to stop
        if want_jump:
            weights = active[options].astype(float) + 1.0
            node = int(rng.choice(options, p=weights / weights.sum()))
            dwell = 0
        elif nxt is not None and remaining > 0:
            node = nxt
            dwell += 1
        else:
            break
        path.append(node)
    if len(path) < min_frames:
        return []
    return path


@dataclass
class ExportReport:
    group_dir: Path
    recipe_path: Path
    clip_frames: list[int]
    clip_jumps: list[int]

    def describe(self) -> str:
        total = sum(self.clip_frames)
        return (
            f"wrote {len(self.clip_frames)} clips ({total} frames, {total * FRAME_DT:.1f} s) to {self.group_dir}\n"
            f"  jumps per clip: {self.clip_jumps}\n  recipe: {self.recipe_path}"
        )


def export_clip_group(
    graph: MotionGraph,
    trajectories: list[ReferenceTrajectory],
    *,
    out_root: Path,
    group_id: str = "06_graph_stitched",
    group_name: str = "graph-stitched pen spinning",
    recipe_dir: Path | None = None,
) -> ExportReport:
    group_dir = Path(out_root) / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    frames, jumps = [], []
    for i, traj in enumerate(trajectories, start=1):
        arrays = trajectory_to_clip_arrays(graph, traj)
        clip_id = f"clip_{i:03d}"
        path = group_dir / f"{clip_id}.npz"
        np.savez(path, **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"clip_id": clip_id, "file": path.name, "sha256": digest})
        frames.append(len(traj))
        # count jumps = transitions between consecutive real nodes that are not t -> t+1 of one episode
        real = traj.nodes[traj.nodes >= 0]
        n_jumps = int(np.sum(~((real[1:] == real[:-1] + 1) & (graph.episode_index[real[1:]] == graph.episode_index[real[:-1]]))))
        jumps.append(n_jumps)
    index = {
        "schema": CLIP_SCHEMA,
        "group_id": group_id,
        "group_name": group_name,
        "frame_dt": FRAME_DT,
        "clip_count": len(entries),
        "clips": entries,
    }
    (group_dir / "index.json").write_text(json.dumps(index, indent=2))
    recipe_dir = Path(recipe_dir) if recipe_dir is not None else Path(out_root).parent / "recipe"
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{group_id}.json"
    recipe_path.write_text(json.dumps({"schema": RECIPE_SCHEMA, "groups": [group_id]}, indent=2))
    return ExportReport(group_dir=group_dir, recipe_path=recipe_path, clip_frames=frames, clip_jumps=jumps)


def build_export_trajectories(
    graph: MotionGraph,
    *,
    loops: list[Loop] | None = None,
    n_walks: int = 20,
    target_seconds: float = 20.0,
    blend_frames: int | None = None,
    seed: int = 0,
    statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,),
) -> list[ReferenceTrajectory]:
    """Loops repeated to ``target_seconds`` plus random walks of similar length."""
    target = int(round(target_seconds / FRAME_DT))
    out: list[ReferenceTrajectory] = []
    for loop in loops or []:
        repeats = max(1, int(np.ceil(target / max(1, loop.n_frames))))
        out.append(loop_to_trajectory(graph, loop, repeats=repeats, blend_frames=blend_frames))
    rng = np.random.default_rng(seed)
    tries = 0
    while len(out) < (len(loops or []) + n_walks) and tries < 20 * n_walks:
        tries += 1
        path = random_walk(graph, rng, min_frames=target // 2, max_frames=target, statuses=statuses)
        if path:
            out.append(path_to_trajectory(graph, path, blend_frames=blend_frames))
    return out
