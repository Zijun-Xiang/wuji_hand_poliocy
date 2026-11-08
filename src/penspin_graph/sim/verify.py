"""End-to-end verification of loops: replay the whole stitched loop in simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.loops import Loop
from ..graph.motion_graph import MotionGraph
from ..graph.planner import loop_to_trajectory
from .controller import Controller
from .rollout import RolloutResult, frame_state_from_episode, rollout
from .scene import PenSpinScene


@dataclass
class LoopVerification:
    index: int
    completed: bool
    survived_fraction: float  # fraction of the reference the hand got through
    ticks: int
    result: RolloutResult
    blamed_jump: tuple[int, int] | None = None  # (src, dst) jump held responsible for a failure

    def summary(self) -> str:
        return f"loop {self.index}: {'COMPLETED' if self.completed else 'failed'} after {100 * self.survived_fraction:.0f}% ({self.ticks} ticks) | {self.result.summary()}"


def _blame_jump(graph: MotionGraph, traj, fail_tick: int, window: int) -> tuple[int, int] | None:
    """The last jump taken within ``window`` frames before the failure, if any."""
    nodes = traj.nodes
    k = min(fail_tick + 1, len(nodes) - 1)
    lo = max(1, k - window)
    for i in range(k, lo - 1, -1):
        a, b = int(nodes[i - 1]), int(nodes[i])
        if b == -1:
            continue
        if a == -1:  # end of a blend: walk back to the real node before the blend
            j = i - 1
            while j >= 0 and nodes[j] == -1:
                j -= 1
            if j >= 0:
                return int(nodes[j]), b
            return None
        if not (b == a + 1 and graph.episode_index[a] == graph.episode_index[b]):
            return a, b
    return None


def verify_loops(
    graph: MotionGraph,
    loops: list[Loop],
    scene: PenSpinScene,
    controller: Controller,
    *,
    repeats: int = 1,
    blame_window: int = 100,
    verbose: bool = True,
) -> list[LoopVerification]:
    out = []
    for i, loop in enumerate(loops):
        traj = loop_to_trajectory(graph, loop, repeats=repeats)
        e, t = graph.node_episode_frame(int(loop.nodes[0]))
        res = rollout(scene, controller, traj, frame_state_from_episode(graph.episodes[e], t))
        frac = res.n_steps / max(len(traj) - 1, 1)
        blamed = None if res.completed else _blame_jump(graph, traj, res.termination_step, blame_window)
        v = LoopVerification(i, res.completed, float(frac), res.n_steps, res, blamed)
        out.append(v)
        if verbose:
            extra = f"  <- blamed jump {graph.node_label(blamed[0])} -> {graph.node_label(blamed[1])}" if blamed else ""
            print("  " + v.summary() + extra)
    return out


def reject_jumps(graph: MotionGraph, pairs: list[tuple[int, int]]) -> int:
    """Mark the given (src, dst) jumps as SIM_REJECTED; returns how many changed."""
    from ..graph.motion_graph import JumpStatus

    J = graph.jumps
    n = 0
    for s_, d_ in set(pairs):
        hit = np.flatnonzero((J.src == s_) & (J.dst == d_) & (J.status == int(JumpStatus.VALIDATED)))
        if hit.size:
            J.status[hit] = int(JumpStatus.SIM_REJECTED)
            n += int(hit.size)
    return n
