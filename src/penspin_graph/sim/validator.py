"""Decide in simulation whether a proposed jump ``u -> v`` is executable.

For every candidate we run two short rollouts from the *same* starting state
(the hand and pen exactly as recorded at frame ``u``):

* **stitched**: ``u`` -> blend -> ``v, v+1, ..., v+H-1``  (the jump)
* **baseline**: ``u, u+1, ..., u+H+B``                      (what the data did)

Both use the same low-level controller (Wuji's released tracking policy by
default).  A jump is **validated** when the stitched rollout keeps the pen
within the termination thresholds for the whole horizon, ends close to its
reference, and - optionally - tracks no worse than the baseline did.  The
baseline comparison matters because the references are retargeted human
motion: even the original clips are only "approximately" trackable, so the
fair question is *"is this transition at least as executable as the data
itself?"*
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..graph.motion_graph import JumpStatus, MotionGraph
from ..graph.planner import path_to_trajectory
from .controller import Controller
from .rollout import RolloutResult, Thresholds, frame_state_from_episode, make_controller, rollout
from .scene import PenSpinScene, SceneConfig


@dataclass
class ValidationConfig:
    controller: str = "onnx"  # "onnx" | "pd"
    blend_frames: int | None = None  # None -> the per-jump blend chosen by the kinematic filter
    follow_frames: int = 25  # 0.5 s after landing in the new episode
    thresholds: Thresholds = field(default_factory=Thresholds)
    success_pos_m: float = 0.03  # mean error over the last ``tail`` ticks
    success_axis_rad: float = np.deg2rad(30.0)
    tail: int = 5
    relative: bool = True
    relative_factor: float = 1.5
    relative_floor_m: float = 0.01
    scene: SceneConfig = field(default_factory=SceneConfig)


@dataclass
class JumpValidation:
    jump_index: int
    src: int
    dst: int
    ok: bool
    reason: str
    stitched: RolloutResult
    baseline: RolloutResult | None
    score: float  # stitched mean pen position error over the horizon (m)

    def summary(self, graph: MotionGraph | None = None) -> str:
        head = f"jump #{self.jump_index}"
        if graph is not None:
            head += f" {graph.node_label(self.src)} -> {graph.node_label(self.dst)}"
        base = "" if self.baseline is None else f" | baseline {self.baseline.summary()}"
        return f"{head}: {'VALID' if self.ok else 'REJECT'} ({self.reason}) | stitched {self.stitched.summary()}{base}"


class TransitionValidator:
    def __init__(self, graph: MotionGraph, cfg: ValidationConfig | None = None, *, scene: PenSpinScene | None = None, controller: Controller | None = None):
        self.graph = graph
        self.cfg = cfg or ValidationConfig()
        self.scene = scene or PenSpinScene(self.cfg.scene)
        self.controller = controller or make_controller(self.cfg.controller)
        self._baseline_cache: dict[tuple[int, int], RolloutResult] = {}

    # ------------------------------------------------------------------ #
    def _stitched_reference(self, u: int, v: int):
        g = self.graph
        H = min(self.cfg.follow_frames, int(g.frames_remaining(np.array([v]))[0]) + 1)
        nodes = [u] + list(range(v, v + H))
        return path_to_trajectory(g, nodes, blend_frames=self.cfg.blend_frames), H

    def _baseline_reference(self, u: int, length: int):
        g = self.graph
        avail = int(g.frames_remaining(np.array([u]))[0])
        n = min(length, avail)
        return path_to_trajectory(g, list(range(u, u + n + 1)), blend_frames=0)

    def validate(self, jump_index: int, *, record_states: bool = False) -> JumpValidation:
        g, cfg = self.graph, self.cfg
        u, v = int(g.jumps.src[jump_index]), int(g.jumps.dst[jump_index])
        e, t = g.node_episode_frame(u)
        start = frame_state_from_episode(g.episodes[e], t)

        ref_jump, H = self._stitched_reference(u, v)
        stitched = rollout(self.scene, self.controller, ref_jump, start, thresholds=cfg.thresholds, record_states=record_states)
        score = stitched.mean_pos_err()

        baseline = None
        if cfg.relative and not g.static_mask[u]:  # a resting source has no meaningful "original continuation"
            key = (u, len(ref_jump) - 1)
            baseline = self._baseline_cache.get(key)
            if baseline is None:
                ref_base = self._baseline_reference(u, len(ref_jump) - 1)
                baseline = rollout(self.scene, self.controller, ref_base, start, thresholds=cfg.thresholds)
                if len(self._baseline_cache) > 4096:
                    self._baseline_cache.clear()
                self._baseline_cache[key] = baseline

        if stitched.terminated:
            return JumpValidation(jump_index, u, v, False, f"terminated:{stitched.termination_reason}", stitched, baseline, score)
        tail_pos = stitched.mean_pos_err(cfg.tail)
        tail_axis = stitched.mean_axis_err(cfg.tail)
        if tail_pos > cfg.success_pos_m:
            return JumpValidation(jump_index, u, v, False, f"end_pos_err {100 * tail_pos:.1f}cm", stitched, baseline, score)
        if tail_axis > cfg.success_axis_rad:
            return JumpValidation(jump_index, u, v, False, f"end_axis_err {np.rad2deg(tail_axis):.0f}deg", stitched, baseline, score)
        if baseline is not None and not baseline.terminated:
            allowed = cfg.relative_factor * max(baseline.mean_pos_err(), cfg.relative_floor_m)
            if score > allowed:
                return JumpValidation(jump_index, u, v, False, f"worse_than_baseline {100 * score:.1f}>{100 * allowed:.1f}cm", stitched, baseline, score)
        return JumpValidation(jump_index, u, v, True, "ok", stitched, baseline, score)

    # ------------------------------------------------------------------ #
    def validate_many(self, indices: np.ndarray, *, verbose: bool = True) -> list[JumpValidation]:
        out = []
        t0 = time.time()
        for n, idx in enumerate(indices):
            out.append(self.validate(int(idx)))
            if verbose and (n % 25 == 0 or n == len(indices) - 1):
                ok = sum(r.ok for r in out)
                el = time.time() - t0
                print(f"  validated {n + 1}/{len(indices)}: {ok} ok ({el:.0f}s)", end="\r")
        if verbose:
            print()
        return out


# ---------------------------------------------------------------------- #
# multiprocessing support (fork): each worker builds its own scene+policy
# ---------------------------------------------------------------------- #
_WORKER: dict[str, object] = {}


def _worker_init(graph: MotionGraph, cfg: ValidationConfig) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER["validator"] = TransitionValidator(graph, cfg)


def _worker_run(idx: int) -> tuple[int, bool, str, float]:
    r = _WORKER["validator"].validate(idx)
    return idx, r.ok, r.reason, r.score


def validate_candidates(
    graph: MotionGraph,
    cfg: ValidationConfig | None = None,
    *,
    max_jumps: int | None = None,
    workers: int = 1,
    verbose: bool = True,
    priority: np.ndarray | None = None,
    only_priority: bool = False,
    interleave_same_episode: bool = True,
    rest_only: bool = False,
) -> dict[str, object]:
    """Validate candidate jumps (cheapest first), updating ``graph.jumps`` in place.

    ``priority`` (jump indices) are validated before the rest, each group
    cheapest-first; with ``only_priority`` nothing else is tested.  Returns a
    small report dict.  With ``workers > 1`` a forked pool is used.
    """
    cfg = cfg or ValidationConfig()
    idx = graph.jumps.with_status(JumpStatus.CANDIDATE)
    if rest_only:
        idx = idx[graph.is_rest_transition(idx)]
    idx = idx[np.argsort(graph.jumps.cost[idx], kind="stable")]
    if priority is not None:
        pri = np.asarray(priority)
        is_pri = np.isin(idx, pri)
        idx = idx[is_pri] if only_priority else np.concatenate([idx[is_pri], idx[~is_pri]])
    if interleave_same_episode and idx.size:
        same = graph.episode_index[graph.jumps.src[idx]] == graph.episode_index[graph.jumps.dst[idx]]
        a, b = idx[same], idx[~same]
        # within one episode, jumps that go *back* in time are the ones that
        # can close a loop on their own: test them first
        if a.size:
            back = graph.frame_index[graph.jumps.dst[a]] < graph.frame_index[graph.jumps.src[a]]
            a = np.concatenate([a[back], a[~back]])
        if a.size and b.size:
            merged = np.empty(idx.size, dtype=idx.dtype)
            n = min(a.size, b.size)
            merged[0 : 2 * n : 2] = a[:n]
            merged[1 : 2 * n : 2] = b[:n]
            merged[2 * n :] = a[n:] if a.size > n else b[n:]
            idx = merged
    if max_jumps is not None:
        idx = idx[:max_jumps]
    # group by source node so a worker's baseline cache is reused across jumps from the same frame
    idx = idx[np.argsort(graph.jumps.src[idx], kind="stable")]
    reasons: dict[str, int] = {}
    t0 = time.time()
    results: list[tuple[int, bool, str, float]] = []
    if workers <= 1 or len(idx) < 2 * workers:
        v = TransitionValidator(graph, cfg)
        for n, i in enumerate(idx):
            r = v.validate(int(i))
            results.append((int(i), r.ok, r.reason, r.score))
            if verbose and (n % 20 == 0 or n == len(idx) - 1):
                ok = sum(x[1] for x in results)
                print(f"  sim-validating {n + 1}/{len(idx)} jumps: {ok} validated ({time.time() - t0:.0f}s)", end="\r")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_worker_init, initargs=(graph, cfg)) as pool:
            for n, res in enumerate(pool.imap_unordered(_worker_run, [int(i) for i in idx], chunksize=16)):
                results.append(res)
                if verbose and (n % 20 == 0 or n == len(idx) - 1):
                    ok = sum(x[1] for x in results)
                    print(f"  sim-validating {n + 1}/{len(idx)} jumps: {ok} validated ({time.time() - t0:.0f}s)", end="\r")
    if verbose:
        print()
    for i, ok, reason, score in results:
        graph.jumps.status[i] = int(JumpStatus.VALIDATED if ok else JumpStatus.SIM_REJECTED)
        graph.jumps.sim_score[i] = score
        key = reason.split(" ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    n_ok = int(sum(x[1] for x in results))
    return {
        "tested": len(results),
        "validated": n_ok,
        "rejected": len(results) - n_ok,
        "reasons": reasons,
        "seconds": time.time() - t0,
        "controller": cfg.controller,
    }
