"""Query + plan on the motion graph.

* :func:`nearest_nodes` – "the hand is in some random state: which recorded
  frame is it closest to?"
* :class:`GraphPlanner` – shortest paths through recorded frames and validated
  jumps, turned into a smooth 50 Hz reference trajectory (joint targets + pen
  reference) with short blends where the path jumps between episodes.
* :class:`MotionMatchingController` – the closed-loop version: every step it
  re-asks "where am I?" and emits the next joint target, following the current
  episode by default and switching only when a clearly closer frame exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import dijkstra

from ..data.transforms import quat_slerp
from ..features.featurizer import FrameState
from .loops import Loop, _path_from_predecessors
from .motion_graph import JumpStatus, MotionGraph


def nearest_nodes(
    graph: MotionGraph, state: FrameState, *, k: int = 1, allowed: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distances_in_step_units, node_ids)`` of the k closest frames."""
    q = graph.featurizer.transform_state(state)[None]
    if allowed is None:
        d, idx = graph.knn(q, k)
    else:
        d, idx = graph.knn_filtered(q, k, allowed=allowed)
    return d[0] / graph.step_unit, idx[0]


@dataclass
class ReferenceTrajectory:
    """A 50 Hz reference the controller / simulator / robot follows."""

    qpos: np.ndarray  # (T, 20) joint targets
    pen_pose: np.ndarray  # (T, 7) wrist-local reference pen pose
    nodes: np.ndarray  # (T,) source node id per frame, -1 for blended frames
    frame_dt: float
    link_pos: np.ndarray | None = None  # (T, 21, 3) wrist-local reference link positions
    group_ids: list[str] | None = None  # motion group of each frame (policy selection)

    def __len__(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration_s(self) -> float:
        return len(self) * self.frame_dt

    def max_joint_velocity(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.abs(np.diff(self.qpos, axis=0)).max() / self.frame_dt)


def _blend(q_a: np.ndarray, q_b: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, l_a: np.ndarray, l_b: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` intermediate frames between frame a and frame b (exclusive)."""
    if n <= 0:
        return np.zeros((0, q_a.shape[-1])), np.zeros((0, 7)), np.zeros((0,) + l_a.shape)
    alphas = (np.arange(1, n + 1) / (n + 1))[:, None]
    q = (1 - alphas) * q_a + alphas * q_b
    pos = (1 - alphas) * p_a[:3] + alphas * p_b[:3]
    quat = quat_slerp(np.broadcast_to(p_a[3:], (n, 4)), np.broadcast_to(p_b[3:], (n, 4)), alphas[:, 0])
    links = (1 - alphas[:, :, None]) * l_a[None] + alphas[:, :, None] * l_b[None]
    return q, np.concatenate([pos, quat], axis=1), links


def _node_link_pos(graph: MotionGraph, node: int) -> np.ndarray:
    e, t = graph.node_episode_frame(node)
    return graph.episodes[e].link_pos[t]


def _node_group(graph: MotionGraph, node: int) -> str:
    return graph.episodes[graph.episode_index[node]].group_id


def path_to_trajectory(graph: MotionGraph, nodes: np.ndarray | list[int], *, blend_frames: int | None = None, close_loop: bool = False) -> ReferenceTrajectory:
    """Convert a node path into a reference, inserting interpolated frames at
    every jump (consecutive nodes that are not ``t, t+1`` of the same episode).

    The number of blended frames per jump is the one chosen by the kinematic
    filter (``graph.jumps.blend``) when the pair is a known jump, otherwise
    ``blend_frames`` (default 2).  With ``close_loop`` the last node also blends
    back into the first one (and the first node is not repeated at the end)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    lookup = graph.jumps.blend_lookup() if len(graph.jumps) else {}
    default_blend = 2 if blend_frames is None else int(blend_frames)
    q_list, p_list, l_list, n_list, g_list = [], [], [], [], []
    seq = list(nodes) + ([int(nodes[0])] if close_loop else [])
    for i, node in enumerate(seq[:-1]):
        nxt = seq[i + 1]
        q_list.append(graph.node_qpos(node)[None])
        p_list.append(graph.node_pen_pose(node)[None])
        l_list.append(_node_link_pos(graph, node)[None])
        n_list.append([node])
        g_list.append(_node_group(graph, node))
        is_intra = nxt == node + 1 and graph.episode_index[node] == graph.episode_index[nxt]
        if not is_intra:
            nb = lookup.get((int(node), int(nxt)), default_blend) if blend_frames is None else default_blend
            qb, pb, lb = _blend(
                graph.node_qpos(node), graph.node_qpos(nxt), graph.node_pen_pose(node), graph.node_pen_pose(nxt),
                _node_link_pos(graph, node), _node_link_pos(graph, nxt), nb,
            )
            if len(qb):
                q_list.append(qb)
                p_list.append(pb)
                l_list.append(lb)
                n_list.append([-1] * len(qb))
                g_list.extend([_node_group(graph, nxt)] * len(qb))
    if not close_loop:
        last = seq[-1]
        q_list.append(graph.node_qpos(last)[None])
        p_list.append(graph.node_pen_pose(last)[None])
        l_list.append(_node_link_pos(graph, last)[None])
        n_list.append([last])
        g_list.append(_node_group(graph, last))
    return ReferenceTrajectory(
        qpos=np.concatenate(q_list, axis=0),
        pen_pose=np.concatenate(p_list, axis=0),
        nodes=np.concatenate([np.asarray(n) for n in n_list]),
        frame_dt=graph.episodes[0].frame_dt,
        link_pos=np.concatenate(l_list, axis=0),
        group_ids=g_list,
    )


def loop_to_trajectory(graph: MotionGraph, loop: Loop, *, repeats: int = 1, blend_frames: int | None = None) -> ReferenceTrajectory:
    one = path_to_trajectory(graph, loop.nodes, blend_frames=blend_frames, close_loop=True)
    return ReferenceTrajectory(
        qpos=np.tile(one.qpos, (repeats, 1)),
        pen_pose=np.tile(one.pen_pose, (repeats, 1)),
        nodes=np.tile(one.nodes, repeats),
        frame_dt=one.frame_dt,
        link_pos=np.tile(one.link_pos, (repeats, 1, 1)),
        group_ids=list(one.group_ids) * repeats,
    )


@dataclass
class GraphPlanner:
    graph: MotionGraph
    statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,)
    intra_cost: float = 1.0
    jump_cost_offset: float = 25.0
    jump_cost_scale: float = 10.0
    blend_frames: int | None = None  # None -> per-jump blend from the kinematic filter
    _adj: object = field(default=None, repr=False)

    def adjacency(self):
        if self._adj is None:
            self._adj = self.graph.adjacency(
                jump_statuses=self.statuses,
                intra_cost=self.intra_cost,
                jump_cost_scale=self.jump_cost_scale,
                jump_cost_offset=self.jump_cost_offset,
            )
        return self._adj

    def shortest_path(self, src: int, dst: int) -> list[int] | None:
        dist, pred = dijkstra(self.adjacency(), directed=True, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None
        return _path_from_predecessors(pred, src, dst)

    def path_to_set(self, src: int, targets: np.ndarray) -> list[int] | None:
        """Shortest path from ``src`` to the closest node of ``targets``."""
        dist, pred = dijkstra(self.adjacency(), directed=True, indices=src, return_predecessors=True)
        targets = np.asarray(targets)
        finite = np.isfinite(dist[targets])
        if not finite.any():
            return None
        best = targets[finite][np.argmin(dist[targets][finite])]
        return _path_from_predecessors(pred, src, int(best))

    def plan(self, state: FrameState, goal_node: int) -> ReferenceTrajectory | None:
        _, idx = nearest_nodes(self.graph, state)
        path = self.shortest_path(int(idx[0]), goal_node)
        return None if path is None else path_to_trajectory(self.graph, path, blend_frames=self.blend_frames)

    def plan_to_loop(self, state: FrameState, loop: Loop, *, repeats: int = 2) -> ReferenceTrajectory | None:
        """Reference that goes from the nearest frame into ``loop`` and then
        circles it ``repeats`` times."""
        _, idx = nearest_nodes(self.graph, state)
        start = int(idx[0])
        path = self.path_to_set(start, loop.nodes)
        if path is None:
            return None
        entry = path[-1]
        k = int(np.flatnonzero(loop.nodes == entry)[0])
        rotated = np.concatenate([loop.nodes[k:], loop.nodes[:k]])
        seq = list(path[:-1]) + list(np.tile(rotated, repeats))
        return path_to_trajectory(self.graph, seq, blend_frames=self.blend_frames)


@dataclass
class TrackedFrame:
    """What the controller wants the hand to track right now."""

    qpos: np.ndarray
    pen_pose: np.ndarray
    link_pos: np.ndarray
    group_id: str
    node: int  # source node, -1 for a blended frame


@dataclass
class MotionMatchingController:
    """Closed-loop "find the closest point, then follow" controller.

    Each :meth:`step` receives the current measured state and returns the joint
    target for the next control tick (the full tracked frame is in
    :attr:`current_ref`).  By default it advances along the frame it is
    currently tracking; every ``requery_every`` ticks (or when the state drifts
    further than ``drift_threshold`` step units from the tracked frame) it
    re-queries the graph and switches if a clearly closer frame exists.

    To keep spinning past the end of a clip it *plans an escape* through
    validated jumps: when fewer than ``jump_when_remaining_below`` active frames
    are left in the current episode (the recorded motion is about to stop), it
    runs Dijkstra over recorded frames + validated jumps to the nearest frame
    with a long active future (in another episode, or far behind in the same
    one) and follows that path, inserting the jump's blend frames.  After a
    jump it dwells for ``cooldown`` ticks before planning again.  With
    ``restrict_to`` the nearest-frame search is limited to a node mask (e.g.
    the largest SCC)."""

    graph: MotionGraph
    requery_every: int = 10
    drift_threshold: float = 3.0
    switch_margin: float = 1.0
    restrict_to: np.ndarray | None = None  # boolean node mask
    jump_when_remaining_below: int = 250
    min_dst_remaining: int = 100
    cooldown: int = 50
    prefer_margin: float = 2.0  # use a frame inside ``restrict_to`` only if it is within this many step units of the true nearest
    jump_statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,)
    current: int = -1
    tick: int = 0
    history: list[int] = field(default_factory=list)
    jumps_taken: int = 0
    current_ref: TrackedFrame | None = None
    _plan: list[int] = field(default_factory=list, repr=False)
    _blend_queue: list[TrackedFrame] = field(default_factory=list, repr=False)
    _blend_target: int = -1
    _since_jump: int = 10**9
    _adj: object = field(default=None, repr=False)
    _targets_ok: np.ndarray | None = field(default=None, repr=False)
    _blend_lookup: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        g = self.graph
        self._adj = g.adjacency(jump_statuses=self.jump_statuses, intra_cost=1.0, jump_cost_scale=10.0, jump_cost_offset=25.0)
        self._targets_ok = g.active_remaining(np.arange(g.num_nodes)) >= self.min_dst_remaining
        if self.restrict_to is not None:
            self._targets_ok &= self.restrict_to
        self._blend_lookup = g.jumps.blend_lookup() if len(g.jumps) else {}

    # ------------------------------------------------------------------ #
    def _frame(self, node: int) -> TrackedFrame:
        g = self.graph
        e, t = g.node_episode_frame(node)
        ep = g.episodes[e]
        return TrackedFrame(ep.qpos[t], ep.pen_pose[t], ep.link_pos[t], ep.group_id, node)

    def _nearest(self, state: FrameState, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Nearest frames; frames inside ``restrict_to`` are preferred when they
        are almost as close as the true nearest frame."""
        d_all, i_all = nearest_nodes(self.graph, state, k=k)
        if self.restrict_to is None:
            return d_all, i_all
        d_in, i_in = nearest_nodes(self.graph, state, k=k, allowed=self.restrict_to)
        if i_in[0] >= 0 and d_in[0] <= d_all[0] + self.prefer_margin:
            return d_in, i_in
        return d_all, i_all

    def reset(self, state: FrameState) -> int:
        _, idx = self._nearest(state)
        self.current = int(idx[0])
        self.tick = 0
        self.history = [self.current]
        self.jumps_taken = 0
        self._plan = []
        self._blend_queue = []
        self._blend_target = -1
        self._since_jump = 10**9
        self.current_ref = self._frame(self.current)
        return self.current

    def _distance_to(self, node: int, state: FrameState) -> float:
        f = self.graph.featurizer.transform_state(state)
        return float(np.linalg.norm(f - self.graph.features[node]) / self.graph.step_unit)

    def _plan_escape(self, node: int) -> list[int]:
        """Shortest path from ``node`` to a frame with a long active future that
        is not just further along the current episode."""
        g = self.graph
        dist, pred = dijkstra(self._adj, directed=True, indices=node, return_predecessors=True, limit=1500.0)
        e = g.episode_index[node]
        same = g.episode_index == e
        not_ahead = (~same) | (g.frame_index < g.frame_index[node] - 50)
        ok = self._targets_ok & np.isfinite(dist) & not_ahead
        # prefer destinations that will not need another escape right away
        long_future = ok & (g.active_remaining(np.arange(g.num_nodes)) >= self.jump_when_remaining_below)
        cand = np.flatnonzero(long_future) if long_future.any() else np.flatnonzero(ok)
        if cand.size == 0:
            return []
        best = int(cand[np.argmin(dist[cand])])
        path = _path_from_predecessors(pred, node, best)
        return [] if path is None or len(path) < 2 else path[1:]

    def _start_jump(self, node: int, nxt: int) -> None:
        """Queue the blend frames of the jump node -> nxt."""
        g = self.graph
        b = int(self._blend_lookup.get((int(node), int(nxt)), 2))
        if b > 0:
            qb, pb, lb = _blend(
                g.node_qpos(node), g.node_qpos(nxt), g.node_pen_pose(node), g.node_pen_pose(nxt),
                _node_link_pos(g, node), _node_link_pos(g, nxt), b,
            )
            group = _node_group(g, nxt)
            self._blend_queue = [TrackedFrame(qb[i], pb[i], lb[i], group, -1) for i in range(b)]
        self._blend_target = int(nxt)
        self.jumps_taken += 1
        self._since_jump = 0

    def step(self, state: FrameState) -> np.ndarray:
        if self.current < 0:
            self.reset(state)
        g = self.graph
        self.tick += 1
        self._since_jump += 1

        # inside a blend: just emit the next blended frame
        if self._blend_queue:
            frame = self._blend_queue.pop(0)
            if not self._blend_queue:
                self.current = self._blend_target
            self.current_ref = frame
            self.history.append(-1)
            return frame.qpos
        if self._blend_target >= 0:  # blend finished last tick: now sit on the destination frame
            node = self._blend_target
            self._blend_target = -1
            self.current = node
            self.current_ref = self._frame(node)
            self.history.append(node)
            return self.current_ref.qpos

        node = self.current
        drift = self._distance_to(node, state)
        if self._plan and drift > 2.0 * self.drift_threshold:  # lost the reference while following a plan
            self._plan = []
        at_end = g.successor(node) is None and not self._plan
        if at_end or (not self._plan and (self.tick % self.requery_every == 0 or drift > self.drift_threshold)):
            d, idx = self._nearest(state)
            for dist_i, cand in zip(d, idx):
                if cand < 0 or cand == node:
                    break
                if at_end or dist_i + self.switch_margin < drift:
                    node = int(cand)
                    self._plan = []
                break
        remaining = int(g.active_remaining(np.array([node]))[0])
        if not self._plan and self._since_jump >= self.cooldown and remaining < self.jump_when_remaining_below:
            self._plan = self._plan_escape(node)
        if self._plan:
            nxt = self._plan.pop(0)
            if not (nxt == node + 1 and g.episode_index[nxt] == g.episode_index[node]):
                self._start_jump(node, nxt)
                if self._blend_queue:
                    frame = self._blend_queue.pop(0)
                    if not self._blend_queue:
                        self.current = self._blend_target
                    self.current_ref = frame
                    self.history.append(-1)
                    return frame.qpos
                self._blend_target = -1
        else:
            nxt = g.successor(node)
        if nxt is None:  # end of an episode with nothing closer: hold the pose
            self.current = node
        else:
            self.current = nxt
        self.current_ref = self._frame(self.current)
        self.history.append(self.current)
        return self.current_ref.qpos
