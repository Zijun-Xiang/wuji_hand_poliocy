"""Find closed walks (loops) in the stitched motion graph.

A single episode is a chain, so a loop *must* use at least one jump.  For a
jump ``u -> v`` the shortest way back from ``v`` to ``u`` (through recorded
frames and other jumps) closes a loop.  Loops are what we ultimately want to
run on the hand: a reference the fingers can follow forever that keeps the pen
spinning, assembled from pieces of different demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from ..data.transforms import quat_angle
from .motion_graph import JumpStatus, MotionGraph


@dataclass
class Loop:
    nodes: np.ndarray  # ordered node ids; nodes[-1] -> nodes[0] closes the loop
    jump_edges: list[tuple[int, int]]  # (src, dst) jumps used, in order
    jump_costs: list[float]
    n_frames: int
    duration_s: float
    pen_rotation_rad: float  # total pen rotation accumulated around the loop
    episodes: list[int] = field(default_factory=list)  # distinct episode indices touched

    @property
    def n_jumps(self) -> int:
        return len(self.jump_edges)

    @property
    def total_jump_cost(self) -> float:
        return float(sum(self.jump_costs))

    @property
    def max_jump_cost(self) -> float:
        return float(max(self.jump_costs)) if self.jump_costs else 0.0

    def key(self) -> frozenset[tuple[int, int]]:
        return frozenset(self.jump_edges)

    def describe(self, graph: MotionGraph) -> str:
        ids = ", ".join(graph.episodes[e].episode_id for e in self.episodes)
        return (
            f"loop: {self.n_frames} frames ({self.duration_s:.2f}s), {self.n_jumps} jumps "
            f"(max cost {self.max_jump_cost:.2f}), pen rotation {np.rad2deg(self.pen_rotation_rad):.0f} deg, "
            f"episodes [{ids}]"
        )


def _path_from_predecessors(pred: np.ndarray, src: int, dst: int) -> list[int] | None:
    path = [dst]
    node = dst
    while node != src:
        node = int(pred[node])
        if node < 0:
            return None
        path.append(node)
        if len(path) > pred.size:  # defensive: malformed predecessor array
            return None
    path.reverse()
    return path


def _jump_lookup(graph: MotionGraph, statuses: tuple[JumpStatus, ...]) -> dict[tuple[int, int], float]:
    J = graph.jumps
    mask = np.isin(J.status, [int(s) for s in statuses])
    out: dict[tuple[int, int], float] = {}
    for s, d, c in zip(J.src[mask], J.dst[mask], J.cost[mask]):
        key = (int(s), int(d))
        if key not in out or c < out[key]:
            out[key] = float(c)
    return out


def _loop_from_nodes(graph: MotionGraph, nodes: list[int], jumps: dict[tuple[int, int], float]) -> Loop:
    arr = np.asarray(nodes, dtype=np.int64)
    edges, costs = [], []
    for a, b in zip(arr, np.roll(arr, -1)):
        if not (b == a + 1 and graph.episode_index[a] == graph.episode_index[b]):
            edges.append((int(a), int(b)))
            costs.append(jumps.get((int(a), int(b)), float("nan")))
    pen_q = np.stack([graph.node_pen_pose(n)[3:] for n in arr])
    rot = float(np.sum(quat_angle(pen_q, np.roll(pen_q, -1, axis=0))))
    dt = graph.episodes[0].frame_dt
    return Loop(
        nodes=arr,
        jump_edges=edges,
        jump_costs=costs,
        n_frames=int(arr.size),
        duration_s=float(arr.size * dt),
        pen_rotation_rad=rot,
        episodes=sorted({int(e) for e in graph.episode_index[arr]}),
    )


def find_loops(
    graph: MotionGraph,
    *,
    statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,),
    max_jumps_considered: int = 3000,
    max_loops: int = 200,
    min_frames: int = 25,
    max_frames: int = 3000,
    intra_cost: float = 1.0,
    jump_cost_offset: float = 25.0,
    jump_cost_scale: float = 10.0,
    verbose: bool = True,
) -> list[Loop]:
    """Close loops through the lowest-cost jumps.

    ``jump_cost_offset``/``jump_cost_scale`` make Dijkstra prefer staying on
    recorded frames (cost 1 per frame) over hopping: a jump costs as much as
    ``offset + scale * window_cost`` frames.
    """
    jumps = _jump_lookup(graph, statuses)
    if not jumps:
        return []
    adj = graph.adjacency(
        jump_statuses=statuses,
        intra_cost=intra_cost,
        jump_cost_scale=jump_cost_scale,
        jump_cost_offset=jump_cost_offset,
    )
    # Only jumps whose endpoints share a non-trivial strongly connected
    # component can lie on a cycle; skip the rest before running Dijkstra.
    _, labels = connected_components(adj, directed=True, connection="strong")
    sizes = np.bincount(labels)
    on_cycle = {k: c for k, c in jumps.items() if labels[k[0]] == labels[k[1]] and sizes[labels[k[0]]] > 1}
    ordered = sorted(on_cycle.items(), key=lambda kv: kv[1])[:max_jumps_considered]
    if verbose:
        print(f"  {len(on_cycle)} of {len(jumps)} jumps lie on cycles")
    loops: dict[frozenset[tuple[int, int]], Loop] = {}
    limit = max_frames * intra_cost + jump_cost_offset * 8 + jump_cost_scale * 8 * 10
    for i, ((u, v), _cost) in enumerate(ordered):
        if verbose and i % 100 == 0:
            print(f"  closing loops: {i}/{len(ordered)} jumps, {len(loops)} loops", end="\r")
        dist, pred = dijkstra(adj, directed=True, indices=v, return_predecessors=True, limit=limit)
        if not np.isfinite(dist[u]):
            continue
        back = _path_from_predecessors(pred, v, u)
        if back is None:
            continue
        loop = _loop_from_nodes(graph, back, jumps)  # closes with u -> v automatically
        if not (min_frames <= loop.n_frames <= max_frames):
            continue
        key = loop.key()
        if key not in loops:
            loops[key] = loop
        if len(loops) >= max_loops:
            break
    if verbose:
        print()
    result = list(loops.values())
    result.sort(key=lambda L: (L.n_jumps, L.max_jump_cost, -L.pen_rotation_rad))
    return result


@dataclass
class ComponentReport:
    labels: np.ndarray  # SCC label per node
    sizes: np.ndarray  # size of each component (index = label)
    largest_label: int

    @property
    def largest_size(self) -> int:
        return int(self.sizes[self.largest_label])

    def nodes_in_largest(self) -> np.ndarray:
        return np.flatnonzero(self.labels == self.largest_label)


def strongly_connected(graph: MotionGraph, *, statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,)) -> ComponentReport:
    """Strongly connected components of the stitched graph.

    Nodes in a non-trivial SCC can be revisited forever: that is the set of
    states from which the pen can keep spinning using only validated moves."""
    adj = graph.adjacency(jump_statuses=statuses)
    n, labels = connected_components(adj, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=n)
    return ComponentReport(labels=labels, sizes=sizes, largest_label=int(np.argmax(sizes)))


def loop_reference(graph: MotionGraph, loop: Loop, *, repeats: int = 1) -> dict[str, np.ndarray]:
    """Raw reference arrays (no blending) for running a loop ``repeats`` times."""
    nodes = np.tile(loop.nodes, repeats)
    qpos = np.stack([graph.node_qpos(n) for n in nodes])
    pen = np.stack([graph.node_pen_pose(n) for n in nodes])
    return {"nodes": nodes, "qpos": qpos, "pen_pose": pen}


def loop_closing_candidates(graph: MotionGraph, *, statuses: tuple[JumpStatus, ...] = (JumpStatus.CANDIDATE, JumpStatus.VALIDATED)) -> np.ndarray:
    """Indices of jumps (with the given statuses) whose endpoints lie in the same
    non-trivial strongly connected component of the graph built from those
    jumps - i.e. the only jumps that can ever be part of a loop."""
    J = graph.jumps
    mask = np.isin(J.status, [int(s) for s in statuses])
    if not mask.any():
        return np.zeros(0, dtype=np.int64)
    adj = graph.adjacency(jump_statuses=statuses)
    _, labels = connected_components(adj, directed=True, connection="strong")
    sizes = np.bincount(labels)
    idx = np.flatnonzero(mask)
    same = labels[J.src[idx]] == labels[J.dst[idx]]
    nontrivial = sizes[labels[J.src[idx]]] > 1
    return idx[same & nontrivial]
