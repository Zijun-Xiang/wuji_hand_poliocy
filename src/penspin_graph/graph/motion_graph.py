"""The motion graph: one node per frame, chains inside episodes, jumps between them.

Terminology
-----------
* **node** – one frame of one episode (a point in feature space).
* **intra edge** – ``(ep, t) -> (ep, t+1)``: what the dataset actually recorded.
* **jump** – a proposed edge ``u -> v`` between two *different* places in the
  data.  It says "after frame ``u`` the hand could continue with frame ``v``'s
  episode instead of its own".  A jump starts life as a *candidate*
  (similar in feature space), may be rejected by kinematic checks, and becomes
  *validated* once the simulator executes it successfully.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ..data.clips import Episode
from ..features.featurizer import FrameFeaturizer


class JumpStatus(IntEnum):
    CANDIDATE = 0
    KINEMATIC_FAIL = 1
    SIM_REJECTED = 2
    VALIDATED = 3


@dataclass
class Jumps:
    """Column store of jump edges (all arrays have the same length)."""

    src: np.ndarray  # node ids
    dst: np.ndarray  # node ids
    cost: np.ndarray  # window-matching cost (step units)
    pose_dist: np.ndarray  # ||f(src+1) - f(dst)|| (step units)
    status: np.ndarray  # JumpStatus values
    sim_score: np.ndarray  # filled by the validator (nan until then)
    blend: np.ndarray | None = None  # blend frames to insert for this jump (set by the kinematic filter)

    def __post_init__(self) -> None:
        if self.blend is None:
            self.blend = np.full(self.src.shape[0], 2, dtype=np.int16)

    def __len__(self) -> int:
        return int(self.src.shape[0])

    @classmethod
    def empty(cls) -> "Jumps":
        z = np.zeros(0)
        return cls(z.astype(np.int64), z.astype(np.int64), z, z, z.astype(np.int8), z, z.astype(np.int16))

    def subset(self, mask: np.ndarray) -> "Jumps":
        return Jumps(
            self.src[mask], self.dst[mask], self.cost[mask], self.pose_dist[mask],
            self.status[mask], self.sim_score[mask], self.blend[mask],
        )

    def blend_lookup(self) -> dict[tuple[int, int], int]:
        return {(int(s), int(d)): int(b) for s, d, b in zip(self.src, self.dst, self.blend)}

    def with_status(self, status: JumpStatus) -> np.ndarray:
        return np.flatnonzero(self.status == int(status))

    def summary(self) -> dict[str, int]:
        return {s.name.lower(): int(np.sum(self.status == int(s))) for s in JumpStatus}


class MotionGraph:
    def __init__(
        self,
        episodes: list[Episode],
        featurizer: FrameFeaturizer,
        features: np.ndarray | None = None,
    ) -> None:
        self.episodes = episodes
        self.featurizer = featurizer
        if features is None:
            feats = [featurizer.transform_episode(ep) for ep in episodes]
            features = np.concatenate(feats, axis=0)
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        lengths = np.array([len(ep) for ep in episodes], dtype=np.int64)
        self.episode_starts = np.concatenate([[0], np.cumsum(lengths)])
        self.episode_index = np.repeat(np.arange(len(episodes)), lengths)
        self.frame_index = np.concatenate([np.arange(n) for n in lengths])
        self.episode_lengths = lengths
        self.jumps = Jumps.empty()
        self._nn: NearestNeighbors | None = None
        # Natural distance unit: the median feature-space displacement of one
        # recorded 50 Hz step.  Jump costs are expressed as multiples of it.
        step = np.linalg.norm(self.features[self.intra_dst] - self.features[self.intra_src], axis=1)
        self.step_unit = float(np.median(step)) if step.size else 1.0
        # Frames where neither the fingers nor the pen move (the clips end with a
        # ~2 s hold).  ``active_end`` is, per episode, the index where the
        # trailing hold starts; a controller should jump before reaching it.
        qv = np.concatenate([np.abs(ep.qvel).max(axis=1) for ep in episodes])
        pw = np.concatenate([np.linalg.norm(ep.pen_angvel, axis=1) for ep in episodes])
        self.static_mask = (qv < 0.2) & (pw < 0.3)
        self.active_end = np.zeros(len(episodes), dtype=np.int64)
        for e in range(len(episodes)):
            s0, s1 = self.episode_starts[e], self.episode_starts[e + 1]
            k = s1 - 1
            while k > s0 and self.static_mask[k]:
                k -= 1
            self.active_end[e] = (k + 1) - s0

    # ------------------------------------------------------------------ #
    # basic accessors
    # ------------------------------------------------------------------ #
    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def intra_src(self) -> np.ndarray:
        has_next = self.frame_index < (self.episode_lengths[self.episode_index] - 1)
        return np.flatnonzero(has_next)

    @property
    def intra_dst(self) -> np.ndarray:
        return self.intra_src + 1

    def node_id(self, episode: int, frame: int) -> int:
        return int(self.episode_starts[episode] + frame)

    def node_episode_frame(self, node: int) -> tuple[int, int]:
        return int(self.episode_index[node]), int(self.frame_index[node])

    def node_label(self, node: int) -> str:
        e, t = self.node_episode_frame(node)
        return f"{self.episodes[e].episode_id}@{t}"

    def frames_remaining(self, nodes: np.ndarray) -> np.ndarray:
        """Number of frames after ``nodes`` inside their own episodes."""
        nodes = np.asarray(nodes)
        return self.episode_lengths[self.episode_index[nodes]] - 1 - self.frame_index[nodes]

    def active_remaining(self, nodes: np.ndarray) -> np.ndarray:
        """Frames left before the episode's trailing hold starts (0 inside the hold)."""
        nodes = np.asarray(nodes)
        return np.maximum(self.active_end[self.episode_index[nodes]] - self.frame_index[nodes], 0)

    def successor(self, node: int) -> int | None:
        return node + 1 if self.frames_remaining(np.array([node]))[0] > 0 else None

    def node_qpos(self, node: int) -> np.ndarray:
        e, t = self.node_episode_frame(node)
        return self.episodes[e].qpos[t]

    def node_pen_pose(self, node: int) -> np.ndarray:
        e, t = self.node_episode_frame(node)
        return self.episodes[e].pen_pose[t]

    # ------------------------------------------------------------------ #
    # nearest neighbours
    # ------------------------------------------------------------------ #
    def _index(self) -> NearestNeighbors:
        if self._nn is None:
            self._nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(self.features)
        return self._nn

    def knn(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Plain k-nearest nodes for feature-space queries (Q, D) -> (dist, idx)."""
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        return self._index().kneighbors(queries, n_neighbors=min(k, self.num_nodes))

    def knn_filtered(
        self,
        queries: np.ndarray,
        k: int,
        *,
        exclude_episode: np.ndarray | None = None,
        min_frame_gap: int = 0,
        query_frame: np.ndarray | None = None,
        allowed: np.ndarray | None = None,
        oversample: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """k nearest nodes with filters.

        ``exclude_episode[i]``: episode index whose nodes are only accepted if
        they are at least ``min_frame_gap`` frames away from ``query_frame[i]``
        (set ``min_frame_gap`` huge to exclude the episode entirely).
        ``allowed``: boolean node mask of eligible targets.  Returns ``-1`` ids
        and ``inf`` distances when fewer than ``k`` neighbours survive.
        """
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        Q = queries.shape[0]
        n_req = min(self.num_nodes, k * oversample + 1)
        dist, idx = self._index().kneighbors(queries, n_neighbors=n_req)
        out_d = np.full((Q, k), np.inf, dtype=np.float32)
        out_i = np.full((Q, k), -1, dtype=np.int64)
        keep = np.ones_like(idx, dtype=bool)
        if allowed is not None:
            keep &= allowed[idx]
        if exclude_episode is not None:
            same = self.episode_index[idx] == exclude_episode[:, None]
            if min_frame_gap > 0 and query_frame is not None:
                gap = np.abs(self.frame_index[idx] - query_frame[:, None])
                keep &= ~(same & (gap < min_frame_gap))
            else:
                keep &= ~same
        for i in range(Q):
            sel = idx[i][keep[i]][:k]
            out_i[i, : sel.size] = sel
            out_d[i, : sel.size] = dist[i][keep[i]][:k]
        return out_d, out_i

    # ------------------------------------------------------------------ #
    # candidate jumps
    # ------------------------------------------------------------------ #
    def propose_jumps(
        self,
        *,
        k: int = 8,
        window: int = 5,
        max_cost: float = 4.0,
        allow_same_episode: bool = False,
        min_frame_gap: int = 50,
        window_decay: float = 0.8,
        batch_size: int = 4096,
        verbose: bool = True,
    ) -> Jumps:
        """Propose cross-episode transitions (motion-graph style window matching).

        For a node ``u`` with recorded successor ``u+1`` we look for nodes ``v``
        in other episodes whose *upcoming* motion ``v, v+1, ..., v+W-1`` matches
        the recorded continuation ``u+1, ..., u+W``.  The cost is the decayed
        mean feature distance over that window, in units of one typical 50 Hz
        step (``self.step_unit``).  Low cost => switching from ``u`` to ``v``
        barely changes what the hand and pen are about to do.
        """
        W = int(window)
        F = self.features
        remaining = self.frames_remaining(np.arange(self.num_nodes))
        # Frames inside a hold (nothing moves) are excluded on both ends: a jump
        # between two resting frames is trivially "valid" and teaches nothing.
        moving = ~self.static_mask
        src_nodes = np.flatnonzero((remaining >= W) & moving)  # u+1..u+W exist
        target_ok = (remaining >= (W - 1)) & moving  # v..v+W-1 exist
        weights = window_decay ** np.arange(W)
        weights = weights / weights.sum()

        src_all, dst_all, cost_all, pose_all = [], [], [], []
        for start in range(0, src_nodes.size, batch_size):
            u = src_nodes[start : start + batch_size]
            q = F[u + 1]
            d, v = self.knn_filtered(
                q,
                k,
                exclude_episode=self.episode_index[u],
                min_frame_gap=(min_frame_gap if allow_same_episode else 10**9),
                query_frame=self.frame_index[u],
                allowed=target_ok,
            )
            valid = v >= 0
            uu = np.repeat(u, k).reshape(-1, k)[valid]
            vv = v[valid]
            pose_dist = d[valid] / self.step_unit
            # window cost
            cost = np.zeros(vv.shape[0], dtype=np.float64)
            for w in range(W):
                diff = F[uu + 1 + w] - F[vv + w]
                cost += weights[w] * np.linalg.norm(diff, axis=1)
            cost /= self.step_unit
            keep = cost <= max_cost
            src_all.append(uu[keep])
            dst_all.append(vv[keep])
            cost_all.append(cost[keep])
            pose_all.append(pose_dist[keep])
            if verbose and (start // batch_size) % 2 == 0:
                done = min(start + batch_size, src_nodes.size)
                print(f"  proposing jumps: {done}/{src_nodes.size} nodes", end="\r")
        if verbose:
            print()
        src = np.concatenate(src_all) if src_all else np.zeros(0, np.int64)
        dst = np.concatenate(dst_all) if dst_all else np.zeros(0, np.int64)
        cost = np.concatenate(cost_all) if cost_all else np.zeros(0)
        pose = np.concatenate(pose_all) if pose_all else np.zeros(0)
        order = np.argsort(cost, kind="stable")
        self.jumps = Jumps(
            src=src[order].astype(np.int64),
            dst=dst[order].astype(np.int64),
            cost=cost[order],
            pose_dist=pose[order],
            status=np.full(order.size, int(JumpStatus.CANDIDATE), dtype=np.int8),
            sim_score=np.full(order.size, np.nan),
        )
        return self.jumps

    def first_moving_frame(self, episode: int) -> int:
        s0, s1 = self.episode_starts[episode], self.episode_starts[episode + 1]
        moving = np.flatnonzero(~self.static_mask[s0:s1])
        return int(moving[0]) if moving.size else 0

    def propose_rest_transitions(
        self,
        *,
        blend_frames: int = 25,
        max_pen_pos_m: float = 0.06,
        max_pen_axis_rad: float = np.deg2rad(60.0),
        max_joint_vel: float = 3.0,
        max_blend_frames: int = 50,
        min_lead_frames: int = 1,
    ) -> int:
        """Propose **trick-chaining** transitions: from the moment a clip comes
        to rest (its trailing hold, pen grasped and still) into the first
        moving frame of another clip, bridged by a slow blend of the hand.

        Unlike window-matched jumps these are allowed to be far apart in
        feature space - the blend (``blend_frames`` or more, so joint speeds
        stay below ``max_joint_vel``) re-shapes the grasp while the pen is held
        - but the pen pose must be within ``max_pen_pos_m`` / ``max_pen_axis_rad``
        so the next clip can start where its reference expects the pen.
        Returns the number of transitions added (status CANDIDATE)."""
        from ..data.clips import PEN_SHAFT_AXIS_LOCAL
        from ..data.transforms import quat_apply

        dt = self.episodes[0].frame_dt
        src_list, dst_list, cost_list, pose_list, blend_list = [], [], [], [], []
        for a in range(self.num_episodes):
            u = self.node_id(a, min(int(self.active_end[a]), int(self.episode_lengths[a]) - 1))
            qu, pu = self.node_qpos(u), self.node_pen_pose(u)
            for b in range(self.num_episodes):
                if b == a:
                    continue
                v = self.node_id(b, self.first_moving_frame(b))
                if self.frames_remaining(np.array([v]))[0] < 25:
                    continue
                qv, pv = self.node_qpos(v), self.node_pen_pose(v)
                dpos = float(np.linalg.norm(pu[:3] - pv[:3]))
                au = quat_apply(pu[3:], PEN_SHAFT_AXIS_LOCAL)
                av = quat_apply(pv[3:], PEN_SHAFT_AXIS_LOCAL)
                dax = float(np.arccos(np.clip(au @ av, -1.0, 1.0)))
                if dpos > max_pen_pos_m or dax > max_pen_axis_rad:
                    continue
                dq = float(np.abs(qv - qu).max())
                blend = int(max(blend_frames, np.ceil(dq / (max_joint_vel * dt) - 1.0)))
                if blend > max_blend_frames:
                    continue
                src_list.append(u)
                dst_list.append(v)
                d = float(np.linalg.norm(self.features[u] - self.features[v]) / self.step_unit)
                cost_list.append(d)
                pose_list.append(d)
                blend_list.append(blend)
        if not src_list:
            return 0
        J = self.jumps
        self.jumps = Jumps(
            src=np.concatenate([J.src, np.asarray(src_list, np.int64)]),
            dst=np.concatenate([J.dst, np.asarray(dst_list, np.int64)]),
            cost=np.concatenate([J.cost, np.asarray(cost_list)]),
            pose_dist=np.concatenate([J.pose_dist, np.asarray(pose_list)]),
            status=np.concatenate([J.status, np.full(len(src_list), int(JumpStatus.CANDIDATE), np.int8)]),
            sim_score=np.concatenate([J.sim_score, np.full(len(src_list), np.nan)]),
            blend=np.concatenate([J.blend, np.asarray(blend_list, np.int16)]),
        )
        return len(src_list)

    def is_rest_transition(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Boolean mask: jumps that start in a hold (trick-chaining transitions)."""
        idx = np.arange(len(self.jumps)) if indices is None else np.asarray(indices)
        return self.static_mask[self.jumps.src[idx]]

    # ------------------------------------------------------------------ #
    # sparse graph views
    # ------------------------------------------------------------------ #
    def adjacency(
        self,
        *,
        jump_statuses: tuple[JumpStatus, ...] = (JumpStatus.VALIDATED,),
        intra_cost: float = 1.0,
        jump_cost_scale: float = 1.0,
        jump_cost_offset: float = 1.0,
    ) -> sparse.csr_matrix:
        """Directed weighted adjacency: intra edges cost ``intra_cost`` each
        (so path cost ~ number of frames), jumps cost
        ``offset + scale * window_cost``."""
        rows = [self.intra_src]
        cols = [self.intra_dst]
        vals = [np.full(self.intra_src.size, intra_cost)]
        if len(self.jumps):
            mask = np.isin(self.jumps.status, [int(s) for s in jump_statuses])
            rows.append(self.jumps.src[mask])
            cols.append(self.jumps.dst[mask])
            vals.append(jump_cost_offset + jump_cost_scale * self.jumps.cost[mask])
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        m = sparse.coo_matrix((v, (r, c)), shape=(self.num_nodes, self.num_nodes))
        # duplicates (same u->v proposed twice) keep the smaller cost
        m = m.tocsr()
        m.sum_duplicates()
        return m

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "graph.npz",
            features=self.features,
            episode_lengths=self.episode_lengths,
            jump_src=self.jumps.src,
            jump_dst=self.jumps.dst,
            jump_cost=self.jumps.cost,
            jump_pose_dist=self.jumps.pose_dist,
            jump_status=self.jumps.status,
            jump_sim_score=self.jumps.sim_score,
            jump_blend=self.jumps.blend,
        )
        self.featurizer.save(d / "featurizer.json")
        meta = {
            "episode_ids": [ep.episode_id for ep in self.episodes],
            "step_unit": self.step_unit,
            "num_nodes": self.num_nodes,
            "jumps": self.jumps.summary(),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path, episodes: list[Episode]) -> "MotionGraph":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        ids = [ep.episode_id for ep in episodes]
        if ids != meta["episode_ids"]:
            raise ValueError("episodes do not match the saved graph (order or content differs)")
        featurizer = FrameFeaturizer.load(d / "featurizer.json")
        with np.load(d / "graph.npz") as z:
            g = cls(episodes, featurizer, features=z["features"])
            g.jumps = Jumps(
                src=z["jump_src"].astype(np.int64),
                dst=z["jump_dst"].astype(np.int64),
                cost=z["jump_cost"],
                pose_dist=z["jump_pose_dist"],
                status=z["jump_status"].astype(np.int8),
                sim_score=z["jump_sim_score"],
                blend=z["jump_blend"].astype(np.int16) if "jump_blend" in z else None,
            )
        g.step_unit = float(meta["step_unit"])
        return g

    def transfer_statuses(self, other: "MotionGraph") -> int:
        """Copy validation outcomes for identical (src, dst) pairs from another
        graph over the same episodes (e.g. after re-proposing candidates)."""
        if [ep.episode_id for ep in other.episodes] != [ep.episode_id for ep in self.episodes]:
            raise ValueError("graphs must be built over the same episodes")
        known = {
            (int(s), int(d)): (int(st), float(sc))
            for s, d, st, sc in zip(other.jumps.src, other.jumps.dst, other.jumps.status, other.jumps.sim_score)
            if st in (int(JumpStatus.VALIDATED), int(JumpStatus.SIM_REJECTED))
        }
        n = 0
        for i, (s_, d_) in enumerate(zip(self.jumps.src, self.jumps.dst)):
            hit = known.get((int(s_), int(d_)))
            if hit is not None and self.jumps.status[i] == int(JumpStatus.CANDIDATE):
                self.jumps.status[i], self.jumps.sim_score[i] = hit
                n += 1
        return n

    def describe(self) -> str:
        lines = [
            f"MotionGraph: {self.num_nodes} nodes in {self.num_episodes} episodes, "
            f"feature dim {self.features.shape[1]}, step unit {self.step_unit:.3f}",
            f"  intra edges: {self.intra_src.size}",
            f"  jumps: {self.jumps.summary()}",
        ]
        return "\n".join(lines)
