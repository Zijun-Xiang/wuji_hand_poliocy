"""Turn every frame of every episode into one normalised high-dimensional point.

The feature vector is a concatenation of *blocks*.  Each block is z-scored with
statistics fitted on the whole dataset and then multiplied by a block weight,
so Euclidean distance in feature space is a weighted, unit-free similarity
between hand+pen states.

Default blocks (wrist-local frame, so the embedding is pose-invariant):

==============  ====  ==================================================
block           dim   content
==============  ====  ==================================================
``qpos``        20    finger joint angles
``qvel``        20    finger joint velocities (finite difference)
``pen_pos``      3    pen centre position
``pen_axis``     3    unit shaft direction (roll about the shaft ignored)
``pen_rot6d``    6    full orientation (optional, off by default)
``pen_linvel``   3    pen linear velocity
``pen_angvel``   3    pen angular velocity (pen frame)
``fingertips``  15    five distal-link positions
``links``       63    all 21 link positions (optional, off by default)
==============  ====  ==================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..data.clips import Episode
from ..data.transforms import quat_apply, quat_to_rot6d
from ..data.clips import PEN_SHAFT_AXIS_LOCAL

DEFAULT_BLOCK_WEIGHTS: dict[str, float] = {
    "qpos": 1.0,
    "qvel": 0.4,
    "pen_pos": 2.0,
    "pen_axis": 2.0,
    "pen_rot6d": 0.0,
    "pen_linvel": 0.4,
    "pen_angvel": 0.4,
    "fingertips": 1.0,
    "links": 0.0,
}

BLOCK_DIMS: dict[str, int] = {
    "qpos": 20,
    "qvel": 20,
    "pen_pos": 3,
    "pen_axis": 3,
    "pen_rot6d": 6,
    "pen_linvel": 3,
    "pen_angvel": 3,
    "fingertips": 15,
    "links": 63,
}


@dataclass
class FrameState:
    """The raw (un-normalised) quantities describing one hand+pen state.

    This is also the query type for :meth:`FrameFeaturizer.transform_state`,
    i.e. what you fill in from the simulator or the real robot."""

    qpos: np.ndarray  # (20,)
    qvel: np.ndarray  # (20,)
    pen_pose: np.ndarray  # (7,) wrist-local
    pen_linvel: np.ndarray  # (3,)
    pen_angvel: np.ndarray  # (3,)
    link_pos: np.ndarray | None = None  # (21, 3) wrist-local; None -> zeros (block weight should be 0)


def raw_blocks_from_episode(ep: Episode) -> dict[str, np.ndarray]:
    T = len(ep)
    axis = quat_apply(ep.pen_pose[:, 3:], np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (T, 3)))
    return {
        "qpos": ep.qpos,
        "qvel": ep.qvel,
        "pen_pos": ep.pen_pose[:, :3],
        "pen_axis": axis,
        "pen_rot6d": quat_to_rot6d(ep.pen_pose[:, 3:]),
        "pen_linvel": ep.pen_linvel,
        "pen_angvel": ep.pen_angvel,
        "fingertips": ep.fingertips.reshape(T, -1),
        "links": ep.link_pos.reshape(T, -1),
    }


def raw_blocks_from_state(state: FrameState) -> dict[str, np.ndarray]:
    pen_pose = np.asarray(state.pen_pose, dtype=np.float64)
    link_pos = (
        np.zeros((21, 3)) if state.link_pos is None else np.asarray(state.link_pos, dtype=np.float64)
    )
    from ..data.clips import FINGERTIP_LINK_INDICES

    return {
        "qpos": np.asarray(state.qpos, dtype=np.float64)[None],
        "qvel": np.asarray(state.qvel, dtype=np.float64)[None],
        "pen_pos": pen_pose[None, :3],
        "pen_axis": quat_apply(pen_pose[None, 3:], PEN_SHAFT_AXIS_LOCAL[None]),
        "pen_rot6d": quat_to_rot6d(pen_pose[None, 3:]),
        "pen_linvel": np.asarray(state.pen_linvel, dtype=np.float64)[None],
        "pen_angvel": np.asarray(state.pen_angvel, dtype=np.float64)[None],
        "fingertips": link_pos[list(FINGERTIP_LINK_INDICES)].reshape(1, -1),
        "links": link_pos.reshape(1, -1),
    }


@dataclass
class FrameFeaturizer:
    """Fit-once, transform-many featuriser (z-score + block weights)."""

    block_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BLOCK_WEIGHTS))
    mean: dict[str, np.ndarray] = field(default_factory=dict)
    std: dict[str, np.ndarray] = field(default_factory=dict)
    min_std: float = 1e-3

    # ------------------------------------------------------------------ #
    @property
    def active_blocks(self) -> list[str]:
        return [b for b, w in self.block_weights.items() if w > 0.0]

    @property
    def dim(self) -> int:
        return sum(BLOCK_DIMS[b] for b in self.active_blocks)

    def block_slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for b in self.active_blocks:
            out[b] = slice(start, start + BLOCK_DIMS[b])
            start += BLOCK_DIMS[b]
        return out

    # ------------------------------------------------------------------ #
    def fit(self, episodes: list[Episode]) -> "FrameFeaturizer":
        blocks = [raw_blocks_from_episode(ep) for ep in episodes]
        for b in self.active_blocks:
            x = np.concatenate([bl[b] for bl in blocks], axis=0)
            self.mean[b] = x.mean(axis=0)
            self.std[b] = np.maximum(x.std(axis=0), self.min_std)
        return self

    def _assemble(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        if not self.mean:
            raise RuntimeError("FrameFeaturizer.fit() must be called first")
        parts = []
        for b in self.active_blocks:
            z = (raw[b] - self.mean[b]) / self.std[b]
            parts.append(self.block_weights[b] * z)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform_episode(self, ep: Episode) -> np.ndarray:
        return self._assemble(raw_blocks_from_episode(ep))

    def transform_state(self, state: FrameState) -> np.ndarray:
        return self._assemble(raw_blocks_from_state(state))[0]

    def fit_transform(self, episodes: list[Episode]) -> list[np.ndarray]:
        self.fit(episodes)
        return [self.transform_episode(ep) for ep in episodes]

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "block_weights": self.block_weights,
            "min_std": self.min_std,
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "std": {k: v.tolist() for k, v in self.std.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FrameFeaturizer":
        obj = cls(block_weights=dict(payload["block_weights"]), min_std=float(payload.get("min_std", 1e-3)))
        obj.mean = {k: np.asarray(v, dtype=np.float64) for k, v in payload["mean"].items()}
        obj.std = {k: np.asarray(v, dtype=np.float64) for k, v in payload["std"].items()}
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "FrameFeaturizer":
        return cls.from_dict(json.loads(Path(path).read_text()))
