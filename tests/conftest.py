from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from penspin_graph.data import Episode, episode_from_arrays, load_clip_groups
from penspin_graph.data.transforms import quat_normalize
from penspin_graph.features import FrameFeaturizer
from penspin_graph.graph import MotionGraph

REPO = Path(__file__).resolve().parents[1]
CLIPS = REPO / "data" / "wuji_pen_spin_clips"


def synthetic_episode(seed: int, frames: int = 120, phase: float = 0.0) -> Episode:
    """A smooth fake pen-spin: joints and pen follow sinusoids with a phase offset.

    Episodes with different seeds but the same phase are near-duplicates, so
    the graph should find cheap jumps between them."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames) * 0.02
    base = np.sin(2 * np.pi * 0.5 * t[:, None] + phase + np.linspace(0, 3, 20)[None]) * 0.6 + 0.4
    qpos = base + 0.002 * rng.standard_normal((frames, 20))
    wrist = np.tile(np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]), (frames, 1))
    ang = 2 * np.pi * 0.5 * t + phase
    quat = quat_normalize(np.stack([np.cos(ang / 2), np.zeros_like(ang), np.sin(ang / 2), np.zeros_like(ang)], 1))
    pen_local = np.concatenate([np.stack([0.02 * np.cos(ang), 0.06 + 0 * ang, -0.09 + 0.01 * np.sin(ang)], 1), quat], 1)
    # wrist frame is identity rotation at (0.1, 0.2, 0.3)
    pen_w = pen_local.copy()
    pen_w[:, :3] += wrist[:, :3]
    links = 0.05 * np.random.default_rng(123).standard_normal((1, 21, 3)) + np.array([0, 0, -0.08])
    links = np.repeat(links, frames, axis=0) + 0.002 * np.sin(ang)[:, None, None]
    links[:, 0] = 0.0
    link_w = links + wrist[:, None, :3]
    return episode_from_arrays(
        episode_id=f"synthetic/clip_{seed:03d}", group_id="synthetic", clip_id=f"clip_{seed:03d}",
        reference_qpos=qpos, wrist_pose_w=wrist, object_pose_w=pen_w, link_pos_w=link_w,
    )


@pytest.fixture(scope="session")
def synthetic_episodes() -> list[Episode]:
    return [synthetic_episode(0), synthetic_episode(1), synthetic_episode(2, phase=np.pi / 3), synthetic_episode(3, phase=np.pi)]


@pytest.fixture(scope="session")
def synthetic_graph(synthetic_episodes) -> MotionGraph:
    g = MotionGraph(synthetic_episodes, FrameFeaturizer().fit(synthetic_episodes))
    g.propose_jumps(k=4, window=3, max_cost=6.0, verbose=False)
    return g


@pytest.fixture(scope="session")
def real_episodes() -> list[Episode]:
    if not CLIPS.is_dir():
        pytest.skip("Wuji clips not present")
    return load_clip_groups(CLIPS, groups=["01_single_axis"])
