"""Matplotlib figures for the motion graph (optional dependency)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..graph.loops import Loop
from ..graph.motion_graph import JumpStatus, MotionGraph


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc[:: max(1, len(Xc) // 5000)], full_matrices=False)
    return Xc @ vt[:2].T


def plot_embedding(graph: MotionGraph, path: Path, *, max_jumps: int = 400) -> Path:
    """2-D PCA of all frames: one colour per motion group, validated jumps drawn as lines."""
    plt = _plt()
    Z = _pca2(graph.features)
    groups = [ep.group_id for ep in graph.episodes]
    uniq = sorted(set(groups))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(uniq))))
    fig, ax = plt.subplots(figsize=(9, 7))
    for gi, gname in enumerate(uniq):
        mask = np.isin(graph.episode_index, [i for i, g in enumerate(groups) if g == gname])
        ax.scatter(Z[mask, 0], Z[mask, 1], s=2, color=colors[gi], alpha=0.5, label=f"{gname} ({mask.sum()} frames)")
    J = graph.jumps
    val = J.with_status(JumpStatus.VALIDATED)
    if val.size:
        sel = val[np.linspace(0, val.size - 1, min(max_jumps, val.size)).astype(int)]
        for s, d in zip(J.src[sel], J.dst[sel]):
            ax.plot([Z[s, 0], Z[d, 0]], [Z[s, 1], Z[d, 1]], color="k", alpha=0.15, lw=0.6)
        ax.plot([], [], color="k", alpha=0.5, lw=0.8, label=f"validated jumps ({val.size}, {sel.size} drawn)")
    ax.set_title("Every frame as a point (PCA of the 67-D embedding); lines = sim-validated jumps")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=6, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_jump_matrix(graph: MotionGraph, path: Path) -> Path:
    """Episode x episode count of validated jumps."""
    plt = _plt()
    n = graph.num_episodes
    M = np.zeros((n, n), int)
    J = graph.jumps
    val = J.with_status(JumpStatus.VALIDATED)
    for s, d in zip(J.src[val], J.dst[val]):
        M[graph.episode_index[s], graph.episode_index[d]] += 1
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.log1p(M), cmap="viridis")
    labels = [ep.episode_id.replace("_", " ") for ep in graph.episodes]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("to episode")
    ax.set_ylabel("from episode")
    ax.set_title(f"Validated jumps between episodes (log scale, total {val.size})")
    fig.colorbar(im, ax=ax, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_loop(graph: MotionGraph, loop: Loop, path: Path) -> Path:
    """Timeline of one loop: which episode/frame is used when, plus pen axis angles."""
    plt = _plt()
    from ..data.clips import PEN_SHAFT_AXIS_LOCAL
    from ..data.transforms import quat_apply

    nodes = loop.nodes
    ep_idx = graph.episode_index[nodes]
    frames = graph.frame_index[nodes]
    t = np.arange(nodes.size) * graph.episodes[0].frame_dt
    pen_q = np.stack([graph.node_pen_pose(n)[3:] for n in nodes])
    axis = quat_apply(pen_q, np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (nodes.size, 3)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].scatter(t, frames, c=ep_idx, cmap="tab20", s=6)
    for a, b, tt in zip(nodes[:-1], nodes[1:], t[1:]):
        if not (b == a + 1 and graph.episode_index[a] == graph.episode_index[b]):
            axes[0].axvline(tt, color="r", lw=0.8, alpha=0.7)
    axes[0].set_ylabel("source frame index")
    axes[0].set_title(loop.describe(graph), fontsize=9)
    axes[1].plot(t, axis[:, 0], label="shaft x")
    axes[1].plot(t, axis[:, 1], label="shaft y")
    axes[1].plot(t, axis[:, 2], label="shaft z")
    axes[1].set_ylabel("pen shaft direction (wrist frame)")
    axes[1].set_xlabel("time in loop (s)   red lines = jumps between episodes")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_rollout(result, path: Path, title: str = "") -> Path:
    plt = _plt()
    t = np.arange(result.n_steps) * 0.02
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, 100 * result.pen_pos_err)
    axes[0].axhline(5.0, color="r", ls="--", lw=0.8)
    axes[0].set_ylabel("pen pos err (cm)")
    axes[1].plot(t, np.rad2deg(result.pen_axis_err))
    axes[1].axhline(60, color="r", ls="--", lw=0.8)
    axes[1].set_ylabel("shaft dir err (deg)")
    axes[2].plot(t, result.joint_err)
    axes[2].set_ylabel("max joint err (rad)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title(title or result.summary(), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
