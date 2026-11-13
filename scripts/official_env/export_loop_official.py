"""Write motion-graph loops as an official wuji-mjlab clip group.

The clip group goes into the official repo's own clip folder, so Wuji's
loader, sha256 check, reset and command code read it exactly like the
released clips:

  <wuji-mjlab>/src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/90_graph_loop/
      index.json, clip_001.npz, ...
  <wuji-mjlab>/src/wuji_mjlab/tasks/pen_spin/data/recipe/90_graph_loop.json

Each loop is rotated to start at a node that is frame 0 of a recorded clip
(so play mode's 1 s first-frame hold and reset are the official ones), then
repeated.  A side file stores the source motion group of every frame, so the
evaluator can run the matching released policy per frame (as the graph
controller does).

Usage: python3 export_loop_official.py <run_dir> <wuji_mjlab_root> [--loops 0] [--laps 6 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from penspin_graph.data import load_clip_groups
from penspin_graph.graph import MotionGraph
from penspin_graph.graph.planner import path_to_trajectory
from penspin_graph.policy.export_clips import trajectory_to_clip_arrays

def rotated_to_clip_start(g: MotionGraph, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a cyclic node list to start at its earliest frame within a recorded clip.

    Returns (prefix, loop): ``prefix`` is the recorded clip from its frame 0 up
    to that node (empty when the loop already passes through a frame 0), so
    every exported clip starts at a recorded clip's first frame, exactly like
    the official clips."""
    frames = [g.node_episode_frame(int(n))[1] for n in nodes]
    i = int(np.argmin(frames))
    loop = np.concatenate([nodes[i:], nodes[:i]])
    start = int(loop[0])
    prefix = np.arange(start - frames[i], start, dtype=np.int64)  # same episode, frames 0 .. t-1
    return prefix, loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("wuji_root")
    ap.add_argument("--loops", type=int, nargs="+", default=[0])
    ap.add_argument("--laps", type=int, nargs="+", default=[6, 20])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out-meta", default="loop_clips.json")
    ap.add_argument("--group-id", default="90_graph_loop")
    args = ap.parse_args()

    run = Path(args.run)
    eps = load_clip_groups(Path(args.repo) / "data" / "wuji_pen_spin_clips")
    g = MotionGraph.load(run / "graph", eps)
    loops = json.loads((run / "loops.json").read_text())

    task = Path(args.wuji_root) / "src" / "wuji_mjlab" / "tasks" / "pen_spin" / "data"
    GROUP_ID = args.group_id
    final_dir = task / "rawdata" / "clips" / GROUP_ID
    # write next to (not inside) the clip root, then move the finished folder in
    # one rename, so a loader scanning the clip root never sees a partial group
    gdir = task / "rawdata" / f".staging_{GROUP_ID}"
    shutil.rmtree(gdir, ignore_errors=True)
    gdir.mkdir(parents=True)

    entries, meta = [], []
    k = 0
    for li in args.loops:
        prefix, nodes = rotated_to_clip_start(g, np.asarray(loops[li]["nodes"], dtype=np.int64))
        e0, _ = g.node_episode_frame(int(nodes[0]))
        lap = path_to_trajectory(g, nodes, close_loop=True)
        pre = path_to_trajectory(g, prefix) if len(prefix) else None
        for laps in args.laps:
            k += 1
            clip_id = f"clip_{k:03d}"
            from penspin_graph.graph.planner import ReferenceTrajectory

            def reps(field):
                parts = ([getattr(pre, field)] if pre is not None else []) + [getattr(lap, field)] * laps
                return np.concatenate(parts, axis=0)

            traj = ReferenceTrajectory(
                qpos=reps("qpos"), pen_pose=reps("pen_pose"), nodes=reps("nodes"),
                frame_dt=lap.frame_dt, link_pos=reps("link_pos"),
                group_ids=(list(pre.group_ids) if pre is not None else []) + list(lap.group_ids) * laps,
            )
            arrays = trajectory_to_clip_arrays(g, traj)
            path = gdir / f"{clip_id}.npz"
            np.savez(path, **arrays)
            # per-frame source group, for deploy/pen_policy_graph.py (kept out of the npz
            # so the clip stays byte-for-byte in the official schema)
            (gdir / f"{clip_id}.groups.json").write_text(json.dumps({"frame_groups": traj.group_ids}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({"clip_id": clip_id, "file": path.name, "sha256": digest})
            meta.append({
                "clip_id": clip_id, "loop_index": li, "laps": laps, "frames": len(traj),
                "lap_frames": len(lap), "lap_s": len(lap) * lap.frame_dt, "duration_s": len(traj) * lap.frame_dt,
                "starts_at": f"{g.episodes[e0].episode_id} frame 0",
                "prefix_frames": int(len(prefix)), "loop_rotation_deg": loops[li]["pen_rotation_deg"],
                "source_clips": sorted({g.episodes[i].episode_id for i in loops[li]["episodes"]}),
                "frame_groups": traj.group_ids,
            })
            print(f"{clip_id}: loop {li} x {laps} laps = {len(traj)} frames ({len(traj) * lap.frame_dt:.1f} s), "
                  f"starts at {g.episodes[e0].episode_id} frame 0, prefix {len(prefix)} frames")

    index = {
        "schema": "wuji_pen_spin_clips:1", "group_id": GROUP_ID, "group_name": "motion-graph loops",
        "frame_dt": float(lap.frame_dt), "clip_count": len(entries), "clips": entries,
    }
    (gdir / "index.json").write_text(json.dumps(index, indent=2))
    shutil.rmtree(final_dir, ignore_errors=True)
    os.replace(gdir, final_dir)
    gdir = final_dir
    (task / "recipe" / f"{GROUP_ID}.json").write_text(json.dumps({"schema": "wuji_pen_spin_recipe:1", "groups": [GROUP_ID]}, indent=2))
    Path(args.out_meta).write_text(json.dumps(meta))
    print("wrote", gdir, "and", task / "recipe" / f"{GROUP_ID}.json")


if __name__ == "__main__":
    main()
