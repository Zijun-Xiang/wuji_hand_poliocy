#!/usr/bin/env python3
"""Head-to-head comparison: Wuji's original clip tracking vs the stitched motion graph.

Three settings, all using Wuji's released ONNX tracking policy as the low-level
controller, all in the same MuJoCo scene:

  A  original      : replay each recorded clip from its first frame (what
                     wuji-mjlab deploys).  The motion ends when the clip ends.
  B  unstitched    : from a random moving state, find the nearest recorded
                     frame and follow its clip (re-matching allowed) but with
                     NO jumps between/inside clips  -> "the idea without the graph".
  C  stitched (ours): same as B, plus the validated jumps and escape planning.
  D  loop seeking (ours, the full idea): from the nearest frame, plan through
                     validated jumps into the best verified loop and circle it.

B and C start from the same random states.  Metrics over a long horizon:
survival, pen drops, shaft rotation (pen spinning, roll about the shaft
ignored), and rotation in the last third of the horizon (is it *still*
spinning?).  Usage:  python scripts/compare_baselines.py runs/full3 [--horizon 1500] [--n-starts 20]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from penspin_graph.data import load_clip_groups
from penspin_graph.data.clips import PEN_SHAFT_AXIS_LOCAL
from penspin_graph.data.transforms import quat_apply
from penspin_graph.graph import MotionGraph
from penspin_graph.graph.planner import MotionMatchingController, loop_to_trajectory
from penspin_graph.policy import GraphTrackingPolicy
from penspin_graph.sim import OnnxTrackingController, PenSpinScene, rollout, track_episode
from penspin_graph.sim.controller import LiveState
from penspin_graph.sim.rollout import frame_state_from_episode

REPO = Path(__file__).resolve().parents[1]
DT = 0.02


def shaft(scene: PenSpinScene) -> np.ndarray:
    return quat_apply(scene.pen_pose_local()[3:], PEN_SHAFT_AXIS_LOCAL)


def run_closed_loop(graph, policy, scene, start_node, horizon, drop_m=0.12):
    e, t = graph.node_episode_frame(start_node)
    st = frame_state_from_episode(graph.episodes[e], t)
    scene.set_state(st.qpos, st.qvel, st.pen_pose, st.pen_linvel, st.pen_angvel)
    policy.reset()
    z0 = scene.pen_pose_world()[2]
    prev = shaft(scene)
    dtheta = np.zeros(horizon)
    alive = 0
    for k in range(horizon):
        live = LiveState(scene.hand_qpos(), scene.hand_qvel(), scene.pen_pose_local(), scene.link_pos_local())
        scene.control_step(policy.act(None, live))
        pw = scene.pen_pose_world()
        if not np.isfinite(pw).all() or pw[2] < z0 - drop_m:
            break
        cur = shaft(scene)
        dtheta[k] = np.arccos(np.clip(prev @ cur, -1, 1))
        prev = cur
        alive = k + 1
    jumps = int(getattr(policy.matcher, "jumps_taken", 0))
    return {"start": graph.node_label(start_node), "survived_s": alive * DT, "dropped": alive < horizon,
            "rotation_deg": float(np.rad2deg(dtheta.sum())),
            "rotation_last_third_deg": float(np.rad2deg(dtheta[2 * horizon // 3:].sum())),
            "spinning_s": float(np.sum(np.rad2deg(dtheta) / DT > 90.0) * DT),  # shaft turning faster than 90 deg/s
            "jumps": jumps}


def summarize(rows):
    a = lambda k: np.array([r[k] for r in rows], float)
    return {"n": len(rows), "survived_mean_s": a("survived_s").mean(), "drop_rate": a("dropped").mean(),
            "rotation_mean_deg": a("rotation_deg").mean(), "rotation_last_third_mean_deg": a("rotation_last_third_deg").mean(),
            "spinning_mean_s": a("spinning_s").mean(), "jumps_mean": a("jumps").mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--n-starts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", choices=["A", "B", "C", "D", "L"], default=None)
    ap.add_argument("--loop-repeats", type=int, default=6)
    args = ap.parse_args()
    run = Path(args.run)
    eps = load_clip_groups(REPO / "data" / "wuji_pen_spin_clips")
    g = MotionGraph.load(run / "graph", eps)
    scene = PenSpinScene()
    rng = np.random.default_rng(args.seed)
    moving = np.flatnonzero(~g.static_mask & (g.frames_remaining(np.arange(g.num_nodes)) > 0))
    starts = rng.choice(moving, size=args.n_starts, replace=False)
    out = {}
    t0 = time.time()

    if args.only in (None, "A"):
        rows = []
        for e, ep in enumerate(eps):
            res = track_episode(scene, OnnxTrackingController(), ep, start=0)
            q = ep.pen_pose[:, 3:]
            ax = quat_apply(q, np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (len(ep), 3)))
            ref_rot = float(np.rad2deg(np.arccos(np.clip(np.sum(ax[1:] * ax[:-1], 1), -1, 1)).sum()))
            rows.append({"clip": ep.episode_id, "completed": res.completed, "tracked_s": res.n_steps * DT,
                         "clip_s": len(ep) * DT, "active_s": float(g.active_end[e] * DT), "reference_rotation_deg": ref_rot,
                         "pen_err_cm": 100 * res.mean_pos_err()})
            print(f"A {ep.episode_id}: {'ok' if res.completed else 'FAIL'} tracked {res.n_steps * DT:.1f}/{len(ep) * DT:.1f}s active {g.active_end[e] * DT:.1f}s rot {ref_rot:.0f}deg")
        act = np.array([r["active_s"] for r in rows])
        out["A_original_clip_replay"] = {
            "clips": rows, "completed": int(sum(r["completed"] for r in rows)), "n": len(rows),
            "active_s_mean": act.mean(), "active_s_max": act.max(),
            "longest_spinning_clip": rows[int(np.argmax(act))]["clip"],
            "rotation_per_clip_mean_deg": float(np.mean([r["reference_rotation_deg"] for r in rows])),
            "rotation_per_clip_max_deg": float(np.max([r["reference_rotation_deg"] for r in rows])),
        }

    best_loop = np.array(json.loads((run / "loops.json").read_text())[0]["nodes"], dtype=np.int64)
    for key, name, statuses in [("B", "B_unstitched_matching", ()), ("C", "C_stitched_graph", None), ("D", "D_stitched_loop_seeking", "loop")]:
        if args.only not in (None, key):
            continue
        rows = []
        for i, s in enumerate(starts):
            if statuses == "loop":
                restrict = np.zeros(g.num_nodes, bool)
                restrict[np.load(run / "scc_largest_nodes.npy")] = True
                matcher = MotionMatchingController(g, restrict_to=restrict, goal_loop=best_loop)
            elif statuses is None:
                restrict = np.zeros(g.num_nodes, bool)
                restrict[np.load(run / "scc_largest_nodes.npy")] = True
                matcher = MotionMatchingController(g, restrict_to=restrict)
            else:
                matcher = MotionMatchingController(g, jump_statuses=statuses)
            pol = GraphTrackingPolicy(g, OnnxTrackingController(), matcher=matcher)
            r = run_closed_loop(g, pol, scene, int(s), args.horizon)
            rows.append(r)
            print(f"{key} {i + 1}/{len(starts)} {r['start']}: survived {r['survived_s']:.1f}s rot {r['rotation_deg']:.0f} (last third {r['rotation_last_third_deg']:.0f}) spinning {r['spinning_s']:.1f}s jumps {r['jumps']}")
        out[name] = {"episodes": rows, **summarize(rows)}

    if args.only in (None, "L"):
        loops = json.loads((run / "loops.json").read_text())
        from penspin_graph.graph.loops import Loop

        L0 = loops[0]
        loop = Loop(np.array(L0["nodes"]), [tuple(x) for x in L0["jump_edges"]], L0["jump_costs"], L0["n_frames"], L0["duration_s"], np.deg2rad(L0["pen_rotation_deg"]), L0["episodes"])
        traj = loop_to_trajectory(g, loop, repeats=args.loop_repeats)
        e, t = g.node_episode_frame(int(loop.nodes[0]))
        res = rollout(scene, OnnxTrackingController(), traj, frame_state_from_episode(g.episodes[e], t))
        out["L_best_loop_endurance"] = {"repeats": args.loop_repeats, "reference_s": len(traj) * DT, "survived_s": res.n_steps * DT,
                                        "completed": res.completed, "summary": res.summary(),
                                        "reference_rotation_deg": L0["pen_rotation_deg"] * args.loop_repeats}
        print("L", out["L_best_loop_endurance"])

    out["horizon_s"] = args.horizon * DT
    out["seconds"] = time.time() - t0
    suffix = "" if args.only is None else f"_{args.only}"
    (run / f"comparison{suffix}.json").write_text(json.dumps(out, indent=2, default=float))
    print("wrote", run / f"comparison{suffix}.json")


if __name__ == "__main__":
    main()
