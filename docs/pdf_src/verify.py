"""Numerically check every formula that appears in the PDF against the code."""
import json, numpy as np
from pathlib import Path
from penspin_graph.data import load_clip_groups
from penspin_graph.graph import MotionGraph, JumpStatus
from penspin_graph.features.featurizer import raw_blocks_from_episode, BLOCK_DIMS
from penspin_graph.graph.kinematics import KinematicLimits, check_jumps
from penspin_graph.data.clips import PEN_SHAFT_AXIS_LOCAL
from penspin_graph.data.transforms import quat_apply, quat_to_rotmat, quat_mul, quat_inv
REPO = Path("/home/claude/work/penspin-graph")
eps = load_clip_groups(REPO / "data/wuji_pen_spin_clips")
g = MotionGraph.load(REPO / "runs/full3/graph", eps)
fz = g.featurizer
ok = lambda name, cond: print(("PASS " if cond else "FAIL ") + name)

# Stage 1: wrist-frame position p = R^T (p_w - t_w)
ep = eps[0]; t = 77
R = quat_to_rotmat(ep.wrist_pose_w[t, 3:])
p = R.T @ (ep.pen_pose_w[t, :3] - ep.wrist_pose_w[t, :3])
ok("R1 wrist-frame position", np.allclose(p, ep.pen_pose[t, :3]))
rho = quat_mul(quat_inv(ep.wrist_pose_w[t, 3:]), ep.pen_pose_w[t, 3:])
ok("R2 wrist-frame quaternion", np.allclose(np.abs(rho @ ep.pen_pose[t, 3:]), 1.0))
ok("R5 finite difference", np.allclose(ep.qvel[t], (ep.qpos[t + 1] - ep.qpos[t]) / 0.02))

# Stage 2: f = w * (x - mu) / sigma, dims and distance decomposition
dims = [BLOCK_DIMS[b] for b in fz.active_blocks]
print("blocks", fz.active_blocks, dims, "sum", sum(dims))
raw = raw_blocks_from_episode(ep)
f_manual = np.concatenate([fz.block_weights[b] * (raw[b][t] - fz.mean[b]) / fz.std[b] for b in fz.active_blocks])
ok("feature formula", np.allclose(f_manual, g.features[g.node_id(0, t)], atol=1e-5))
a, b2 = g.node_id(0, t), g.node_id(3, 50)
d = np.linalg.norm(g.features[a] - g.features[b2]); sl = fz.block_slices()
parts = {k: float(np.sum((g.features[a][s] - g.features[b2][s]) ** 2)) for k, s in sl.items()}
ok("distance^2 = sum of block parts", np.isclose(d ** 2, sum(parts.values()), rtol=1e-5))
print("  example block parts of d^2:", {k: round(v, 2) for k, v in parts.items()})
step = np.linalg.norm(g.features[g.intra_dst] - g.features[g.intra_src], axis=1)
ok("step unit = median one-step distance", np.isclose(np.median(step), g.step_unit))
print("  step unit", round(g.step_unit, 4), "one-step distance pct10/50/90:", np.percentile(step, [10, 50, 90]).round(3))
print("  joint-angle std (rad) range", fz.std["qpos"].min().round(3), fz.std["qpos"].max().round(3), " pen-pos std (cm)", (100 * fz.std["pen_pos"]).round(2))

# Stage 3: window weights and cost
lam = 0.8 ** np.arange(5); lam /= lam.sum(); print("lambda_j", lam.round(4), "sum", lam.sum())
J = g.jumps
val = J.with_status(JumpStatus.VALIDATED)
non_rest = val[~g.static_mask[J.src[val]]]
i = int(non_rest[1234]); u, v = int(J.src[i]), int(J.dst[i])
C = sum(lam[j] * np.linalg.norm(g.features[u + 1 + j] - g.features[v + j]) for j in range(5)) / g.step_unit
ok(f"window cost recomputed ({C:.4f} vs stored {J.cost[i]:.4f})", np.isclose(C, J.cost[i], rtol=1e-4))
ex = np.array([2, 2, 3, 3, 4.0]); print("  example cost for distances 2,2,3,3,4:", round(float(lam @ ex), 3))
same = g.episode_index[J.src[val]] == g.episode_index[J.dst[val]]
print("  validated jumps: same-episode", int(same.sum()), "cross-episode", int((~same).sum()))
cand_all = len(J)
print("  total proposals", cand_all, "rest transitions", int(g.static_mask[J.src].sum()))

# Stage 5: blend length and tolerances
lim = KinematicLimits.from_dataset(g)
print("omega_max", round(lim.max_joint_vel, 4), "P99.9 |qdot|", round(lim.max_joint_vel / 1.5, 4))
dq = 0.5; bneed = int(np.ceil(dq / (lim.max_joint_vel * 0.02) - 1)); bb = int(np.clip(max(bneed, 2), 2, 10))
print("  example dq=0.5 rad -> b_need", bneed, "b", bb, "speed", round(dq / ((bb + 1) * 0.02), 3))
idx = J.with_status(JumpStatus.VALIDATED)[:5000]; idx = idx[~g.static_mask[J.src[idx]]]
rep = check_jumps(g, lim, indices=idx)
ok("blend formula matches stored blend", np.array_equal(rep.blend, J.blend[idx]))
for bl in [2, 4, 6, 10]:
    print(f"  tolerance at b={bl}: pos {min(0.02 * (1 + 0.5 * (bl - 2)), 0.06):.3f} m, axis {min(20 * (1 + 0.5 * (bl - 2)), 45):.0f} deg")
print("  blend histogram (validated, non-rest):", np.bincount(J.blend[non_rest]).tolist())

# Stage 4: rest transitions
rest = np.flatnonzero(g.static_mask[J.src])
print("  rest transitions: proposed", rest.size, "validated", int((J.status[rest] == int(JumpStatus.VALIDATED)).sum()),
      "blend min/median/max", int(J.blend[rest].min()), int(np.median(J.blend[rest])), int(J.blend[rest].max()))
for dqr in [1.2, 2.4]:
    print(f"  rest blend for dq={dqr}: max(25, ceil({dqr}/(3*0.02)-1)) =", max(25, int(np.ceil(dqr / 0.06 - 1))))

# Stage 4 motivation: periodicity inside clips (moving frames, >= 1 s apart)
best = []
for e in range(len(eps)):
    s0, s1 = g.episode_starts[e], g.episode_starts[e + 1]
    mv = np.flatnonzero(~g.static_mask[s0:s1]); X = g.features[s0:s1][mv]
    if len(X) < 120: continue
    D = np.linalg.norm(X[:, None] - X[None], axis=-1) / g.step_unit
    D[np.abs(mv[:, None] - mv[None]) < 50] = np.inf
    best.append(np.median(D.min(1)))
print("  median nearest far (>=1 s) self-match per clip, step units: median over clips", round(float(np.median(best)), 2), "range", round(min(best), 2), round(max(best), 2))

# Stage 6: residual + EMA and obs size
a1 = 2.0; r1 = np.clip(0.5 * a1, -0.6, 0.6); e1 = 0.5 * r1 + 0.5 * 0.0; e2 = 0.5 * r1 + 0.5 * e1
print("  EMA example: r", r1, "rbar1", e1, "rbar2", e2, "steady", r1, " obs size", 20 + 6 + 63 + 20 + 20 + 6 + 20)
x = 0.12; print("  tanh bound 0.15*tanh(x/0.15) at x=0.12:", round(0.15 * np.tanh(x / 0.15), 4), "x=0.03:", round(0.15 * np.tanh(0.03 / 0.15), 4))

# Stage 8: edge cost and rotation metric of the best loop
L = json.load(open(REPO / "runs/full3/loops.json"))[0]
nodes = np.array(L["nodes"]); n = quat_apply(np.stack([g.node_pen_pose(k)[3:] for k in nodes]), np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (len(nodes), 3)))
theta = np.rad2deg(np.arccos(np.clip(np.sum(n * np.roll(n, -1, 0), 1), -1, 1)).sum())
print("  best loop: nodes", len(nodes), "shaft rotation per lap", round(theta), "deg; jump edge costs 25+10C:", [round(25 + 10 * c, 1) for c in L["jump_costs"]])
print("  jump C=2 costs like", 25 + 10 * 2, "recorded frames")
scc = np.load(REPO / "runs/full3/scc_largest_nodes.npy"); print("  SCC", scc.size, "of", g.num_nodes, "=", round(100 * scc.size / g.num_nodes, 1), "%")
