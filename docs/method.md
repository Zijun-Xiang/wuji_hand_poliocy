# Method notes

This document spells out what the code does, the decisions behind it, and
what is still open. Terms are given in English and 中文 where that helps.

## 1. The data

Wuji released 40 pen-spinning clips for the right Wuji Hand 2 (20 joints,
four per finger). They were captured from a human hand with MediaPipe and
retargeted to the robot, so every frame has

* `reference_qpos` (20) - target joint angles,
* `object_pose_w` (7) - pen pose (position + quaternion `wxyz`),
* `wrist_pose_w` (7) - wrist pose during capture,
* `link_pos_w` (21 x 3) - positions of palm + 20 finger links,

sampled at 50 Hz. Five motion groups: single-axis rotation (10 clips),
continuous single-axis (4), continuous multi-axis (9), stable-grasp rotation
(10), two rotations (7). 25,885 frames in total (~8.6 minutes).

Important property: **the references are not physically perfect**. Replaying
them open loop with the hand's position controller drops the pen within
~0.2 s (see `tests/test_sim_and_policy.py`). Wuji's PPO policies learn
*residual corrections* on top of the reference and keep the pen in hand. We
therefore never judge a transition by naive replay; we judge it with the
tracking policy in the loop.

## 2. Frame embedding (特征嵌入)

`FrameFeaturizer` maps a frame to a 67-D vector made of blocks, each z-scored
over the dataset and multiplied by a weight:

| block | dim | weight | note |
|---|---|---|---|
| joint angles | 20 | 1.0 | |
| joint velocities | 20 | 0.4 | finite difference |
| pen position (wrist frame) | 3 | 2.0 | |
| pen shaft direction | 3 | 2.0 | roll about the shaft is ignored, like Wuji's policy |
| pen linear velocity | 3 | 0.4 | |
| pen angular velocity | 3 | 0.4 | |
| fingertip positions | 15 | 1.0 | distal link of each finger |

All geometry is expressed in the **wrist frame** so the embedding does not care
where the hand was during capture, and the same vector can be computed from
the simulator or the real robot (`FrameState`). Weights are a knob
(`penspin-graph build --weights pen_pos=3,qvel=0.2`); the natural unit for
distances is the median one-step displacement (`graph.step_unit`), so a
distance of 1 means "about one 50 Hz step of motion".

## 3. Motion graph and candidate jumps (运动图 / 候选跳转)

Nodes are frames. Intra-episode edges `t -> t+1` are the recorded motion. A
**jump** `u -> v` (u and v in different episodes) means "after frame u,
continue with v's episode". Candidates are proposed by window matching
(Kovar et al. 2002 style): for each u, the kNN of `f(u+1)` among other
episodes are tested, and the cost is the decayed mean distance between the
recorded continuation `u+1 .. u+W` and the new continuation `v .. v+W-1`
(`W = 5`, i.e. 0.1 s). Low cost means the two demos are about to do the same
thing. Candidates above `max_cost` (default 4 step units) are discarded.

Two details matter in practice:

* **Holds are excluded.** Every clip ends with a ~2 s hold (the hand and pen
  do not move). Resting frames are identical to each other, so without this
  rule the cheapest "jumps" are hold -> hold inside one clip (cost 0, trivially
  executable, useless). Frames with |joint speed| < 0.2 rad/s and |pen angular
  velocity| < 0.3 rad/s are never jump endpoints.
* **Same-episode jumps are allowed** when the frames are >= 50 apart (1 s).
  A jump *back* in time inside a continuous-rotation clip is the most natural
  loop: "the pen has come around, repeat the last turn".

A **kinematic filter** removes jumps the real hand cannot do: joint limits,
max joint speed over the blend (default: dataset 99.9th percentile x 1.5
~ 5.8 rad/s), and pen continuity (the pen at `u+1` and at `v` must agree
within 2 cm / 20 deg because the pen cannot teleport). The blend length is
chosen per jump (2-10 frames) so that the joint speed limit holds; the pen
tolerances grow with the blend length up to 6 cm / 45 deg.

On the 40 clips with `k=12`, `max_cost=6`: 101,934 candidates, 5,403
removed kinematically; the cheapest ~40k were validated in simulation (see
`runs/full2/report.md`).

## 4. Simulation validation (仿真验证)

`PenSpinScene` builds the hand + a 250 mm pen in MuJoCo (fixed wrist pose,
0.005 s physics, 4 substeps per 50 Hz tick, implicitfast integrator,
condim 4, 30 g pen) - the same essentials as wuji-mjlab, without mjlab.
Wuji's released policies run unchanged in this scene (`OnnxTrackingController`
reproduces the 155-D observation and the residual/EMA action post-processing of
`deploy/pen_spin`), which is itself evidence the scene is faithful: the
policy keeps the pen within ~1 cm of the reference for seconds, while PD
replay fails in ~8 ticks.

For every candidate jump, `TransitionValidator` runs two rollouts from the
identical recorded state at `u`:

* **stitched**: `u -> 2 blended frames -> v, v+1, ... v+H-1`,
* **baseline**: `u, u+1, ..., u+H+2` (what the data did).

A jump is validated when the stitched rollout never trips wuji-mjlab's
termination thresholds (pen position error > 5 cm or shaft angle error >
60 deg), ends within 3 cm / 30 deg of its reference, and its mean pen error
is at most 1.5x the baseline's. The relative criterion is what makes the test
fair for imperfect references. Validation costs ~60 ms per jump on one CPU
core (`--workers N` forks a pool).

## 5. Loops, SCC and planning (环路 / 强连通分量 / 规划)

Each episode is a chain, so any loop needs at least one jump. For each
validated jump `u -> v` we run Dijkstra from `v` back to `u` on the graph
(1 per recorded frame, `25 + 10 x cost` per jump) and close the loop. Loops are
filtered by length and by accumulated pen rotation, and ranked by rotation per
second. The strongly connected components of the stitched graph are reported
too: the largest SCC is the **spinnable core** - the frames from which the
hand can keep spinning forever using only validated moves.

Planning from an arbitrary state: embed it, take the nearest node (optionally
restricted to the core), Dijkstra to a goal node or into a loop, and emit a
50 Hz reference with short linear/slerp blends at jumps
(`GraphPlanner`, `path_to_trajectory`). `MotionMatchingController` is the
reactive variant: follow the current frame's successor, re-query the nearest
node every few ticks or when the state drifts, switch only if a clearly closer
frame exists. `GraphTrackingPolicy` = that controller + Wuji's tracker, which
is the original idea in closed loop with no external reference signal.

## 6. From graph to policy (从图到策略)

Two routes, both implemented:

1. **Stitched clips -> PPO** (`penspin-graph export`). Loops and random walks on
   the validated graph are written as a new clip group in the
   `wuji_pen_spin_clips:1` format with SHA-256 digests and a recipe. Training
   wuji-mjlab's `WujiHand2_PenSpin` on that recipe yields a tracking policy
   whose references are longer, more varied, and - by construction - already
   executable. Deploy exactly as Wuji's own clips.
2. **Behaviour cloning** (`penspin-graph train-bc`). Random walks are executed
   in sim with the graph + tracker; `(joint angles, joint velocities, pen axis
   points) -> joint target residual` pairs are recorded and an MLP is fitted.
   The BC policy needs no reference at run time. Expect it to be weaker than
   PPO (covariate shift, no recovery data); DAgger-style relabelling with the
   graph policy as the expert is the obvious upgrade.

## 7. What happened on the real data (and why the hold trick was needed)

Three validation passes, each reusing the previous outcomes:

| pass | change | jumps tested / passed | largest SCC |
|---|---|---|---|
| 1 | cross-episode window matching only, k=8, cost <= 4 | 9,320 / 8,928 | 230 frames |
| 2 | + same-episode jumps (>= 50 frames apart), k=12, cost <= 6, holds excluded | 36,000 + 24,000 / 58,803 | 670 frames |
| 3 | + hold -> clip-start transitions (25-50 frame blends) | 482 / 99 | **15,215 frames** |

Passes 1-2 produced only tiny loops (median 3 frames, < 20 deg of rotation),
although 96% of jumps were executable. The reason is simple once measured:
the 67-D state is *not* periodic within a clip. Even for continuous-rotation
clips the nearest earlier frame (>= 1 s apart) is typically > 8 step units
away, because fingers re-grasp differently every revolution and the human
demonstrations drift. The clips do, however, share one state: **the resting
grasp at the end** (every clip ends with a ~2 s hold with the pen pinched).
Blending slowly (0.5-1 s) from one clip's rest pose into the start pose of
another clip whose pen pose is within 6 cm / 60 deg gave 600 proposals, 99
survived the tracking policy, and those 99 edges connected 59% of all frames
into one strongly connected component. In animation terms: the window-matched
jumps give within-trick variety, the hold transitions give trick-to-trick
chaining.

End-to-end replay of whole loops then pruned 5 jumps that passed the local
0.8 s test but broke a 15 s loop (the local test is necessary, not
sufficient); after that 16 of the top 30 loops run a full repeat, and the best
one runs 3 repeats (45 s, 7,000 deg of pen rotation) without dropping the pen.

## 8. Open questions

* **Goal as a point** (如何把目标表示为一个点): a target pen orientation can be
  embedded as a partial feature vector (only the pen blocks) and matched with a
  masked distance; not implemented yet.
* **Shaking when path points are far apart**: the kinematic filter and the
  blend handle the joint side; residual shaking on the real hand would be
  addressed by the tracking policy's EMA and by lowering `max_cost`.
* **Validation horizon**: 0.5-1 s after the jump is tested. A loop is also
  re-validated end to end by `penspin-graph replay`, which is the real test.
* **Cross-group jumps** are rare (< 1% of candidates) because the groups spin
  the pen about different axes; the graph mostly stitches within a group.
  Larger datasets (more teleop / more sim rollouts) would change this.
* **Sim-to-real**: the validated references can be played on the real hand via
  wuji-mjlab's deploy runtime (`pen-policy`) with the matched group policy;
  cross-group loops would need a policy trained on the stitched clips (route 1).

## 9. Roadmap

- [ ] masked goal query (pen-only features) and goal-directed planning
- [ ] DAgger loop for the BC policy; evaluate against Wuji's tracker
- [ ] PPO on `06_graph_stitched` with wuji-mjlab (RTX 4060 / cluster)
- [ ] replay a validated loop on the physical Wuji Hand 2
- [ ] grow the dataset with validated sim rollouts (self-training the graph)
