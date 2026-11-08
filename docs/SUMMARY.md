# Project summary: motion-graph pen spinning on the Wuji Hand 2

## Short answer

* **What was built.** Every frame of Wuji's 40 pen-spinning demos is a point in a 67-D space. Frames are chained inside each demo, and **38,624 transitions (跳转)** that were tested in simulation link different demos and different moments of the same demo. A controller finds the nearest point to the current hand state, plans into a verified **loop (环路)**, and circles it with Wuji's own tracking policy doing the low-level control.
* **How it differs from the original.** Wuji's original pipeline can only play one recorded clip from its first frame until the clip ends. Our version can start from any state, switch between demos, and spin indefinitely.
* **Gains, all in simulation.** Continuous spinning went from **21.8 s to 84.7 s (3.9x)**. Pen-shaft rotation per run went from **1,584° to 8,709° (5.5x)**. The rotation speed of the best loop is **114°/s**, while the clips average 53°/s. From random states, a 30 s test gives **2.5x more rotation** than plain nearest-frame tracking, and **4x as many runs are still spinning after 20 s** (8/20 vs 2/20). The price is a slightly higher drop rate (15% vs 10%).

---

## 1. The full pipeline, step by step (实现流程)

| # | step | what it does | key numbers |
|---|---|---|---|
| 0 | **Data and assets** | Wuji release v2026.10.22: 40 retargeted human pen-spinning clips in 5 motion groups, the Hand 2 MJCF (MuJoCo XML Format, MuJoCo 模型文件格式) model, and 5 released PPO (Proximal Policy Optimization, 近端策略优化) tracking policies in ONNX (Open Neural Network Exchange, 开放神经网络交换格式) format | 25,885 frames at 50 Hz (8.6 min) |
| 1 | **Wrist-frame conversion** (手腕坐标系) | expresses the pen pose, link positions and velocities relative to the wrist, so where the hand stood during capture does not matter | — |
| 2 | **Embedding** (嵌入) | each frame becomes one 67-D vector: joint angles, joint velocities, pen position, pen shaft direction, pen velocities and fingertips. Each block is z-scored (标准化) and weighted | unit = 1 typical 50 Hz step |
| 3 | **Motion graph** (运动图) | nodes = frames; edges = recorded `t -> t+1`. **Candidate jumps** come from kNN (k-Nearest Neighbours, k近邻) search plus **window matching** (窗口匹配): the next 5 frames after the jump must match too. Same-demo jumps are allowed if the two frames are at least 1 s apart, and the resting **holds** (静止段) are excluded | 101,934 candidates |
| 4 | **Trick chaining** (动作衔接) | each demo's final resting grasp is blended slowly (0.5-1 s) into the start of another demo, provided the pen poses agree | 600 candidates |
| 5 | **Kinematic filter** (运动学过滤) | checks joint limits and joint speed ≤ 5.8 rad/s (with an adaptive blend of 2-10 frames), and requires the pen not to "teleport" | 5,403 removed |
| 6 | **Simulation validation** (仿真验证) | in a standalone MuJoCo scene, each jump is executed from the recorded state with Wuji's tracking policy in the loop. It must stay within Wuji's failure thresholds and track no worse than 1.5x the original demo | **40,208 tested, 38,624 passed (96%)** |
| 7 | **Loops and SCC** (Strongly Connected Component, 强连通分量) | Dijkstra (迪杰斯特拉最短路径) closes cycles through validated edges. Whole loops are then replayed end to end, and any jump blamed for a failure is pruned (5 rounds) | SCC = 15,215 frames (59%); 151 loops; 16 of the top 30 run a full lap |
| 8 | **Closed-loop controller** (闭环控制器) | nearest frame → plan into the best loop → circle it. It re-matches only when the **pen** leaves its reference (> 3 cm or 40°). The tracker deliberately moves the joints up to ±0.6 rad away from the reference, so joint distance is a bad "am I lost?" signal | see section 4 |
| 9 | **Policy** (策略) | (a) export stitched clips in wuji-mjlab's exact format for PPO training on a GPU (ready, not trained yet); (b) BC (Behaviour Cloning, 行为克隆) into a reference-free MLP (Multi-Layer Perceptron, 多层感知机) | 30 stitched clips (12.3 min); BC: 3.6 s mean survival |
| 10 | **Comparison** (对比实验) | `scripts/compare_baselines.py`: original clip replay vs plain matching vs graph vs loop seeking | section 4 |

## 2. Differences from the original idea (与最初想法的区别)

| original idea | what was implemented | why it changed |
|---|---|---|
| every frame → a high-dimensional point | same; 67-D, in the wrist frame, normalized per block | the representation must not depend on where the hand was during capture |
| frames of one episode connected, different episodes not | same as the starting point; **jumps inside one episode are also allowed** (≥ 1 s apart) | a jump back in time inside a continuous-spin demo is the most natural way to repeat a turn |
| from a random state, find the closest point | same (kNN); the controller **re-matches only when the pen is lost** | the tracking policy keeps the joints far from the reference, so re-matching on full distance caused constant, wrong switching |
| find the most related point other than its own `t+1`, then repeat | done **offline for all frames**, comparing **5-frame windows** rather than single frames, plus a kinematic filter | single-frame similarity ignores where the motion is going |
| keep a loop only if the real hand can run it | judged in **MuJoCo with Wuji's tracking policy**, relative to how well the original demo itself is tracked. Whole loops are then verified and broken jumps pruned | open-loop replay drops the pen within 0.2 s even for the original demos, so it cannot be the test. A jump that passes locally can still break a 15 s loop |
| success → connect the two episodes | same: validated jumps become graph edges | — |
| try all points in sim, then train a policy | 40,208 of 102,534 candidates tested (cheapest and loop-closing first). BC is trained; the PPO export is ready | testing every candidate costs CPU time; testing ~40% was enough for a 59% SCC |
| (not in the idea) | **trick chaining through the resting grasp**; adaptive blends; loop-seeking controller | window-matched jumps alone produced only tiny loops (< 20°), because the full state is not periodic within a demo. The shared resting grasp is what connects the demos |

## 3. Differences from Wuji's original pipeline (与 Wuji 原始实现的区别)

| | Wuji `wuji-mjlab` (original) | this project |
|---|---|---|
| reference motion | one fixed recorded clip | a walk on the graph, chosen online |
| start state | must be the clip's first frame | any state (nearest frame) |
| duration | ends when the clip ends (≤ 21.8 s of motion) | unbounded loops (84.7 s tested) |
| combining demos | impossible | 38,624 validated transitions; the best loop uses 3 demos |
| after a disturbance | keeps following the clip timeline | re-matches to the nearest frame and re-plans |
| low-level control | PPO residual tracking policy, one per group | **the same policies, reused unchanged** |
| training | PPO on GPU with mjlab / MuJoCo-Warp | none needed for the graph (CPU only); the stitched clips can feed the same PPO training |

## 4. Results vs the original (效果对比)

All numbers come from the same MuJoCo scene and the same Wuji tracking policy. Rotation = **rotation of the pen shaft** (笔杆方向的转动), with roll about the shaft ignored.

**Endurance (持续转笔能力)**

| | original (best single clip) | ours (best stitched loop) | gain |
|---|---|---|---|
| continuous spinning | 21.8 s | **84.7 s** (5.2 laps of 16.4 s) | 3.9x |
| shaft rotation in one run | 1,584° | **8,709° (24 turns)** | 5.5x |
| average rotation speed | 53°/s (clip mean), 108°/s (fastest clip) | **114°/s** | 2.2x vs mean |
| original clips completed by Wuji's policy in this scene | 30 / 40 | — | — |

**From 20 random moving states, 30 s each (随机初始状态)**

| setting | mean rotation | rotation in last 10 s | still spinning after 20 s | drops |
|---|---|---|---|---|
| B: nearest frame + follow its demo (no graph) | 555° | 72° | 2 / 20 | 10% |
| C: + validated jumps (escape when a demo ends) | 902° | 123° | 7 / 20 | 10% |
| **D: + seek and circle the best loop (full idea)** | **1,385°** | **384°** | **8 / 20** | 15% |

D beats B in 14 of 20 starts, ties in 3 and is worse in 3. In 3 starts no path into the loop existed, so D behaved exactly like B.

**Coverage (覆盖率):** 59% of all frames can keep spinning forever using only validated moves. In clip replay this is 0%, because every clip ends.

## 5. Limitations and next steps (局限与下一步)

* All results are **in simulation**. The scene matches Wuji's essentials (fixed wrist, 0.005 s physics, condim-4 contacts, 30 g pen) but has no domain randomization. Wuji's own policy finishes only 30 of 40 original clips here, so absolute numbers will differ on the real hand.
* Loop seeking slightly increases the drop rate (15% vs 10%). The jumps on the way into the loop are the risky part.
* The BC policy is weaker than graph + tracker (3.6 s mean survival). Next steps: DAgger (Dataset Aggregation, 数据集聚合) with the graph controller as the expert, or PPO on the exported stitched clips (`runs/full3/export`, RTX 4060, about 2048 envs).
* Real-robot test: play `runs/full3/loop_0_reference.npz` through `wuji-mjlab/deploy/pen_spin` with the group policies.
