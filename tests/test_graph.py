import numpy as np

from penspin_graph.features import FrameFeaturizer, FrameState
from penspin_graph.graph import JumpStatus, MotionGraph
from penspin_graph.graph.kinematics import KinematicLimits, apply_kinematic_filter
from penspin_graph.graph.loops import find_loops, strongly_connected
from penspin_graph.graph.planner import (
    GraphPlanner,
    MotionMatchingController,
    loop_to_trajectory,
    nearest_nodes,
    path_to_trajectory,
)


def test_featurizer_dims_and_normalisation(synthetic_episodes):
    fz = FrameFeaturizer().fit(synthetic_episodes)
    X = np.concatenate([fz.transform_episode(e) for e in synthetic_episodes])
    assert X.shape[1] == fz.dim == 67
    sl = fz.block_slices()["qpos"]
    # z-scored block times weight 1 -> unit variance per dimension
    assert np.allclose(X[:, sl].std(0), 1.0, atol=0.05)
    ep = synthetic_episodes[0]
    st = FrameState(ep.qpos[7], ep.qvel[7], ep.pen_pose[7], ep.pen_linvel[7], ep.pen_angvel[7], ep.link_pos[7])
    assert np.allclose(fz.transform_state(st), fz.transform_episode(ep)[7], atol=1e-5)


def test_graph_structure(synthetic_graph):
    g = synthetic_graph
    assert g.num_nodes == sum(len(e) for e in g.episodes)
    assert g.intra_src.size == g.num_nodes - g.num_episodes
    assert g.successor(g.node_id(0, len(g.episodes[0]) - 1)) is None
    assert g.successor(g.node_id(1, 3)) == g.node_id(1, 4)
    J = g.jumps
    assert len(J) > 0
    # no jump stays inside its own episode
    assert np.all(g.episode_index[J.src] != g.episode_index[J.dst])
    # near-duplicate episodes (seed 0 and 1) should be connected cheaply
    pair = (g.episode_index[J.src] == 0) & (g.episode_index[J.dst] == 1)
    assert pair.any() and J.cost[pair].min() < 2.0
    assert np.all(np.diff(J.cost) >= 0)  # sorted by cost


def test_nearest_node_recovers_frame(synthetic_graph):
    g = synthetic_graph
    ep = g.episodes[2]
    t = 33
    st = FrameState(ep.qpos[t], ep.qvel[t], ep.pen_pose[t], ep.pen_linvel[t], ep.pen_angvel[t], ep.link_pos[t])
    d, idx = nearest_nodes(g, st, k=1)
    assert idx[0] == g.node_id(2, t) and d[0] < 1e-4


def test_kinematic_filter_marks_failures(synthetic_graph):
    g = synthetic_graph
    limits = KinematicLimits(max_joint_vel=1e-6)  # impossible limit -> everything fails
    before = g.jumps.status.copy()
    rep = apply_kinematic_filter(g, limits)
    assert (~rep.ok).all()
    assert np.all(g.jumps.status[g.jumps.with_status(JumpStatus.KINEMATIC_FAIL)] == int(JumpStatus.KINEMATIC_FAIL))
    g.jumps.status[:] = before  # restore for other tests


def test_loops_and_scc_with_candidates(synthetic_graph):
    g = synthetic_graph
    loops = find_loops(g, statuses=(JumpStatus.CANDIDATE,), max_loops=10, min_frames=2, verbose=False)
    assert loops, "near-duplicate episodes must close at least one loop"
    L = loops[0]
    assert L.n_jumps >= 1 and L.n_frames == len(L.nodes)
    # every consecutive pair is either an intra step or a known jump
    jumps = set(zip(g.jumps.src.tolist(), g.jumps.dst.tolist()))
    for a, b in zip(L.nodes, np.roll(L.nodes, -1)):
        assert (b == a + 1 and g.episode_index[a] == g.episode_index[b]) or (int(a), int(b)) in jumps
    scc = strongly_connected(g, statuses=(JumpStatus.CANDIDATE,))
    assert scc.largest_size > 1
    traj = loop_to_trajectory(g, L, repeats=2, blend_frames=2)
    assert len(traj) == 2 * (L.n_frames + 2 * L.n_jumps)
    assert traj.link_pos.shape == (len(traj), 21, 3) and len(traj.group_ids) == len(traj)


def test_path_to_trajectory_blends(synthetic_graph):
    g = synthetic_graph
    u, v = int(g.jumps.src[0]), int(g.jumps.dst[0])
    traj = path_to_trajectory(g, [u, v, v + 1], blend_frames=3)
    assert len(traj) == 3 + 3
    assert (traj.nodes == -1).sum() == 3
    # blended joint targets lie between the end points
    lo = np.minimum(g.node_qpos(u), g.node_qpos(v)) - 1e-9
    hi = np.maximum(g.node_qpos(u), g.node_qpos(v)) + 1e-9
    assert np.all((traj.qpos[1:4] >= lo) & (traj.qpos[1:4] <= hi))


def test_planner_and_controller(synthetic_graph):
    g = synthetic_graph
    planner = GraphPlanner(g, statuses=(JumpStatus.CANDIDATE,))
    u, v = int(g.jumps.src[0]), int(g.jumps.dst[0])
    path = planner.shortest_path(u, v + 2)
    assert path is not None and path[0] == u and path[-1] == v + 2
    ep = g.episodes[0]
    st = FrameState(ep.qpos[5], ep.qvel[5], ep.pen_pose[5], ep.pen_linvel[5], ep.pen_angvel[5], ep.link_pos[5])
    ctrl = MotionMatchingController(g, requery_every=3)
    ctrl.reset(st)
    for _ in range(5):
        target = ctrl.step(st)
        assert target.shape == (20,)
    assert len(ctrl.history) == 6


def test_save_load_roundtrip(tmp_path, synthetic_graph):
    g = synthetic_graph
    g.save(tmp_path / "graph")
    g2 = MotionGraph.load(tmp_path / "graph", g.episodes)
    assert np.allclose(g2.features, g.features)
    assert np.array_equal(g2.jumps.src, g.jumps.src) and np.array_equal(g2.jumps.status, g.jumps.status)
    assert np.isclose(g2.step_unit, g.step_unit)


def test_active_remaining_and_escape_planning(synthetic_graph):
    g = synthetic_graph
    # synthetic clips never hold still, so the active part is the whole clip
    assert np.all(g.active_end == g.episode_lengths)
    assert g.active_remaining(np.array([g.node_id(0, 10)]))[0] == g.episode_lengths[0] - 10
    from penspin_graph.graph import JumpStatus
    from penspin_graph.graph.planner import MotionMatchingController

    # treat candidates as validated for the test
    ctrl = MotionMatchingController(g, jump_statuses=(JumpStatus.CANDIDATE,), jump_when_remaining_below=30, min_dst_remaining=40)
    late = g.node_id(0, g.episode_lengths[0] - 25)
    path = ctrl._plan_escape(late)
    assert path, "a near-duplicate episode must offer an escape route"
    assert g.episode_index[path[-1]] != 0 or g.frame_index[path[-1]] < g.frame_index[late] - 50
    assert g.active_remaining(np.array([path[-1]]))[0] >= 40


def test_random_walk_dwells_and_jumps(synthetic_graph):
    from penspin_graph.graph import JumpStatus
    from penspin_graph.policy.export_clips import random_walk

    g = synthetic_graph
    rng = np.random.default_rng(1)
    path = []
    for _ in range(20):  # a walk that starts near a clip end may be too short: retry like the exporter does
        path = random_walk(g, rng, min_frames=50, max_frames=400, jump_prob=0.05, min_dwell_frames=20, statuses=(JumpStatus.CANDIDATE,))
        if path:
            break
    assert len(path) >= 50
    arr = np.asarray(path)
    jumps = np.flatnonzero(~((arr[1:] == arr[:-1] + 1) & (g.episode_index[arr[1:]] == g.episode_index[arr[:-1]])))
    # consecutive jumps are separated by at least the dwell (except forced end-of-clip jumps)
    if jumps.size > 1:
        assert np.median(np.diff(jumps)) >= 10
