import numpy as np
import pytest

from penspin_graph.features import FrameFeaturizer
from penspin_graph.graph import JumpStatus, MotionGraph
from penspin_graph.graph.kinematics import KinematicLimits, apply_kinematic_filter
from penspin_graph.policy.export_clips import build_export_trajectories, export_clip_group, random_walk, trajectory_to_clip_arrays
from penspin_graph.graph.planner import path_to_trajectory
from penspin_graph.sim import OnnxTrackingController, PDReferenceController, PenSpinScene, TransitionValidator, ValidationConfig, track_episode
from penspin_graph.sim.controller import bounded_axis_points
from penspin_graph.sim.rollout import frame_state_from_episode


@pytest.fixture(scope="module")
def scene():
    return PenSpinScene()


@pytest.fixture(scope="module")
def real_graph(real_episodes):
    g = MotionGraph(real_episodes, FrameFeaturizer().fit(real_episodes))
    g.propose_jumps(k=6, window=5, max_cost=4.0, verbose=False)
    apply_kinematic_filter(g, KinematicLimits.from_dataset(g))
    return g


def test_scene_matches_dataset_geometry(scene, real_episodes):
    ep = real_episodes[0]
    scene.set_state(ep.qpos[40], ep.qvel[40], ep.pen_pose[40], ep.pen_linvel[40], ep.pen_angvel[40])
    assert scene.wrist_pose_check() < 1e-9
    # forward kinematics of the MJCF reproduces the dataset's link positions (mm-level)
    err = np.linalg.norm(scene.link_pos_local() - ep.link_pos[40], axis=1)
    assert err.max() < 0.01
    assert np.allclose(scene.pen_pose_local()[:3], ep.pen_pose[40, :3], atol=1e-9)
    assert np.allclose(scene.hand_qpos(), ep.qpos[40], atol=1e-9)


def test_bounded_axis_points():
    pts = bounded_axis_points(np.array([0, 0, 0, 1, 0, 0, 0]))
    assert pts.shape == (6,)
    assert np.isclose(pts[0], 0.15 * np.tanh(0.10 / 0.15)) and np.isclose(pts[3], -pts[0])


def test_onnx_policy_tracks_clip_better_than_pd(scene, real_episodes):
    ep = real_episodes[0]
    pd = track_episode(scene, PDReferenceController(), ep, start=0, n_frames=60)
    onnx = track_episode(scene, OnnxTrackingController(), ep, start=0, n_frames=60)
    assert onnx.completed, onnx.summary()
    assert onnx.mean_pos_err() < 0.02
    assert onnx.mean_pos_err() < pd.mean_pos_err() or pd.terminated


def test_validator_accepts_cheapest_jump_and_rejects_absurd_one(real_graph, scene):
    v = TransitionValidator(real_graph, ValidationConfig(follow_frames=20), scene=scene)
    cands = real_graph.jumps.with_status(JumpStatus.CANDIDATE)
    res = v.validate(int(cands[0]))
    assert res.stitched.n_steps > 0 and res.baseline is not None
    assert res.ok, res.summary(real_graph)
    # an impossible success threshold must reject
    v_strict = TransitionValidator(real_graph, ValidationConfig(follow_frames=20, success_pos_m=1e-6), scene=scene)
    assert not v_strict.validate(int(cands[0])).ok


def test_random_walk_and_export(tmp_path, real_graph):
    g = real_graph
    # pretend the cheapest 50 candidates are validated
    idx = g.jumps.with_status(JumpStatus.CANDIDATE)[:50]
    g.jumps.status[idx] = int(JumpStatus.VALIDATED)
    rng = np.random.default_rng(0)
    path = random_walk(g, rng, min_frames=10, max_frames=200, jump_prob=1.0)
    assert len(path) >= 10
    traj = path_to_trajectory(g, path, blend_frames=2)
    arrays = trajectory_to_clip_arrays(g, traj)
    assert arrays["reference_qpos"].shape == (len(traj), 20)
    assert arrays["object_pose_w"].shape == (len(traj), 7) and arrays["link_pos_w"].shape == (len(traj), 21, 3)
    assert np.isfinite(arrays["mediapipe_landmarks_w"]).all()
    trajs = build_export_trajectories(g, loops=None, n_walks=2, target_seconds=3.0, seed=1)
    rep = export_clip_group(g, trajs, out_root=tmp_path / "clips", group_id="06_test")
    # the exported group must load back through our own catalog loader
    from penspin_graph.data import load_clip_groups

    eps = load_clip_groups(tmp_path / "clips", verify_hashes=True)
    assert len(eps) == len(trajs) and eps[0].group_id == "06_test"
    assert rep.recipe_path.is_file()
    g.jumps.status[idx] = int(JumpStatus.CANDIDATE)
