import numpy as np

from penspin_graph.data import load_clip_groups
from penspin_graph.data.transforms import (
    angular_velocity_from_quats,
    pose_to_local,
    pose_to_world,
    quat_angle,
    quat_apply,
    quat_mul,
    quat_normalize,
    quat_slerp,
    quat_to_rot6d,
    quat_to_rotmat,
    rotmat_to_quat,
)


def _random_quats(n, seed=0):
    return quat_normalize(np.random.default_rng(seed).standard_normal((n, 4)))


def test_quat_roundtrips():
    q = _random_quats(50)
    r = quat_to_rotmat(q)
    q2 = rotmat_to_quat(r)
    assert np.allclose(quat_angle(q, q2), 0.0, atol=1e-6)
    v = np.random.default_rng(1).standard_normal((50, 3))
    assert np.allclose(quat_apply(q, v), np.einsum("nij,nj->ni", r, v), atol=1e-9)
    assert quat_to_rot6d(q).shape == (50, 6)


def test_pose_local_world_roundtrip():
    frame = np.concatenate([[0.1, -0.2, 0.3], _random_quats(1, 3)[0]])
    poses = np.concatenate([np.random.default_rng(4).standard_normal((20, 3)), _random_quats(20, 5)], axis=1)
    local = pose_to_local(poses, frame)
    back = pose_to_world(local, frame)
    assert np.allclose(back[:, :3], poses[:, :3], atol=1e-9)
    assert np.allclose(quat_angle(back[:, 3:], poses[:, 3:]), 0.0, atol=1e-6)


def test_slerp_endpoints_and_midpoint():
    a = quat_normalize(np.array([[1.0, 0, 0, 0]]))
    b = quat_normalize(np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]]))  # 90 deg about z
    assert np.allclose(quat_slerp(a, b, 0.0), a)
    assert np.allclose(quat_angle(quat_slerp(a, b, 1.0), b), 0.0, atol=1e-9)
    mid = quat_slerp(a, b, 0.5)
    assert np.isclose(quat_angle(a, mid)[0], np.pi / 4, atol=1e-6)


def test_angular_velocity_constant_spin():
    dt, w = 0.02, 3.0
    t = np.arange(50) * dt
    q = np.stack([np.cos(w * t / 2), np.zeros_like(t), np.sin(w * t / 2), np.zeros_like(t)], 1)  # about y
    omega = angular_velocity_from_quats(q, dt)
    assert np.allclose(omega[:-1], [0, w, 0], atol=1e-6)


def test_quat_mul_identity():
    q = _random_quats(5)
    e = np.array([1.0, 0, 0, 0])
    assert np.allclose(quat_mul(q, np.broadcast_to(e, q.shape)), q)


def test_real_clips_load_and_are_wrist_local(real_episodes):
    assert len(real_episodes) == 10
    ep = real_episodes[0]
    assert ep.qpos.shape[1] == 20 and ep.link_pos.shape[1:] == (21, 3)
    # palm link is the wrist frame origin
    assert np.allclose(ep.link_pos[:, 0], 0.0, atol=1e-6)
    # the pen stays within reach of the fingers
    dist = np.linalg.norm(ep.pen_pose[:, :3], axis=1)
    assert 0.05 < dist.mean() < 0.2
    assert np.allclose(np.linalg.norm(ep.pen_pose[:, 3:], axis=1), 1.0, atol=1e-6)
