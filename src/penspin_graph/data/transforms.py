"""Small, dependency-free rigid-body math on NumPy arrays.

Conventions
-----------
* Quaternions are ``(w, x, y, z)`` — the MuJoCo / mjlab convention used by the
  Wuji pen-spin clips.
* A *pose* is a length-7 vector ``[px, py, pz, qw, qx, qy, qz]``.
* Every function is vectorised over a leading batch dimension.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, _EPS)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (its conjugate)."""
    return quat_conj(quat_normalize(q))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` for (w, x, y, z) quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def rotmat_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> unit quaternions (w, x, y, z)."""
    r = np.asarray(r, dtype=np.float64)
    m00, m01, m02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    m10, m11, m12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    m20, m21, m22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    trace = m00 + m11 + m22
    q = np.empty(r.shape[:-2] + (4,), dtype=np.float64)

    # Branch on the largest diagonal element for numerical stability.
    c0 = trace > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)

    s = np.sqrt(np.maximum(trace[c0] + 1.0, _EPS)) * 2.0
    q[c0, 0] = 0.25 * s
    q[c0, 1] = (m21[c0] - m12[c0]) / s
    q[c0, 2] = (m02[c0] - m20[c0]) / s
    q[c0, 3] = (m10[c0] - m01[c0]) / s

    s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], _EPS)) * 2.0
    q[c1, 0] = (m21[c1] - m12[c1]) / s
    q[c1, 1] = 0.25 * s
    q[c1, 2] = (m01[c1] + m10[c1]) / s
    q[c1, 3] = (m02[c1] + m20[c1]) / s

    s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], _EPS)) * 2.0
    q[c2, 0] = (m02[c2] - m20[c2]) / s
    q[c2, 1] = (m01[c2] + m10[c2]) / s
    q[c2, 2] = 0.25 * s
    q[c2, 3] = (m12[c2] + m21[c2]) / s

    s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], _EPS)) * 2.0
    q[c3, 0] = (m10[c3] - m01[c3]) / s
    q[c3, 1] = (m02[c3] + m20[c3]) / s
    q[c3, 2] = (m12[c3] + m21[c3]) / s
    q[c3, 3] = 0.25 * s
    return quat_normalize(q)


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """Continuous 6-D rotation representation (first two matrix columns)."""
    r = quat_to_rotmat(q)
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between two unit quaternions."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions ``a`` and ``b``."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    # Fall back to lerp when the quaternions are nearly parallel.
    linear = quat_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe = sin_theta > 1e-6
    wa = np.where(safe, np.sin((1.0 - t) * theta) / np.where(safe, sin_theta, 1.0), 0.0)
    wb = np.where(safe, np.sin(t * theta) / np.where(safe, sin_theta, 1.0), 0.0)
    slerp = wa * a + wb * b
    return np.where(safe, slerp, linear)


def quat_log_vector(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a unit quaternion, shortest arc."""
    q = quat_normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    sin_half = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, q[..., :1])
    axis = q[..., 1:] / np.maximum(sin_half, _EPS)
    return axis * angle


# --------------------------------------------------------------------------- #
# Frame transforms
# --------------------------------------------------------------------------- #
def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray, q02: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Express frame 2 in frame 1: given world poses of frames 1 and 2 return
    ``(t12, q12)`` such that ``T02 = T01 * T12``. Mirrors mjlab's helper."""
    q10 = quat_inv(q01)
    t12 = quat_apply(q10, np.asarray(t02, dtype=np.float64) - np.asarray(t01, dtype=np.float64))
    q12 = quat_mul(q10, q02)
    return t12, quat_normalize(q12)


def combine_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t12: np.ndarray, q12: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose ``T02 = T01 * T12``."""
    t02 = np.asarray(t01, dtype=np.float64) + quat_apply(q01, t12)
    q02 = quat_mul(q01, q12)
    return t02, quat_normalize(q02)


def pose_to_local(pose_w: np.ndarray, frame_pose_w: np.ndarray) -> np.ndarray:
    """World poses (..., 7) expressed in a (possibly per-row) frame (..., 7)."""
    pose_w = np.asarray(pose_w, dtype=np.float64)
    frame_pose_w = np.asarray(frame_pose_w, dtype=np.float64)
    frame_pose_w = np.broadcast_to(frame_pose_w, pose_w.shape)
    t, q = subtract_frame_transforms(
        frame_pose_w[..., :3], frame_pose_w[..., 3:], pose_w[..., :3], pose_w[..., 3:]
    )
    return np.concatenate([t, q], axis=-1)


def pose_to_world(pose_l: np.ndarray, frame_pose_w: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pose_to_local`."""
    pose_l = np.asarray(pose_l, dtype=np.float64)
    frame_pose_w = np.broadcast_to(np.asarray(frame_pose_w, dtype=np.float64), pose_l.shape)
    t, q = combine_frame_transforms(
        frame_pose_w[..., :3], frame_pose_w[..., 3:], pose_l[..., :3], pose_l[..., 3:]
    )
    return np.concatenate([t, q], axis=-1)


def points_to_local(points_w: np.ndarray, frame_pose_w: np.ndarray) -> np.ndarray:
    """World points (T, K, 3) -> frame-local points given frame poses (T, 7)."""
    points_w = np.asarray(points_w, dtype=np.float64)
    frame_pose_w = np.asarray(frame_pose_w, dtype=np.float64)
    pos = frame_pose_w[..., None, :3]
    quat = np.broadcast_to(frame_pose_w[..., None, 3:], points_w.shape[:-1] + (4,))
    return quat_apply(quat_inv(quat), points_w - pos)


def points_to_world(points_l: np.ndarray, frame_pose_w: np.ndarray) -> np.ndarray:
    points_l = np.asarray(points_l, dtype=np.float64)
    frame_pose_w = np.asarray(frame_pose_w, dtype=np.float64)
    pos = frame_pose_w[..., None, :3]
    quat = np.broadcast_to(frame_pose_w[..., None, 3:], points_l.shape[:-1] + (4,))
    return quat_apply(quat, points_l) + pos


def angular_velocity_from_quats(q: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference body-frame angular velocity (rad/s) of a quaternion
    sequence (T, 4); the last sample repeats the previous one."""
    q = quat_normalize(q)
    dq = quat_mul(quat_inv(q[:-1]), q[1:])  # rotation from frame t to t+1, in frame t
    omega = quat_log_vector(dq) / dt
    return np.concatenate([omega, omega[-1:]], axis=0) if len(omega) else np.zeros((1, 3))


def finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """Forward difference with the last row repeated (same length as input)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros_like(x)
    d = (x[1:] - x[:-1]) / dt
    return np.concatenate([d, d[-1:]], axis=0)
