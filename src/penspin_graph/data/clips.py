"""Load the Wuji Hand 2 pen-spinning motion clips into wrist-local episodes.

The clips come from the public ``wuji-technology/wuji-mjlab`` v2026.10.22
asset bundle (Apache-2.0).  Each ``clip_XXX.npz`` contains

======================  ===========  =================================================
array                   shape        meaning
======================  ===========  =================================================
``reference_qpos``      (T, 20)      finger joint angles, finger1..5 x joint1..4 (rad)
``wrist_pose_w``        (T, 7)       wrist/palm pose in the capture world frame
``object_pose_w``       (T, 7)       pen pose in the capture world frame
``link_pos_w``          (T, 21, 3)   palm + 20 finger-link positions (world)
``mediapipe_landmarks`` (T, 21, 3)   human-hand landmarks the clip was retargeted from
======================  ===========  =================================================

Frames are 0.02 s apart (50 Hz).  Everything we compute downstream lives in the
**wrist-local frame**, which makes the representation independent of where the
hand happened to be in the capture volume and lets us re-express the data in
the simulator's fixed wrist pose.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .transforms import (
    angular_velocity_from_quats,
    finite_difference,
    points_to_local,
    pose_to_local,
    quat_apply,
    quat_normalize,
)

FRAME_DT = 0.02
NUM_JOINTS = 20
NUM_LINKS = 21

JOINT_NAMES: tuple[str, ...] = tuple(
    f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)
LINK_NAMES: tuple[str, ...] = ("right_palm_link",) + tuple(
    f"right_finger{finger}_link{link}" for finger in range(1, 6) for link in range(1, 5)
)
# Index of the distal link (the fingertip link) of each finger inside ``link_pos``.
FINGERTIP_LINK_INDICES: tuple[int, ...] = tuple(1 + 4 * f + 3 for f in range(5))

# The pen shaft runs along the pen body's +X axis (see wuji-mjlab pen_tracking.xml).
PEN_SHAFT_AXIS_LOCAL = np.array([1.0, 0.0, 0.0])

REQUIRED_ARRAYS = ("reference_qpos", "wrist_pose_w", "object_pose_w", "link_pos_w")


@dataclass
class Episode:
    """One motion clip, expressed in the wrist-local frame."""

    episode_id: str  # "<group_id>/<clip_id>"
    group_id: str
    clip_id: str
    frame_dt: float
    qpos: np.ndarray  # (T, 20)
    qvel: np.ndarray  # (T, 20) forward difference
    pen_pose: np.ndarray  # (T, 7) wrist-local pen pose (pos, wxyz quat)
    pen_linvel: np.ndarray  # (T, 3) wrist-local linear velocity
    pen_angvel: np.ndarray  # (T, 3) pen-body-frame angular velocity
    link_pos: np.ndarray  # (T, 21, 3) wrist-local link positions
    wrist_pose_w: np.ndarray  # (T, 7) raw capture-frame wrist pose
    pen_pose_w: np.ndarray  # (T, 7) raw capture-frame pen pose
    source_path: Path | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration_s(self) -> float:
        return len(self) * self.frame_dt

    @property
    def pen_axis(self) -> np.ndarray:
        """Unit shaft direction of the pen in the wrist frame, (T, 3)."""
        return quat_apply(self.pen_pose[:, 3:], np.broadcast_to(PEN_SHAFT_AXIS_LOCAL, (len(self), 3)))

    @property
    def fingertips(self) -> np.ndarray:
        """(T, 5, 3) wrist-local fingertip (distal link) positions."""
        return self.link_pos[:, list(FINGERTIP_LINK_INDICES), :]

    def slice(self, start: int, stop: int, *, new_id: str | None = None) -> "Episode":
        sl = slice(start, stop)
        return Episode(
            episode_id=new_id or f"{self.episode_id}[{start}:{stop}]",
            group_id=self.group_id,
            clip_id=self.clip_id,
            frame_dt=self.frame_dt,
            qpos=self.qpos[sl].copy(),
            qvel=self.qvel[sl].copy(),
            pen_pose=self.pen_pose[sl].copy(),
            pen_linvel=self.pen_linvel[sl].copy(),
            pen_angvel=self.pen_angvel[sl].copy(),
            link_pos=self.link_pos[sl].copy(),
            wrist_pose_w=self.wrist_pose_w[sl].copy(),
            pen_pose_w=self.pen_pose_w[sl].copy(),
            source_path=self.source_path,
            extras={k: v[sl].copy() for k, v in self.extras.items()},
        )


def episode_from_arrays(
    *,
    episode_id: str,
    group_id: str,
    clip_id: str,
    reference_qpos: np.ndarray,
    wrist_pose_w: np.ndarray,
    object_pose_w: np.ndarray,
    link_pos_w: np.ndarray,
    frame_dt: float = FRAME_DT,
    source_path: Path | None = None,
    extras: dict[str, np.ndarray] | None = None,
) -> Episode:
    """Build an :class:`Episode` from raw world-frame arrays."""
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    wrist_pose_w = np.asarray(wrist_pose_w, dtype=np.float64)
    object_pose_w = np.asarray(object_pose_w, dtype=np.float64)
    link_pos_w = np.asarray(link_pos_w, dtype=np.float64)
    frames = qpos.shape[0]
    expected = {
        "reference_qpos": (frames, NUM_JOINTS),
        "wrist_pose_w": (frames, 7),
        "object_pose_w": (frames, 7),
        "link_pos_w": (frames, NUM_LINKS, 3),
    }
    for name, (arr, shape) in zip(
        expected, [(qpos, expected["reference_qpos"]), (wrist_pose_w, expected["wrist_pose_w"]),
                   (object_pose_w, expected["object_pose_w"]), (link_pos_w, expected["link_pos_w"])]
    ):
        if tuple(arr.shape) != shape:
            raise ValueError(f"{episode_id}: {name} must have shape {shape}, got {tuple(arr.shape)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{episode_id}: {name} contains non-finite values")

    wrist_pose_w = np.concatenate([wrist_pose_w[:, :3], quat_normalize(wrist_pose_w[:, 3:])], axis=1)
    object_pose_w = np.concatenate([object_pose_w[:, :3], quat_normalize(object_pose_w[:, 3:])], axis=1)

    pen_pose = pose_to_local(object_pose_w, wrist_pose_w)
    link_pos = points_to_local(link_pos_w, wrist_pose_w)
    qvel = finite_difference(qpos, frame_dt)
    pen_linvel = finite_difference(pen_pose[:, :3], frame_dt)
    pen_angvel = angular_velocity_from_quats(pen_pose[:, 3:], frame_dt)

    return Episode(
        episode_id=episode_id,
        group_id=group_id,
        clip_id=clip_id,
        frame_dt=float(frame_dt),
        qpos=qpos,
        qvel=qvel,
        pen_pose=pen_pose,
        pen_linvel=pen_linvel,
        pen_angvel=pen_angvel,
        link_pos=link_pos,
        wrist_pose_w=wrist_pose_w,
        pen_pose_w=object_pose_w,
        source_path=source_path,
        extras=dict(extras or {}),
    )


def load_clip(path: str | Path, *, group_id: str | None = None, frame_dt: float = FRAME_DT) -> Episode:
    path = Path(path)
    group = group_id or path.parent.name
    clip_id = path.stem
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in REQUIRED_ARRAYS if k not in data]
        if missing:
            raise KeyError(f"{path}: missing arrays {missing}")
        extras = {k: np.asarray(data[k]) for k in data.files if k not in REQUIRED_ARRAYS}
        return episode_from_arrays(
            episode_id=f"{group}/{clip_id}",
            group_id=group,
            clip_id=clip_id,
            reference_qpos=data["reference_qpos"],
            wrist_pose_w=data["wrist_pose_w"],
            object_pose_w=data["object_pose_w"],
            link_pos_w=data["link_pos_w"],
            frame_dt=frame_dt,
            source_path=path,
            extras=extras,
        )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clip_groups(
    clips_root: str | Path,
    *,
    groups: list[str] | tuple[str, ...] | None = None,
    verify_hashes: bool = False,
) -> list[Episode]:
    """Load every clip under ``clips_root/<group>/`` (wuji-mjlab catalog layout).

    Groups without an ``index.json`` are still loaded from their ``*.npz`` files.
    """
    root = Path(clips_root)
    if not root.is_dir():
        raise FileNotFoundError(f"clip root is not a directory: {root}")
    episodes: list[Episode] = []
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if groups is not None and group_dir.name not in groups:
            continue
        index_path = group_dir / "index.json"
        frame_dt = FRAME_DT
        entries: list[tuple[Path, str | None]]
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            frame_dt = float(index.get("frame_dt", FRAME_DT))
            if index.get("group_id", group_dir.name) != group_dir.name:
                raise ValueError(f"{index_path}: group_id does not match directory name")
            entries = [(group_dir / e["file"], e.get("sha256")) for e in index["clips"]]
        else:
            entries = [(p, None) for p in sorted(group_dir.glob("*.npz"))]
        for npz_path, expected in entries:
            if verify_hashes and expected is not None and sha256_of(npz_path) != expected:
                raise ValueError(f"SHA-256 mismatch for {npz_path}")
            episodes.append(load_clip(npz_path, group_id=group_dir.name, frame_dt=frame_dt))
    if not episodes:
        raise ValueError(f"no clips found under {root}")
    return episodes


def summarize(episodes: list[Episode]) -> str:
    by_group: dict[str, list[Episode]] = {}
    for ep in episodes:
        by_group.setdefault(ep.group_id, []).append(ep)
    lines = [f"{len(episodes)} episodes, {sum(len(e) for e in episodes)} frames"]
    for group, eps in by_group.items():
        frames = sum(len(e) for e in eps)
        lines.append(f"  {group}: {len(eps)} clips, {frames} frames ({frames * eps[0].frame_dt:.1f} s)")
    return "\n".join(lines)
