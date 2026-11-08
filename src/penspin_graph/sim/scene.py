"""A standalone MuJoCo scene: right Wuji Hand 2 on a fixed wrist mount + 250 mm pen.

This intentionally mirrors the essentials of wuji-mjlab's ``WujiHand2_PenSpin``
simulation (fixed wrist pose, 0.005 s physics step with 4 substeps per 50 Hz
control tick, implicitfast integrator, condim-4 contacts, 30 g pen) without
depending on mjlab / MuJoCo-Warp, so it runs on a laptop CPU.  It is used to
*validate* proposed transitions and to replay references, not to train PPO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ..data.clips import JOINT_NAMES, NUM_JOINTS
from ..data.transforms import (
    combine_frame_transforms,
    pose_to_local,
    pose_to_world,
    quat_apply,
    quat_inv,
    quat_mul,
    quat_normalize,
    subtract_frame_transforms,
)

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets"
HAND2_XML = ASSET_DIR / "wuji_hand2" / "mjcf" / "right.xml"

# Physical Wuji Hand 2 wrist pose used by wuji-mjlab for sim and deploy (pos, wxyz).
FIXED_WRIST_POSE_W = np.array(
    [
        -0.028110361960901435,
        0.00300004,
        0.4952974018391883,
        0.5416752204197018,
        0.45451947767204376,
        0.45451947767204376,
        0.5416752204197018,
    ]
)

# MJCF joint names in ``reference_qpos`` order (finger1..5 x joint1..4).
HAND2_MJCF_JOINT_NAMES: tuple[str, ...] = (
    "r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip",
    "r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip",
    "r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd", "r_middle_finger_pip", "r_middle_finger_dip",
    "r_ring_finger_mcp_flex", "r_ring_finger_mcp_abd", "r_ring_finger_pip", "r_ring_finger_dip",
    "r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip",
)
# Bodies in ``link_pos`` order (palm + finger1..5 x link1..4).
HAND2_MJCF_LINK_NAMES: tuple[str, ...] = ("r_wrist",) + tuple(
    f"{finger}_{part}"
    for finger in ("r_thumb", "r_index_finger", "r_middle_finger", "r_ring_finger", "r_pinky")
    for part in ("proximal", "proximal_abd", "middle", "distal")
)


@dataclass
class SceneConfig:
    hand_xml: Path = HAND2_XML
    wrist_pose_w: np.ndarray = field(default_factory=lambda: FIXED_WRIST_POSE_W.copy())
    pen_mass_kg: float = 0.030
    pen_half_length_m: float = 0.089  # shaft fromto +-0.089
    pen_radius_m: float = 0.00745
    end_block_center_m: float = 0.107
    end_block_half_size_m: tuple[float, float, float] = (0.018, 0.009, 0.009)  # x long
    timestep: float = 0.005
    substeps: int = 4  # 4 x 0.005 s = 50 Hz control
    impratio: float = 2.0
    condim: int = 4
    friction: tuple[float, float, float] = (0.7, 0.005, 0.0001)
    solref: tuple[float, float] = (0.0075, 1.0)  # time constant 1.5 * timestep
    add_floor: bool = True
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.substeps


class PenSpinScene:
    """Build and drive the hand+pen MuJoCo model."""

    def __init__(self, cfg: SceneConfig | None = None) -> None:
        self.cfg = cfg or SceneConfig()
        self.model, self.data = self._build()
        m = self.model
        self.joint_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND2_MJCF_JOINT_NAMES])
        if (self.joint_ids < 0).any():
            raise RuntimeError("hand joints missing from MJCF")
        self.qpos_adr = m.jnt_qposadr[self.joint_ids]
        self.dof_adr = m.jnt_dofadr[self.joint_ids]
        pen_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pen_free")
        self.pen_qpos_adr = int(m.jnt_qposadr[pen_jnt])
        self.pen_dof_adr = int(m.jnt_dofadr[pen_jnt])
        self.pen_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pen")
        self.wrist_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist")
        self.link_bodies = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in HAND2_MJCF_LINK_NAMES])
        # actuators are listed in the same joint order in the MJCF; map explicitly anyway
        act_joint = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
        self.actuator_ids = np.array([act_joint.index(j) for j in self.joint_ids])
        self.ctrl_lower = m.actuator_ctrlrange[self.actuator_ids, 0]
        self.ctrl_upper = m.actuator_ctrlrange[self.actuator_ids, 1]
        self.joint_lower = m.jnt_range[self.joint_ids, 0]
        self.joint_upper = m.jnt_range[self.joint_ids, 1]

    # ------------------------------------------------------------------ #
    def _build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        cfg = self.cfg
        spec = mujoco.MjSpec.from_file(str(cfg.hand_xml))
        spec.option.timestep = cfg.timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.impratio = cfg.impratio
        spec.option.gravity = list(cfg.gravity)
        spec.option.iterations = 10
        spec.option.ls_iterations = 20

        # Place the mount so that the wrist body lands on the fixed wrist pose.
        wrist = spec.body("r_wrist")
        mount = spec.body("r_mount")
        t_wm, q_wm = np.asarray(wrist.pos, float), quat_normalize(np.asarray(wrist.quat, float))
        # T_mount = T_wrist_world * inv(T_wrist_in_mount)
        q_inv = quat_inv(q_wm)
        t_inv = -quat_apply(q_inv, t_wm)
        t_m, q_m = combine_frame_transforms(cfg.wrist_pose_w[:3], cfg.wrist_pose_w[3:], t_inv, q_inv)
        mount.pos = t_m
        mount.quat = q_m

        # Contact tuning on the hand's collision geoms (mirrors pen-spin's task tuning).
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.condim = cfg.condim
                geom.friction = list(cfg.friction)
                geom.solref = list(cfg.solref)
                geom.priority = 1

        # Pen: shaft cylinder + two tagged end blocks, inertia scaled to pen_mass_kg.
        pen = spec.worldbody.add_body(name="pen", pos=[0.0, 0.0, 1.0])
        L, r, c = cfg.pen_half_length_m, cfg.pen_radius_m, cfg.end_block_center_m
        hx, hy, hz = cfg.end_block_half_size_m
        common = dict(contype=1, conaffinity=2, condim=cfg.condim, friction=list(cfg.friction),
                      solref=list(cfg.solref), density=469.05, group=1)
        pen.add_geom(name="pen_shaft", type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[-L, 0, 0, L, 0, 0], size=[r, 0, 0], rgba=[0.95, 0.95, 0.95, 1], **common)
        pen.add_geom(name="pen_end_a", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-c, 0, 0], size=[hx, hy, hz], rgba=[0.1, 0.1, 0.1, 1], **common)
        pen.add_geom(name="pen_end_b", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[c, 0, 0], size=[hx, hy, hz], rgba=[0.1, 0.1, 0.1, 1], **common)
        pen.add_freejoint(name="pen_free")
        pen.add_site(name="pen_center", pos=[0, 0, 0], size=[0.002, 0, 0], group=4)

        if cfg.add_floor:
            spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.8, 0.8, 0.8, 1])
            spec.worldbody.add_light(pos=[0.5, -0.5, 1.5], dir=[-0.3, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
            spec.worldbody.add_light(pos=[-0.5, 0.5, 1.2], dir=[0.3, -0.3, -1], diffuse=[0.5, 0.5, 0.5])

        model = spec.compile()
        # scale pen inertia to the target mass (same recipe as wuji-mjlab's pen_cfg)
        pen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
        scale = cfg.pen_mass_kg / float(model.body_mass[pen_id])
        model.body_mass[pen_id] *= scale
        model.body_inertia[pen_id] *= scale
        mujoco.mj_setConst(model, mujoco.MjData(model))
        return model, mujoco.MjData(model)

    # ------------------------------------------------------------------ #
    # frames
    # ------------------------------------------------------------------ #
    @property
    def wrist_pose_w(self) -> np.ndarray:
        return self.cfg.wrist_pose_w

    def pen_local_to_world(self, pen_pose_wl: np.ndarray) -> np.ndarray:
        return pose_to_world(pen_pose_wl, self.wrist_pose_w)

    def pen_world_to_local(self, pen_pose_w: np.ndarray) -> np.ndarray:
        return pose_to_local(pen_pose_w, self.wrist_pose_w)

    # ------------------------------------------------------------------ #
    # state I/O
    # ------------------------------------------------------------------ #
    def set_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        pen_pose_wl: np.ndarray,
        pen_linvel_wl: np.ndarray | None = None,
        pen_angvel_body: np.ndarray | None = None,
    ) -> None:
        """Teleport the hand and pen to a dataset frame (wrist-local inputs)."""
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qpos_adr] = np.clip(qpos, self.joint_lower, self.joint_upper)
        d.qvel[self.dof_adr] = qvel
        pen_w = self.pen_local_to_world(np.asarray(pen_pose_wl, float))
        d.qpos[self.pen_qpos_adr : self.pen_qpos_adr + 7] = pen_w
        if pen_linvel_wl is not None:
            d.qvel[self.pen_dof_adr : self.pen_dof_adr + 3] = quat_apply(self.wrist_pose_w[3:], np.asarray(pen_linvel_wl, float))
        if pen_angvel_body is not None:
            # MuJoCo free-joint angular velocity is expressed in the body frame.
            d.qvel[self.pen_dof_adr + 3 : self.pen_dof_adr + 6] = np.asarray(pen_angvel_body, float)
        d.ctrl[self.actuator_ids] = d.qpos[self.qpos_adr]
        d.act[:] = 0.0 if d.act.size else d.act
        mujoco.mj_forward(self.model, d)

    def hand_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    def hand_qvel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    def pen_pose_world(self) -> np.ndarray:
        return self.data.qpos[self.pen_qpos_adr : self.pen_qpos_adr + 7].copy()

    def pen_pose_local(self) -> np.ndarray:
        return self.pen_world_to_local(self.pen_pose_world())

    def pen_velocity_local(self) -> tuple[np.ndarray, np.ndarray]:
        lin_w = self.data.qvel[self.pen_dof_adr : self.pen_dof_adr + 3]
        ang_b = self.data.qvel[self.pen_dof_adr + 3 : self.pen_dof_adr + 6]
        lin_wl = quat_apply(quat_inv(self.wrist_pose_w[3:]), lin_w)
        return lin_wl, ang_b.copy()

    def link_pos_local(self) -> np.ndarray:
        pos_w = self.data.xpos[self.link_bodies]
        rel = pos_w - self.wrist_pose_w[:3]
        return quat_apply(np.broadcast_to(quat_inv(self.wrist_pose_w[3:]), (rel.shape[0], 4)), rel)

    # ------------------------------------------------------------------ #
    def control_step(self, target_qpos: np.ndarray) -> None:
        """Apply joint-position targets for one 50 Hz tick (``substeps`` physics steps)."""
        self.data.ctrl[self.actuator_ids] = np.clip(target_qpos, self.ctrl_lower, self.ctrl_upper)
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)

    def wrist_pose_check(self) -> float:
        """Distance between the wrist body and the configured wrist pose (should be ~0)."""
        return float(np.linalg.norm(self.data.xpos[self.wrist_body] - self.wrist_pose_w[:3]))
