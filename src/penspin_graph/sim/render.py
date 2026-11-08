"""Offscreen rendering of rollouts (needs an OpenGL backend: ``MUJOCO_GL=egl`` on
headless Linux, the default GLFW/WGL on a desktop)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..features.featurizer import FrameState
from ..graph.planner import ReferenceTrajectory
from .controller import Controller, LiveState, ReferenceFrame
from .rollout import controller_lookahead
from .scene import PenSpinScene


def _camera(lookat=(-0.03, 0.03, 0.47), distance=0.33, elevation=-25.0, azimuth=40.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    return cam


def render_rollout(
    scene: PenSpinScene,
    controller: Controller,
    ref: ReferenceTrajectory,
    start: FrameState,
    out_path: str | Path,
    *,
    width: int = 480,
    height: int = 360,
    fps: int = 25,
    max_steps: int | None = None,
    show_reference_pen: bool = True,
    label: str | None = None,
) -> Path:
    """Track ``ref`` from ``start`` and write an MP4 (or GIF if the suffix is .gif).

    The reference pen pose is drawn as a translucent green capsule so you can
    see how far the simulated pen lags the stitched reference."""
    import imageio.v2 as imageio

    out_path = Path(out_path)
    renderer = mujoco.Renderer(scene.model, height=height, width=width)
    cam = _camera()
    vopt = mujoco.MjvOption()
    vopt.geomgroup[2] = 0  # hide the translucent collision meshes, show the visual hand
    scene.set_state(start.qpos, start.qvel, start.pen_pose, start.pen_linvel, start.pen_angvel)
    controller.reset()
    look = controller_lookahead(controller)
    T = len(ref)
    n = T - 1 if max_steps is None else min(T - 1, max_steps)
    stride = max(1, int(round((1.0 / ref.frame_dt) / fps)))
    frames = []
    for k in range(n):
        live = LiveState(scene.hand_qpos(), scene.hand_qvel(), scene.pen_pose_local(), scene.link_pos_local())
        j = min(k + look, T - 1)
        frame = ReferenceFrame(ref.qpos[j], ref.pen_pose[j], None if ref.link_pos is None else ref.link_pos[j], None if ref.group_ids is None else ref.group_ids[j])
        scene.control_step(controller.act(frame, live))
        if k % stride == 0:
            renderer.update_scene(scene.data, camera=cam, scene_option=vopt)
            if show_reference_pen:
                ref_w = scene.pen_local_to_world(ref.pen_pose[min(k + 1, T - 1)])
                mat = np.zeros(9)
                mujoco.mju_quat2Mat(mat, ref_w[3:])
                scn = renderer.scene
                if scn.ngeom < scn.maxgeom:
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.array([0.006, 0.089, 0.0]), ref_w[:3], mat, np.array([0.2, 0.9, 0.3, 0.35], dtype=np.float32))
                    # capsule axis is local z; rotate so that it lies along the pen's +X
                    rot = np.zeros(9)
                    mujoco.mju_quat2Mat(rot, np.array([0.7071068, 0.0, 0.7071068, 0.0]))  # +90 deg about y: z -> x
                    g.mat[:] = (mat.reshape(3, 3) @ rot.reshape(3, 3)).reshape(g.mat.shape)
                    scn.ngeom += 1
            img = renderer.render()
            frames.append(img.copy())
    renderer.close()
    if out_path.suffix.lower() == ".gif":
        imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    return out_path
