#!/usr/bin/env python3
"""Assemble the printed rig + Wuji's jig + the hand (start pose) + a D435i, render it.

Writes hardware/renders/*.png and hardware/parts/assembly.glb (open in any 3D viewer /
Windows 3D Viewer / the HTML viewer in hardware/viewer.html).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import pyrender
import trimesh

HERE = Path(__file__).resolve().parent
PARTS = HERE / "parts"
REND = HERE / "renders"
REND.mkdir(exist_ok=True)
info = json.loads((PARTS / "parts.json").read_text())
T_JIG = np.array(info["jig_transform"])
COL = info["column_centre"]
PLATE = info["plate_centre"]
TILT = math.radians(10.0)


def load(name):
    return trimesh.load(str(PARTS / f"{name}.stl"), force="mesh")


def translate(x, y, z):
    T = np.eye(4); T[:3, 3] = [x, y, z]; return T


def colored(mesh, rgba):
    m = mesh.copy(); m.visual = trimesh.visual.ColorVisuals(m, face_colors=(np.array(rgba) * 255).astype(np.uint8)); return m


def build():
    scene = trimesh.Scene()
    tile = load("breadboard_tile")
    for i, (x, y) in enumerate([(0, 0), (175, 0), (0, 125), (175, 125)]):
        scene.add_geometry(colored(tile, (0.22, 0.22, 0.24, 1)), node_name=f"tile{i}", geom_name=f"tile{i}", transform=translate(x, y, -12.0))
    sp = load("splice_plate")
    for i, (x, y) in enumerate([(150, 100), (100, 100), (250, 100), (150, 0), (150, 200)]):
        scene.add_geometry(colored(sp, (0.35, 0.35, 0.38, 1)), node_name=f"splice{i}", geom_name=f"splice{i}", transform=translate(x, y, -20.0))
    jig = load("wuji_hand2_base")
    scene.add_geometry(colored(jig, (0.90, 0.25, 0.12, 1)), node_name="wuji_jig", geom_name="wuji_jig", transform=T_JIG)
    base = load("camera_column_base"); top = load("camera_column_top")
    scene.add_geometry(colored(base, (0.20, 0.45, 0.85, 1)), node_name="column_base", geom_name="column_base", transform=translate(COL[0], COL[1], 0))
    scene.add_geometry(colored(top, (0.15, 0.60, 0.95, 1)), node_name="column_top", geom_name="column_top", transform=translate(COL[0], COL[1], 0))
    # hand + pen from the MuJoCo scene (start pose of the best loop), mounted on the jig flange
    hand = trimesh.load("/tmp/hand_scene_mm.stl", force="mesh"); pen = trimesh.load("/tmp/pen_scene_mm.stl", force="mesh")
    hinfo = json.loads(Path("/tmp/hand_scene_info.json").read_text())
    mount_c = np.array([hinfo["mount_xmax_mm"], hinfo["mount_centre_mm"][1], hinfo["mount_centre_mm"][2]])
    h = np.array([math.cos(TILT), 0, -math.sin(TILT)]); n = np.array([math.sin(TILT), 0, math.cos(TILT)])
    Rw = np.column_stack([-h, np.array([0, -1, 0]), n])
    flange_w = (T_JIG @ np.array([115.6, 44.8, 70.0, 1.0]))[:3]
    Th = np.eye(4); Th[:3, :3] = Rw; Th[:3, 3] = flange_w - Rw @ mount_c
    scene.add_geometry(colored(hand, (0.75, 0.75, 0.78, 1)), node_name="wuji_hand2", geom_name="wuji_hand2", transform=Th)
    scene.add_geometry(colored(pen, (0.95, 0.85, 0.2, 1)), node_name="pen", geom_name="pen", transform=Th)
    # wrist tag on top of the jig (50.4 mm, approximate position)
    tag = trimesh.creation.box((50.4, 50.4, 1.0)); top_z = (T_JIG @ np.array([150, 45, 35, 1]))[2]
    scene.add_geometry(colored(tag, (0.1, 0.1, 0.1, 1)), node_name="wrist_tag", geom_name="wrist_tag", transform=translate(flange_w[0] - 45, 100, top_z + 0.5))
    # D435i (90 x 25 x 25) hanging UNDER the plate on the user's bracket (18 mm), tilted 15 deg
    # towards the fingertips about the bracket hinge; the lenses face down-forward
    bracket = trimesh.creation.box((50, 40, 18))
    scene.add_geometry(colored(bracket, (0.1, 0.1, 0.1, 1)), node_name="bracket", geom_name="bracket",
                       transform=translate(PLATE[0], PLATE[1], PLATE[2] - 8 - 9))
    cam = trimesh.creation.box((90, 25, 25))     # long axis along X (the hand's length): the 87 deg IR field of view runs along it
    Tc = trimesh.transformations.rotation_matrix(math.radians(-10), [0, 1, 0])   # tilt: front face towards +X/down
    Tc = translate(PLATE[0] + 8, PLATE[1], PLATE[2] - 8 - 18 - 14) @ Tc
    scene.add_geometry(colored(cam, (0.85, 0.85, 0.85, 1)), node_name="d435i", geom_name="d435i", transform=Tc)
    return scene, Tc


def render(scene, name, eye, target, w=1600, h=1000, yfov=0.75, up=None):
    pr = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[1, 1, 1, 1])
    for node_name in scene.graph.nodes_geometry:
        T, geom_name = scene.graph.get(node_name)
        pr.add(pyrender.Mesh.from_trimesh(scene.geometry[geom_name], smooth=False), pose=T)
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    if up is None:
        up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([0, 1.0, 0])
    up = np.asarray(up, float)
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    pose = np.eye(4); pose[:3, 0] = r; pose[:3, 1] = u; pose[:3, 2] = -f; pose[:3, 3] = eye
    pr.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=w / h), pose=pose)
    pr.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
    lp = np.eye(4); lp[:3, 3] = [100, -300, 600]
    pr.add(pyrender.PointLight(color=np.ones(3), intensity=4e5), pose=lp)
    color, depth = pyrender.OffscreenRenderer(w, h).render(pr)
    from PIL import Image
    img = color.copy(); img[depth == 0] = 255
    Image.fromarray(img).save(REND / f"{name}.png")


def main():
    scene, Tc = build()
    scene.export(str(PARTS / "assembly.glb"))
    render(scene, "assembly_iso", eye=(-230, -420, 400), target=(175, 110, 150), yfov=0.70)
    render(scene, "assembly_front", eye=(620, 110, 230), target=(140, 110, 150), yfov=0.70)
    render(scene, "assembly_side", eye=(175, -560, 250), target=(175, 110, 160), yfov=0.70)
    render(scene, "assembly_top", eye=(175, 125, 800), target=(175, 125, 0), yfov=0.55)
    render(scene, "camera_plate_closeup", eye=(40, -60, 470), target=(135, 100, 340), yfov=0.5)
    # single parts
    for name, eye, tgt in [("breadboard_tile", (-120, -180, 160), (87, 62, 0)), ("splice_plate", (-40, -60, 50), (25, 25, 0)),
                           ("camera_column_base", (-220, -260, 220), (0, 0, 90)), ("camera_column_top", (-240, -300, 420), (-10, -50, 250)),
                           ("wuji_hand2_base", (-150, -180, 150), (75, 45, 40))]:
        single = trimesh.Scene(); single.add_geometry(colored(load(name), (0.35, 0.55, 0.85, 1)), node_name=name, geom_name=name)
        render(single, f"part_{name}", eye=eye, target=tgt, w=1000, h=700, yfov=0.7)
    # tile underside: nut pockets
    single = trimesh.Scene(); single.add_geometry(colored(load("breadboard_tile"), (0.35, 0.55, 0.85, 1)), node_name="t", geom_name="t")
    render(single, "part_breadboard_tile_underside", eye=(-80, -140, -150), target=(87, 62, 0), w=1000, h=700, yfov=0.7, up=(0, 0, -1))
    look = np.array([math.sin(math.radians(10)), 0, -math.cos(math.radians(10))])
    cam_pos = Tc[:3, 3] + 14 * look   # lenses on the down-facing front face
    # D435i IR imagers: 87 deg along the body's long axis (here world X, the hand's length), 58 deg across
    render(scene, "camera_view_d435i_ir", eye=cam_pos, target=cam_pos + 300 * look, w=1280, h=720, yfov=math.radians(58), up=(0, -1, 0))
    print("renders:", sorted(p.name for p in REND.glob("*.png")))
    print("camera at", np.round(cam_pos, 1), "fingertip region ~ (300, 100, 110); distance",
          round(float(np.linalg.norm(cam_pos - np.array([300, 100, 110]))), 0), "mm")


if __name__ == "__main__":
    main()
