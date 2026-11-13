#!/usr/bin/env python3
"""Printable rig for the Wuji Hand 2 pen-spinning setup (CadQuery, all mm).

Parts (STL + STEP written next to this file, into hardware/parts/):

  breadboard_tile      4x   175 x 125 x 12, M6 grid at 25 mm pitch (12.5 mm margin),
                            hex nut pocket under every hole -> a 350 x 250 "breadboard"
  splice_plate         5x   50 x 50 x 8 under the tile seams, 4 holes, nut pockets
  camera_column_base   1x   100 x 100 foot (4x M6 on the grid) + 45 x 45 column, lap joint on top
  camera_column_top    1x   column + arm over the hand + plate with the D435i bracket pattern
  wuji_hand2_base      1x   Wuji's own jig (asset bundle hardware/hand2/base.obj, Apache-2.0),
                            re-exported in mm; its 4 x M6 feet sit on the 25 mm grid

Coordinates: X along the hand (wrist -> fingertips), Y across, Z up, origin at the
breadboard's top-surface corner.  Hole grid: X = 12.5 + 25 i, Y = 12.5 + 25 j.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cadquery as cq
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts"
OUT.mkdir(exist_ok=True)

PITCH = 25.0
MARGIN = 12.5
TILE_X, TILE_Y, TILE_T = 175.0, 125.0, 12.0          # 4 tiles -> 350 x 250
M6_CLEAR = 6.6
M6_NUT_AF = 10.0 + 0.4                              # across flats + clearance
M6_NUT_T = 5.2 + 0.6
M3_CLEAR = 3.4
M3_NUT_AF = 5.5 + 0.4
M3_NUT_T = 2.4 + 0.6

# camera mount geometry
FOOT, FOOT_T = 100.0, 10.0
COL = 45.0
COL_BASE_H = 200.0          # foot bottom -> top of the base column
LAP = 60.0                  # overlap of the two column halves
COL_TOP_Z = 360.0           # top of the column; the camera hangs ~35 mm below the plate top
ARM_T = 25.0
ARM_W = 45.0
PLATE_X, PLATE_Y, PLATE_T = 60.0, 50.0, 8.0
# D435i bracket pattern, centre-to-centre (the user measured outer-edge distances with 3 mm holes)
BRK_LONG = 40.0 - 3.0       # 37 mm
BRK_SHORT = 17.0 - 3.0      # 14 mm
BRK_H = 25.4 - 3.0          # 22.4 mm between the two rows
BRK_SLOT = 1.5              # +-1.5 mm radial slack in every hole

# layout on the breadboard (hole indices)
JIG_BOLTS_X = (37.5, 87.5)          # 50 mm apart along the hand
JIG_BOLTS_Y = (62.5, 137.5)         # 75 mm apart across
COL_CENTRE = (162.5, 187.5)         # column axis; foot bolts at +-25 around it
PLATE_CENTRE_Y = 100.0              # camera plate centred over the hand (hand centre Y = 100)
PLATE_OFFSET_X = -32.5              # plate centre at X = 130: the IR imager then sees the wrist tag (X~60) and the pen (X 230-380)


def grid_x(n):
    return [MARGIN + PITCH * i for i in range(n)]


def hex_pocket(af: float, depth: float):
    """Hexagonal prism (across-flats af) of height depth, centred at origin, from z=0 up."""
    r = af / math.sqrt(3.0)  # circumradius
    return cq.Workplane("XY").polygon(6, 2 * r).extrude(depth)


def breadboard_tile() -> cq.Workplane:
    tile = cq.Workplane("XY").box(TILE_X, TILE_Y, TILE_T, centered=(False, False, False))
    tile = tile.edges("|Z").fillet(4.0)
    pts = [(x, y) for x in grid_x(7) for y in grid_x(5)]
    tile = tile.faces(">Z").workplane().pushPoints(pts).hole(M6_CLEAR)
    # nut pockets from below
    for x, y in pts:
        tile = tile.cut(hex_pocket(M6_NUT_AF, M6_NUT_T).translate((x, y, 0)))
    # 1 mm chamfer on the top edge of every hole helps the bolt find the hole; skipped (printers vary)
    return tile


def splice_plate() -> cq.Workplane:
    s = 50.0
    p = cq.Workplane("XY").box(s, s, 8.0, centered=(False, False, False)).edges("|Z").fillet(4.0)
    pts = [(12.5, 12.5), (37.5, 12.5), (12.5, 37.5), (37.5, 37.5)]
    p = p.faces(">Z").workplane().pushPoints(pts).hole(M6_CLEAR)
    for x, y in pts:
        p = p.cut(hex_pocket(M6_NUT_AF, M6_NUT_T).translate((x, y, 0)))
    return p


def lap_bolt_z():
    return [COL_BASE_H - LAP + 15.0, COL_BASE_H - 15.0]


def camera_column_base() -> cq.Workplane:
    foot = (cq.Workplane("XY").box(FOOT, FOOT, FOOT_T, centered=(True, True, False))
            .edges("|Z").fillet(8.0))
    foot = foot.faces(">Z").workplane().pushPoints([(-25, -25), (25, -25), (-25, 25), (25, 25)]).hole(M6_CLEAR)
    col = cq.Workplane("XY").box(COL, COL, COL_BASE_H - FOOT_T, centered=(True, True, False)).translate((0, 0, FOOT_T))
    # lap joint: keep only the +X half of the column over the top LAP mm
    cut = cq.Workplane("XY").box(COL / 2 + 0.2, COL + 2, LAP + 1, centered=(False, True, False)).translate((-COL / 2 - 0.2, 0, COL_BASE_H - LAP))
    col = col.cut(cut)
    body = foot.union(col)
    # gussets between foot and column
    for sx in (-1, 1):
        g = (cq.Workplane("YZ").polyline([(COL / 2, FOOT_T), (COL / 2 + 25, FOOT_T), (COL / 2, FOOT_T + 40)]).close()
             .extrude(8.0, both=True))
        body = body.union(g.mirror("XZ") if sx < 0 else g)
    for sy in (-1, 1):
        g = (cq.Workplane("XZ").polyline([(COL / 2, FOOT_T), (COL / 2 + 25, FOOT_T), (COL / 2, FOOT_T + 40)]).close()
             .extrude(8.0, both=True))
        body = body.union(g.mirror("YZ") if sy < 0 else g)
    # two horizontal M6 holes through the lap (along X)
    for z in lap_bolt_z():
        body = body.cut(cq.Workplane("YZ").circle(M6_CLEAR / 2).extrude(COL + 2, both=True).translate((0, 0, z)))
    # counterbore recess on the -X side of the lap would weaken it; bolts use washers instead
    return body


def camera_column_top() -> cq.Workplane:
    z0 = COL_BASE_H - LAP
    col = cq.Workplane("XY").box(COL, COL, COL_TOP_Z - z0, centered=(True, True, False)).translate((0, 0, z0))
    # lap joint: this half keeps the -X side over the lower LAP mm
    cut = cq.Workplane("XY").box(COL / 2 + 0.2, COL + 2, LAP + 1, centered=(False, True, False)).translate((-0.0, 0, z0 - 1))
    col = col.cut(cut)
    for z in lap_bolt_z():
        col = col.cut(cq.Workplane("YZ").circle(M6_CLEAR / 2).extrude(COL + 2, both=True).translate((0, 0, z)))
    # arm towards -Y (over the hand), top flush with the column top
    reach = COL_CENTRE[1] - PLATE_CENTRE_Y            # 87.5 mm from column axis to plate centre
    arm_len = reach - PLATE_Y / 2 + COL / 2 + 10      # overlap the plate by 10 mm
    arm = (cq.Workplane("XY").box(ARM_W, arm_len, ARM_T, centered=(True, False, False))
           .translate((0, -(arm_len) + COL / 2, COL_TOP_Z - ARM_T)))
    # diagonal brace under the arm
    brace = (cq.Workplane("YZ").polyline([(-COL / 2, COL_TOP_Z - ARM_T), (-COL / 2 - 60, COL_TOP_Z - ARM_T),
                                          (-COL / 2, COL_TOP_Z - ARM_T - 70)]).close().extrude(ARM_W / 2, both=True))
    body = col.union(arm).union(brace)
    # camera plate at the arm end, centred at (PLATE_OFFSET_X, -reach), top at COL_TOP_Z
    plate = (cq.Workplane("XY").box(PLATE_X, PLATE_Y, PLATE_T, centered=(True, True, False))
             .edges("|Z").fillet(5.0).translate((PLATE_OFFSET_X, -reach, COL_TOP_Z - PLATE_T)))
    body = body.union(plate)
    # rib under the plate, from the arm to the plate's far end (the plate cantilevers towards -X)
    rib = (cq.Workplane("XY").box(PLATE_X + abs(PLATE_OFFSET_X) - 5, 10.0, 22.0, centered=(False, True, False))
           .translate((PLATE_OFFSET_X - PLATE_X / 2 + 3, -reach, COL_TOP_Z - PLATE_T - 22.0)))
    body = body.union(rib)
    # bracket pattern: trapezoid, long pair towards the column (+Y), short pair outboard (-Y)
    cx, cy = PLATE_OFFSET_X, -reach
    holes = [(cx - BRK_LONG / 2, cy + BRK_H / 2), (cx + BRK_LONG / 2, cy + BRK_H / 2),
             (cx - BRK_SHORT / 2, cy - BRK_H / 2), (cx + BRK_SHORT / 2, cy - BRK_H / 2)]
    for hx, hy in holes:
        ang = math.degrees(math.atan2(hy - cy, hx - cx))
        slot = (cq.Workplane("XY").slot2D(2 * BRK_SLOT + M3_CLEAR, M3_CLEAR, angle=ang)
                .extrude(PLATE_T + 2).translate((hx, hy, COL_TOP_Z - PLATE_T - 1)))
        body = body.cut(slot)
        nut = (cq.Workplane("XY").slot2D(2 * BRK_SLOT + M3_NUT_AF, M3_NUT_AF, angle=ang)
               .extrude(M3_NUT_T).translate((hx, hy, COL_TOP_Z - PLATE_T)))
        body = body.cut(nut)
    # spare 1/4"-20 tripod clearance in the plate centre
    body = body.cut(cq.Workplane("XY").circle(6.6 / 2).extrude(PLATE_T + 2).translate((cx, cy, COL_TOP_Z - PLATE_T - 1)))
    return body


def wuji_jig_transform():
    """Rotation + translation that puts Wuji's base.obj (mm) on the breadboard.

    In the mesh the mounting face has normal (-cos10, 0, sin10) and the hand flange
    faces +z. World: mounting face down, hand towards +X, feet on the 25 mm grid."""
    import trimesh

    m = trimesh.load(str(HERE / "parts" / "wuji_hand2_base.stl"))
    t = math.radians(10.0)
    e1 = np.array([math.sin(t), 0.0, math.cos(t)])     # -> +X (hand direction)
    e2 = np.array([0.0, -1.0, 0.0])                     # -> +Y
    e3 = np.cross(e1, e2)                               # -> +Z (normal of the mounting face, pointing into the part)
    R = np.vstack([e1, e2, e3])                         # world = R @ mesh
    v = (R @ m.vertices.T).T
    # mounting face plane: n_mesh . p = 1.4  ->  world z of that face
    z_face = v[:, 2].min()
    # bolt hole axes (mesh): through (x=12, y=7.5, z=20.9) etc., direction = e3 in mesh = -n_mesh
    holes_mesh = np.array([[12, 7.5, 20.91], [12, 82.5, 20.91], [12, 7.5, 71.7], [12, 82.5, 71.7]])
    hw = (R @ holes_mesh.T).T
    cx, cy = hw[:, 0].mean(), hw[:, 1].mean()
    tx = (JIG_BOLTS_X[0] + JIG_BOLTS_X[1]) / 2 - cx
    ty = (JIG_BOLTS_Y[0] + JIG_BOLTS_Y[1]) / 2 - cy
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, -z_face]
    hw2 = (T @ np.c_[holes_mesh, np.ones(4)].T).T[:, :2]
    return T, hw2


def export(name: str, wp: cq.Workplane):
    cq.exporters.export(wp, str(OUT / f"{name}.stl"), tolerance=0.05, angularTolerance=0.1)
    cq.exporters.export(wp, str(OUT / f"{name}.step"))
    bb = wp.val().BoundingBox()
    vol = wp.val().Volume()
    print(f"{name:22s} {bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f} mm  {vol / 1000:.0f} cm3  ~{vol / 1000 * 1.24 * 0.35:.0f} g at 20% infill")
    return dict(name=name, size_mm=[round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1)], volume_cm3=round(vol / 1000, 1))


def main():
    import trimesh

    src = HERE.parents[0] / "third_party" / "wuji_hand2_base.obj"
    if not src.is_file():
        src = Path("/home/claude/ext/bundle/wuji-mjlab-v2026.10.22-assets/hardware/hand2/base.obj")
    jig = trimesh.load(str(src), force="mesh")
    if jig.extents.max() < 1.0:
        jig.apply_scale(1000.0)
    jig.apply_translation(-jig.bounds[0])
    jig.export(str(OUT / "wuji_hand2_base.stl"))

    info = []
    info.append(export("breadboard_tile", breadboard_tile()))
    info.append(export("splice_plate", splice_plate()))
    info.append(export("camera_column_base", camera_column_base()))
    info.append(export("camera_column_top", camera_column_top()))
    T, holes = wuji_jig_transform()
    print("Wuji jig feet land on the grid at:", np.round(holes, 1).tolist())
    np.save(OUT / "wuji_jig_world_transform.npy", T)
    (OUT / "parts.json").write_text(json.dumps(dict(parts=info, jig_transform=T.tolist(), jig_feet_xy=holes.tolist(),
                                                     column_centre=COL_CENTRE, plate_centre=[COL_CENTRE[0] + PLATE_OFFSET_X, PLATE_CENTRE_Y, COL_TOP_Z]), indent=2))


if __name__ == "__main__":
    main()
