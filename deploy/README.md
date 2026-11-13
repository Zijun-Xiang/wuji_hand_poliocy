# Deploying a motion-graph loop on the real Wuji Hand 2

The loop runs through Wuji's **official** deploy runtime (`wuji-mjlab/deploy/pen_spin`,
the `pen-policy` command). Nothing that touches the hardware is changed: hand driver,
camera stream, safety gates (FROZEN → HOLD → POLICY, placement check, Ctrl+C eases the
hand open) and the action mapping are Wuji's code. `pen_policy_graph.py` adds one thing:
every frame is driven by the released policy of the motion group that frame came from,
because a stitched loop mixes groups and one policy for the whole loop drops the pen
within 5 s (`runs/official_env/90_loop_pol02.json`, `90_loop_pol03.json`).

## What you need (same as Wuji's own deployment)

| Item | Why | Without it |
|---|---|---|
| Wuji Hand 2 (right), USB or UDP | the hand | nothing runs |
| Hikrobot camera + MVS SDK, 250 mm AprilTag pen, wrist tag | the policy's 6 numbers of pen pose | **the policy drops the pen within seconds** — measured in Wuji's simulator with the reference pen pose fed instead of the real one: 12 of 13 official clips fail in 0.7–7 s and our loop at 1.9 s (`runs/official_env/02_objref.json`, `03_objref.json`, `90_loop_objref.json`) |
| Linux host with `pixi` (Wuji's instructions) — or Windows with Python ≥ 3.11 and `wuji_sdk==2026.11.9` (first release with a Windows wheel; same `SdkManager / ConnectOptions / MitParam / JointCommand` API the adapter uses) | runs the runtime | — |

`--obj-from-ref` and `--no-hardware` exist for dry runs only. `--obj-from-ref` on the real
hand is not a camera-less deployment; it is the experiment above.

## Files

| File | Content |
|---|---|
| `pen_policy_graph.py` | launcher: `pen-policy` + per-frame policy switching |
| `clips/graph_loop_1lap.npz` (+ `.groups.json`) | the best loop, one lap, 821 frames = 16.4 s, official clip schema |
| `clips/graph_loop_20laps.npz` (+ `.groups.json`) | the same loop repeated 20 times, 328 s — the run that was verified in Wuji's simulator (`runs/official_env/90_loop.json`: never dropped, worst pen error 2.4 cm, worst shaft error 26.6°) |

Both clips start at frame 0 of `02_continuous_single_axis/clip_003`, so the first-frame
hold and the pen placement check are the official ones. The loop needs the policies of
groups **02** and **03** from Wuji's `v2026.10.22` asset bundle.

## Steps

1. Set up wuji-mjlab deployment exactly as in
   [`src/wuji_mjlab/tasks/pen_spin/docs/deploy.md`](https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/pen_spin/docs/deploy.md)
   (`pixi install -e pen-spin-deploy`, checkpoints copied into
   `src/wuji_mjlab/tasks/pen_spin/data/checkpoints/`, camera calibrated).
2. **Dry run without hardware** (checks the files, the policies and the frame → policy map):
   ```bash
   cd <wuji-mjlab>
   pixi run -e pen-spin-deploy python <penspin-graph>/deploy/pen_policy_graph.py \
     --motion <penspin-graph>/deploy/clips/graph_loop_1lap.npz \
     --policy-map 02_continuous_single_axis=src/wuji_mjlab/tasks/pen_spin/data/checkpoints/02_continuous_single_axis/policy.onnx \
                  03_continuous_multi_axis=src/wuji_mjlab/tasks/pen_spin/data/checkpoints/03_continuous_multi_axis/policy.onnx \
     --no-hardware --no-sim --obj-from-ref < /dev/null
   ```
   Expected: `all checks passed`, the clip plays through, `policy switches during the run: 2`.
3. **Camera first** (terminal 1), as in Wuji's guide: `pixi run -e pen-spin-deploy pen-observer --preview`.
   Wait until the wrist tag is sampled and the pen wireframe follows the real pen.
4. **Verify Wuji's own clip on the hand first** (terminal 2) — this proves the rig, not our code:
   `pixi run -e pen-spin-deploy pen-policy --policy .../02_continuous_single_axis/policy.onnx --motion .../02_continuous_single_axis/clip_003.npz`
5. **Run the loop** (terminal 2), one lap first, then 20 laps:
   ```bash
   pixi run -e pen-spin-deploy python <penspin-graph>/deploy/pen_policy_graph.py \
     --motion <penspin-graph>/deploy/clips/graph_loop_1lap.npz \
     --policy-map 02_continuous_single_axis=.../02_continuous_single_axis/policy.onnx \
                  03_continuous_multi_axis=.../03_continuous_multi_axis/policy.onnx
   ```
   The hand eases to the start pose and freezes. Place the pen as in frame 0 of
   `02/clip_003` (the mirror shows the translucent reference), press **Enter** once
   (the policy holds the first frame), check the placement line, press **Enter** again
   to start. `r` opens the hand, **Ctrl+C** aborts. Keep a hand on the power switch:
   Ctrl+C is not an emergency stop.

## Pen placement at the start

The loop starts at frame 0 of `02_continuous_single_axis/clip_003`. `start_pose_02_clip_003.png`
shows it from three sides; `start_pose_02_clip_003.json` has the numbers. In the palm frame the
pen's centre is at (1.2, 7.4, −9.5) cm and its shaft points along (0.84, 0.09, −0.53): the pen lies
across the fingertips and is pinched between the thumb, index and middle fingertips (each 2.0–2.6 cm
from the pen axis); the ring and little fingers are 5.7 and 7.8 cm away. The official start gate
accepts a pen within 5 cm and 60° of this pose, so "lying across the three fingertips, roughly
level" is enough; the MuJoCo mirror shows the translucent reference pen for fine adjustment.

## Camera without the Hikrobot rig: RealSense + printed markers

The policy's pen pose comes only from Wuji's markers, so a plain pen cannot be tracked by any
camera. What is needed, in order of effort:

1. **Markers.** `markers/wuji_pen_spin_markers_A4.pdf` prints at 100 % scale: the wrist AprilTag
   36h11 ID 0 (50.4 mm black edge) and the ten faces of the two 18 × 18 × 36 mm pen end blocks
   (Wuji's sticker fallback). Better: 3D-print `pen_heads.3mf` + a shaft from Wuji's asset bundle.
   Pen length 250 mm, 178 mm exposed shaft, core diameter 14.9 mm.
2. **Camera.** `pen_observer_realsense.py` runs Wuji's unmodified observer on a RealSense colour
   stream with the device's factory intrinsics (`deploy/windows/pen_observer_realsense.bat`;
   installs opencv-contrib-python, pyzmq, pyrealsense2 into `.venv-deploy`). A RealSense lens is
   about 2× wider than Wuji's, so place it 30–40 cm from the pen and start at 1280 × 720 @ 30 fps.
   Not yet run on hardware.
3. Then the official two-terminal flow above: observer first, `pen_policy_graph.py` second.

## No camera yet? Play the loop's joint motion on the hand

`play_reference.py` sends the loop's reference joint angles to the hand at 50 Hz
through Wuji's own deploy-side driver, gains and easing (`runtime/hand.py`,
`lib/hand2_adapter.py`, `wuji_sdk`), without a camera and without the policy. It shows
the stitched motion on the real hand, including the blends between demos, and reports
the joint tracking error. It does **not** spin a pen (see the table above). It runs on
Windows too: only the terminal handling of Wuji's runtime is replaced.

```bat
deploy\windows\play_loop.bat                    :: first run: creates .venv-deploy, installs wuji_sdk 2026.11.9,
                                                 :: clones wuji-mjlab into .deps\ if no sibling checkout exists;
                                                 :: then plays one lap at half speed
deploy\windows\play_loop.bat --no-hardware      :: dry run with Wuji's stub hand
deploy\windows\play_loop.bat --speed 1.0 --laps 3
```

Linux: `python deploy/play_reference.py --wuji-root <wuji-mjlab> --motion deploy/clips/graph_loop_1lap.npz --speed 0.5`.
The hand eases to frame 0 and waits; **clear the area**, press Enter; Ctrl+C eases it open.

## What is still untested

Only the simulator runs above exist. The loop has **not** been run on a physical hand
yet, and the camera-based pen pose on a real rig has latency and dropouts that the
official policies were trained for but the stitched blends were not. Start with one lap.
