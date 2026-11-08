# penspin-graph run report

## build.json

```json
{
  "seconds": 3.0236010551452637,
  "num_nodes": 25885,
  "num_episodes": 40,
  "feature_dim": 67,
  "blocks": [
    "qpos",
    "qvel",
    "pen_pos",
    "pen_axis",
    "pen_linvel",
    "pen_angvel",
    "fingertips"
  ],
  "block_weights": {
    "qpos": 1.0,
    "qvel": 0.4,
    "pen_pos": 2.0,
    "pen_axis": 2.0,
    "pen_rot6d": 0.0,
    "pen_linvel": 0.4,
    "pen_angvel": 0.4,
    "fingertips": 1.0,
    "links": 0.0
  },
  "step_unit": 0.5653595328330994,
  "k": 12,
  "window": 5,
  "max_cost": 6.0,
  "jumps": {
    "candidate": 57405,
    "kinematic_fail": 5403,
    "sim_rejected": 1197,
    "validated": 38529
  },
  "max_joint_vel": 5.810183683689825
}
```

## validation.json

```json
{
  "tested": 482,
  "validated": 99,
  "rejected": 383,
  "reasons": {
    "ok": 99,
    "end_pos_err": 104,
    "terminated:pen_position": 244,
    "end_axis_err": 5,
    "terminated:pen_axis": 30
  },
  "seconds": 20.545623540878296,
  "controller": "onnx",
  "config": {
    "controller": "onnx",
    "blend_frames": null,
    "follow_frames": 50,
    "success_pos_m": 0.03,
    "success_axis_deg": 29.999999999999996,
    "relative": true,
    "relative_factor": 1.5
  }
}
```

## loops_summary.json

```json
{
  "n_loops": 151,
  "largest_scc": 15215,
  "n_scc_gt1": 21,
  "verification": [
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 820,
      "summary": "ok: 820 ticks, pen pos err mean 1.2 cm (final 1.1), axis err mean 10 deg, joint err mean 0.609 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 827,
      "summary": "ok: 827 ticks, pen pos err mean 1.3 cm (final 1.1), axis err mean 13 deg, joint err mean 0.612 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 828,
      "summary": "ok: 828 ticks, pen pos err mean 1.2 cm (final 1.0), axis err mean 10 deg, joint err mean 0.609 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 831,
      "summary": "ok: 831 ticks, pen pos err mean 1.3 cm (final 0.9), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 829,
      "summary": "ok: 829 ticks, pen pos err mean 1.2 cm (final 1.0), axis err mean 10 deg, joint err mean 0.609 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 837,
      "summary": "ok: 837 ticks, pen pos err mean 1.4 cm (final 1.1), axis err mean 13 deg, joint err mean 0.616 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 835,
      "summary": "ok: 835 ticks, pen pos err mean 1.3 cm (final 1.0), axis err mean 12 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 841,
      "summary": "ok: 841 ticks, pen pos err mean 1.4 cm (final 1.1), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 841,
      "summary": "ok: 841 ticks, pen pos err mean 1.4 cm (final 1.1), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 841,
      "summary": "ok: 841 ticks, pen pos err mean 1.4 cm (final 1.1), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 841,
      "summary": "ok: 841 ticks, pen pos err mean 1.4 cm (final 1.1), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 841,
      "summary": "ok: 841 ticks, pen pos err mean 1.4 cm (final 1.1), axis err mean 13 deg, joint err mean 0.609 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 842,
      "summary": "ok: 842 ticks, pen pos err mean 1.4 cm (final 1.0), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 841,
      "summary": "ok: 841 ticks, pen pos err mean 1.5 cm (final 1.1), axis err mean 13 deg, joint err mean 0.610 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 849,
      "summary": "ok: 849 ticks, pen pos err mean 1.4 cm (final 1.0), axis err mean 13 deg, joint err mean 0.609 rad"
    },
    {
      "completed": true,
      "survived_fraction": 1.0,
      "ticks": 857,
      "summary": "ok: 857 ticks, pen pos err mean 1.5 cm (final 1.1), axis err mean 14 deg, joint err mean 0.614 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.3028169014084507,
      "ticks": 387,
      "summary": "FAILED@386 (pen_axis): 387 ticks, pen pos err mean 1.5 cm (final 3.8), axis err mean 17 deg, joint err mean 0.588 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.2914375490966222,
      "ticks": 371,
      "summary": "FAILED@370 (pen_axis): 371 ticks, pen pos err mean 1.5 cm (final 3.9), axis err mean 18 deg, joint err mean 0.583 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.29044684129429893,
      "ticks": 377,
      "summary": "FAILED@376 (pen_axis): 377 ticks, pen pos err mean 1.5 cm (final 3.6), axis err mean 18 deg, joint err mean 0.585 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.19953775038520802,
      "ticks": 259,
      "summary": "FAILED@258 (pen_axis): 259 ticks, pen pos err mean 1.5 cm (final 3.7), axis err mean 13 deg, joint err mean 0.618 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.1983663943990665,
      "ticks": 170,
      "summary": "FAILED@169 (pen_axis): 170 ticks, pen pos err mean 1.9 cm (final 1.7), axis err mean 17 deg, joint err mean 0.644 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.1956689868522815,
      "ticks": 253,
      "summary": "FAILED@252 (pen_axis): 253 ticks, pen pos err mean 1.6 cm (final 3.9), axis err mean 13 deg, joint err mean 0.616 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.1955177743431221,
      "ticks": 253,
      "summary": "FAILED@252 (pen_axis): 253 ticks, pen pos err mean 1.6 cm (final 3.9), axis err mean 13 deg, joint err mean 0.616 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.19415204678362574,
      "ticks": 166,
      "summary": "FAILED@165 (pen_axis): 166 ticks, pen pos err mean 1.9 cm (final 1.7), axis err mean 17 deg, joint err mean 0.648 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.19279907084785133,
      "ticks": 166,
      "summary": "FAILED@165 (pen_axis): 166 ticks, pen pos err mean 1.9 cm (final 1.7), axis err mean 17 deg, joint err mean 0.648 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.1925754060324826,
      "ticks": 166,
      "summary": "FAILED@165 (pen_axis): 166 ticks, pen pos err mean 1.9 cm (final 1.7), axis err mean 17 deg, joint err mean 0.648 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.16281895504252733,
      "ticks": 134,
      "summary": "FAILED@133 (pen_axis): 134 ticks, pen pos err mean 2.0 cm (final 1.8), axis err mean 18 deg, joint err mean 0.668 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.16281895504252733,
      "ticks": 134,
      "summary": "FAILED@133 (pen_axis): 134 ticks, pen pos err mean 2.0 cm (final 1.8), axis err mean 18 deg, joint err mean 0.668 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.16262135922330098,
      "ticks": 134,
      "summary": "FAILED@133 (pen_axis): 134 ticks, pen pos err mean 2.0 cm (final 1.8), axis err mean 18 deg, joint err mean 0.668 rad"
    },
    {
      "completed": false,
      "survived_fraction": 0.1614457831325301,
      "ticks": 134,
      "summary": "FAILED@133 (pen_axis): 134 ticks, pen pos err mean 2.0 cm (final 1.8), axis err mean 18 deg, joint err mean 0.668 rad"
    }
  ],
  "verify_rounds": 5,
  "jumps": {
    "candidate": 56923,
    "kinematic_fail": 5403,
    "sim_rejected": 1584,
    "validated": 38624
  },
  "best": {
    "frames": 752,
    "jumps": 9,
    "rotation_deg": 2359.963761800727,
    "episodes": [
      "02_continuous_single_axis/clip_003",
      "03_continuous_multi_axis/clip_002",
      "03_continuous_multi_axis/clip_003"
    ]
  }
}
```

## eval_graph_policy.json

```json
{
  "summary": "20 starts: survived mean 9.55s (min 1.08s), pen rotation mean 381 deg, drop rate 5%, jumps taken mean 2.1, demos visited mean 3.5",
  "controller": "onnx",
  "horizon": 500
}
```

## bc/eval.json

```json
{
  "summary": "20 starts: survived mean 3.58s (min 1.34s), pen rotation mean 811 deg, drop rate 55%, jumps taken mean 0.0, demos visited mean 1.0",
  "samples": 30323,
  "walks": 200,
  "epochs": 40,
  "hidden": [
    512,
    512,
    256
  ]
}
```
