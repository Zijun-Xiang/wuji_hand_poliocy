"""Headless evaluation in Wuji's OFFICIAL pen-spin environment (wuji-mjlab, mjlab + MuJoCo Warp).

Nothing in the environment or the policy is re-implemented here: the env
config comes from wuji-mjlab's own `prepare_task_cfgs(..., play=True)` (the
deploy-like play setting: 1 s first-frame hold, no domain randomization), and
the policy is the released `model.pt`, loaded exactly as wuji-mjlab's play
runner loads it.

Only measurement is added, outside the simulation:
  * env i is pinned to clip i of the recipe, starting at frame 0 (play samples
    clips at random);
  * Wuji's training failure test is evaluated every step, because play mode
    removes it: pen position > 5 cm or pen shaft >= 60 deg away from the
    reference, both in the palm frame (wuji-mjlab's `object_pose_wrist_local`),
    against the reference frame the step was tracking -- the same quantities
    and the same frame that `object_position_error_exceeded` /
    `object_shaft_direction_error_exceeded` see during training;
  * pen-shaft rotation (sim and reference) and errors are logged until the
    first failure or the clip end.

Optional: `--policy-map` runs, per frame, the released policy of the motion
group that frame was taken from (for stitched clips that mix groups).

Usage: python eval_official.py --recipe <recipe.json> --checkpoint <model.pt> --out result.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

import wuji_mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_runner_cls
from wuji_mjlab.tasks.pen_spin import mdp
from wuji_mjlab.utils.cli_override_utils import DEFAULT_TASK_ALIASES, resolve_task
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

PALM_LINK = "right_palm_link"


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by wxyz quaternions q."""
    qv = q[:, 1:]
    uv = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (q[:, :1] * uv + torch.cross(qv, uv, dim=-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--checkpoint", default=None, help="one released model.pt for every clip")
    ap.add_argument("--policy-map", nargs="+", default=None, metavar="GROUP=MODEL_PT",
                    help="per-frame policy: frames whose source group is GROUP run that group's model.pt")
    ap.add_argument("--frame-groups", default=None, help="loop_clips.json written by export_loop_official.py")
    ap.add_argument("--zero-policy", action="store_true", help="sanity check: zero residual (open-loop reference replay)")
    ap.add_argument("--trace", default=None, help="save per-step errors and rotation to this .npz")
    ap.add_argument("--obj-from-ref", action="store_true",
                    help="camera-less deployment check: the policy sees the REFERENCE pen pose instead of the simulated one "
                         "(what wuji-mjlab's deploy runtime does with --obj-from-ref)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=None, help="cap each clip (after the 1 s hold)")
    ap.add_argument("--episode-length-s", type=float, default=None, help="raise the env's time limit for long clips")
    args = ap.parse_args()

    torch.manual_seed(0)
    task_id = resolve_task("WujiHand2_PenSpin", task_aliases=DEFAULT_TASK_ALIASES, all_tasks=list_tasks())
    env_cfg, agent_cfg = prepare_task_cfgs(task_id, [f"env.commands.motion.recipe_path={args.recipe}"], play=True)
    recipe_groups = json.loads(Path(args.recipe).read_text())["groups"]
    clips_root = Path(wuji_mjlab.tasks.__file__).parent / "pen_spin" / "data" / "rawdata" / "clips"
    n_clips = sum(json.loads((clips_root / g / "index.json").read_text())["clip_count"] for g in recipe_groups)
    env_cfg.scene.num_envs = n_clips
    if args.episode_length_s:
        env_cfg.episode_length_s = args.episode_length_s

    if args.obj_from_ref:
        from wuji_mjlab.tasks.pen_spin.mdp.observations import _axis_points

        def _ref_object_points(env, robot_cfg=None, object_cfg=None, command_name="motion"):
            pose = env.command_manager.get_term(command_name).object_pose
            return _axis_points(env, pose, command_name, bounded=True)

        env_cfg.observations["policy"].terms["object_pose"].func = _ref_object_points
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    cmd = env.command_manager.get_term("motion")
    hold = round(1.0 / cmd.group.frame_dt)

    # pin env i -> clip i, starting at frame 0 after the official 1 s first-frame hold
    original_prepare = cmd.prepare_reset

    def prepare_reset(env_ids: torch.Tensor) -> None:
        original_prepare(env_ids)
        cmd.trajectory_ids[env_ids] = env_ids % cmd.group.num_trajectories
        cmd.time_steps[env_ids] = -hold
        cmd._episode_start_steps[env_ids] = cmd.time_steps[env_ids]

    cmd.prepare_reset = prepare_reset

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner

    def load_policy(path: str):
        # exactly what wuji-mjlab's play runner does
        runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
        runner.load(path, load_cfg={"actor": True}, strict=True, map_location="cpu")
        return runner.get_inference_policy(device="cpu")

    if args.policy_map:
        pairs = [x.split("=", 1) for x in args.policy_map]
        group_names = [k for k, _ in pairs]
        policies = [load_policy(v) for _, v in pairs]
        meta = {m["clip_id"]: m for m in json.loads(Path(args.frame_groups).read_text())}
        clip_order = [c["clip_id"] for g in recipe_groups for c in json.loads((clips_root / g / "index.json").read_text())["clips"]]
        frame_policy = torch.as_tensor([group_names.index(gid) for c in clip_order for gid in meta[c]["frame_groups"]], dtype=torch.long)
        rows_idx = torch.arange(env.num_envs)

        def policy(o):
            acts = torch.stack([p(o) for p in policies])  # (P, N, A)
            return acts[frame_policy[cmd.frame_ids], rows_idx]
    elif args.zero_policy:
        def policy(o):
            return torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    else:
        policy = load_policy(args.checkpoint)
    obs, _ = wrapped.reset()

    palm = SceneEntityCfg("robot", body_names=(PALM_LINK,))
    palm.resolve(env.scene)
    pen = SceneEntityCfg("object")
    pen.resolve(env.scene)

    N = env.num_envs
    clip_ids = cmd.trajectory_ids.clone()
    lengths = cmd.trajectory_lengths.clone()
    ordered = [f"{g}/{c['clip_id']}" for g in recipe_groups for c in json.loads((clips_root / g / "index.json").read_text())["clips"]]
    names = [ordered[int(i)] for i in clip_ids]
    active = torch.ones(N, dtype=torch.bool)
    failed = torch.zeros(N, dtype=torch.bool)
    fail_reason = [""] * N
    tracked = torch.zeros(N, dtype=torch.long)  # control ticks after the 1 s hold
    rot_sim = torch.zeros(N)
    rot_ref = torch.zeros(N)
    pos_err_sum = torch.zeros(N)
    shaft = torch.zeros(N, 3)
    shaft[:, 0] = 1.0
    max_ticks = None if args.max_seconds is None else round(args.max_seconds / cmd.group.frame_dt)

    def wrist_local_shaft(pose):
        return mdp.quat_apply(pose[:, 3:7], shaft) if hasattr(mdp, "quat_apply") else _quat_apply(pose[:, 3:7], shaft)

    prev_sim = wrist_local_shaft(mdp.object_pose_wrist_local(env, palm, pen))
    prev_ref = wrist_local_shaft(cmd.object_pose)
    trace = {"pos_err": [], "ax_err": [], "rot_sim": [], "rot_ref": [], "spinning": []}
    t0 = time.time()
    step = 0
    with torch.inference_mode():
        while active.any():
            ts_before = cmd.time_steps.clone()
            ref_before = cmd.object_pose.clone()  # the frame the policy is tracking during this step
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            step += 1
            done = dones.bool().reshape(-1)
            alive = active & ~done
            # same quantities as Wuji's terminations, evaluated against the frame tracked in this step
            cur = mdp.object_pose_wrist_local(env, palm, pen)
            pos_err = torch.linalg.norm(cur[:, :3] - ref_before[:, :3], dim=-1)
            n_sim = wrist_local_shaft(cur)
            n_ref = wrist_local_shaft(ref_before)
            ax_err = torch.arccos(torch.clamp((n_sim * n_ref).sum(-1), -1, 1))
            spinning = alive & (ts_before >= 0)
            rot_sim[spinning] += torch.arccos(torch.clamp((prev_sim * n_sim).sum(-1), -1, 1))[spinning]
            rot_ref[spinning] += torch.arccos(torch.clamp((prev_ref * n_ref).sum(-1), -1, 1))[spinning]
            prev_sim, prev_ref = n_sim, n_ref
            tracked[spinning] += 1
            pos_err_sum[spinning] += pos_err[spinning]
            bad = alive & ((pos_err > 0.05) | (ax_err >= math.radians(60.0)))
            for i in torch.nonzero(bad).flatten().tolist():
                failed[i] = True
                fail_reason[i] = ("pen_position>5cm" if pos_err[i] > 0.05 else "shaft>=60deg") + (" (during hold)" if ts_before[i] < 0 else "")
            early = alive.new_zeros(N) | (active & done & (ts_before + 1 < lengths))
            for i in torch.nonzero(early & ~bad).flatten().tolist():
                failed[i] = True
                fail_reason[i] = "env_terminated_before_clip_end"
            finished = active & (done | (ts_before + 1 >= lengths - 1))
            if max_ticks is not None:
                finished |= active & (tracked >= max_ticks)
            if args.trace:
                trace["pos_err"].append(pos_err.clone()); trace["ax_err"].append(ax_err.clone())
                trace["rot_sim"].append(rot_sim.clone()); trace["rot_ref"].append(rot_ref.clone())
                trace["spinning"].append(spinning.clone())
            active &= ~(bad | finished)
            if step % 100 == 0:
                print(f"  step {step}: {int(active.sum())} clips still running ({time.time() - t0:.0f}s)", flush=True)

    dt = cmd.group.frame_dt
    rows = []
    for i in range(N):
        rows.append({
            "clip": names[i],
            "clip_index": int(clip_ids[i]),
            "clip_s": float(lengths[i]) * dt,
            "completed": not bool(failed[i]),
            "fail_reason": fail_reason[i],
            "tracked_s": float(tracked[i]) * dt,
            "rotation_sim_deg": math.degrees(float(rot_sim[i])),
            "rotation_ref_deg": math.degrees(float(rot_ref[i])),
            "mean_pos_err_cm": 100 * float(pos_err_sum[i]) / max(int(tracked[i]), 1),
        })
        r = rows[-1]
        print(f"{r['clip']:>10s}: {'OK  ' if r['completed'] else 'FAIL'} tracked {r['tracked_s']:5.1f}/{r['clip_s']:5.1f}s "
              f"rot sim {r['rotation_sim_deg']:6.0f} ref {r['rotation_ref_deg']:6.0f} err {r['mean_pos_err_cm']:.2f}cm {r['fail_reason']}")
    if args.trace:
        import numpy as np
        np.savez(args.trace, names=np.array(names), **{k: torch.stack(v).numpy() for k, v in trace.items()})
    out = {"recipe": args.recipe, "checkpoint": args.checkpoint, "policy_map": args.policy_map, "zero_policy": args.zero_policy, "device": "cpu", "seconds": time.time() - t0,
           "steps": step, "clips": rows}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote", args.out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
