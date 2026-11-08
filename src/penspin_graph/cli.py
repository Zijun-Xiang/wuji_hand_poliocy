"""Command line interface: ``penspin-graph <command> [options]``.

Typical session (all commands share ``--run runs/<name>``)::

    penspin-graph build      --run runs/demo                  # embed frames, propose jumps, kinematic filter
    penspin-graph validate   --run runs/demo --workers 4      # MuJoCo + Wuji tracking policy
    penspin-graph loops      --run runs/demo                  # closed walks through validated jumps
    penspin-graph eval       --run runs/demo --n-starts 20    # closed-loop graph policy from random states
    penspin-graph export     --run runs/demo                  # stitched clips for wuji-mjlab PPO
    penspin-graph train-bc   --run runs/demo                  # distil into a reference-free MLP
    penspin-graph report     --run runs/demo                  # markdown summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .data import load_clip_groups, summarize
from .features import DEFAULT_BLOCK_WEIGHTS, FrameFeaturizer
from .graph import JumpStatus, MotionGraph
from .graph.kinematics import KinematicLimits, apply_kinematic_filter
from .graph.loops import Loop, find_loops, strongly_connected

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIPS = REPO_ROOT / "data" / "wuji_pen_spin_clips"


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _run_dir(args) -> Path:
    d = Path(args.run)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_graph(args) -> MotionGraph:
    run = _run_dir(args)
    episodes = load_clip_groups(args.clips, groups=args.groups)
    return MotionGraph.load(run / "graph", episodes)


def _save_loops(path: Path, loops: list[Loop]) -> None:
    payload = [
        {
            "nodes": L.nodes.tolist(),
            "jump_edges": [list(e) for e in L.jump_edges],
            "jump_costs": L.jump_costs,
            "n_frames": L.n_frames,
            "duration_s": L.duration_s,
            "pen_rotation_deg": float(np.rad2deg(L.pen_rotation_rad)),
            "episodes": L.episodes,
        }
        for L in loops
    ]
    path.write_text(json.dumps(payload))


def _load_loops(path: Path) -> list[Loop]:
    if not path.is_file():
        return []
    out = []
    for d in json.loads(path.read_text()):
        out.append(
            Loop(
                nodes=np.asarray(d["nodes"], dtype=np.int64),
                jump_edges=[tuple(e) for e in d["jump_edges"]],
                jump_costs=list(d["jump_costs"]),
                n_frames=int(d["n_frames"]),
                duration_s=float(d["duration_s"]),
                pen_rotation_rad=float(np.deg2rad(d["pen_rotation_deg"])),
                episodes=list(d["episodes"]),
            )
        )
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))


def _parse_weights(spec: str | None) -> dict[str, float]:
    weights = dict(DEFAULT_BLOCK_WEIGHTS)
    if spec:
        for item in spec.split(","):
            k, v = item.split("=")
            weights[k.strip()] = float(v)
    return weights


# ---------------------------------------------------------------------- #
# commands
# ---------------------------------------------------------------------- #
def cmd_info(args) -> None:
    episodes = load_clip_groups(args.clips, groups=args.groups)
    print(summarize(episodes))
    if Path(args.run, "graph", "meta.json").is_file():
        print(json.dumps(json.loads(Path(args.run, "graph", "meta.json").read_text()), indent=2))


def cmd_build(args) -> None:
    run = _run_dir(args)
    t0 = time.time()
    episodes = load_clip_groups(args.clips, groups=args.groups, verify_hashes=args.verify_hashes)
    print(summarize(episodes))
    featurizer = FrameFeaturizer(block_weights=_parse_weights(args.weights)).fit(episodes)
    graph = MotionGraph(episodes, featurizer)
    print(f"features: {featurizer.dim}-D ({', '.join(featurizer.active_blocks)}); step unit {graph.step_unit:.3f}")
    graph.propose_jumps(
        k=args.k, window=args.window, max_cost=args.max_cost,
        allow_same_episode=not args.no_same_episode, min_frame_gap=args.min_frame_gap, verbose=True,
    )
    limits = KinematicLimits.from_dataset(graph, blend_frames=args.min_blend, max_blend_frames=args.max_blend)
    if args.max_joint_vel is not None:
        limits.max_joint_vel = args.max_joint_vel
    report = apply_kinematic_filter(graph, limits)
    if not args.no_rest_transitions:
        n_rest = graph.propose_rest_transitions(blend_frames=args.rest_blend, max_pen_pos_m=args.rest_pen_cm / 100.0, max_pen_axis_rad=np.deg2rad(args.rest_axis_deg))
        print(f"rest-to-start (trick-chaining) transitions proposed: {n_rest}")
    if args.reuse_from:
        old = MotionGraph.load(Path(args.reuse_from) / "graph", episodes)
        n = graph.transfer_statuses(old)
        print(f"reused {n} simulation outcomes from {args.reuse_from}")
    print(graph.describe())
    print(
        f"kinematic limits: |qdot| <= {limits.max_joint_vel:.2f} rad/s, blend {limits.blend_frames}-{limits.max_blend_frames} frames, "
        f"pen jump <= {100 * limits.max_pen_pos_jump_m:.1f} cm / {np.rad2deg(limits.max_pen_axis_jump_rad):.0f} deg (growing with blend); "
        f"failed {int((~report.ok).sum())} of {report.ok.size}; blend histogram {np.bincount(report.blend[report.ok]).tolist()}"
    )
    graph.save(run / "graph")
    _write_json(run / "build.json", {
        "seconds": time.time() - t0, "num_nodes": graph.num_nodes, "num_episodes": graph.num_episodes,
        "feature_dim": featurizer.dim, "blocks": featurizer.active_blocks, "block_weights": featurizer.block_weights,
        "step_unit": graph.step_unit, "k": args.k, "window": args.window, "max_cost": args.max_cost,
        "jumps": graph.jumps.summary(), "max_joint_vel": limits.max_joint_vel,
    })
    print(f"saved to {run / 'graph'} ({time.time() - t0:.1f}s)")


def cmd_validate(args) -> None:
    from .sim import ValidationConfig, validate_candidates
    from .sim.rollout import Thresholds

    run = _run_dir(args)
    graph = _load_graph(args)
    cfg = ValidationConfig(
        controller=args.controller, blend_frames=None, follow_frames=args.follow_frames,
        success_pos_m=args.success_pos_cm / 100.0, success_axis_rad=np.deg2rad(args.success_axis_deg),
        relative=not args.no_relative, relative_factor=args.relative_factor,
        thresholds=Thresholds(pos_fail_m=args.fail_pos_cm / 100.0, axis_fail_rad=np.deg2rad(args.fail_axis_deg)),
    )
    print(graph.describe())
    priority = None
    if args.loop_closing_first or args.only_loop_closing:
        from .graph.loops import loop_closing_candidates

        priority = loop_closing_candidates(graph)
        print(f"{priority.size} candidates lie on cycles of the candidate graph (validated first)")
    report = validate_candidates(graph, cfg, max_jumps=args.max_jumps, workers=args.workers, priority=priority, only_priority=args.only_loop_closing, rest_only=args.rest_only)
    print(json.dumps(report, indent=2))
    graph.save(run / "graph")
    _write_json(run / "validation.json", {**report, "config": {
        "controller": cfg.controller, "blend_frames": cfg.blend_frames, "follow_frames": cfg.follow_frames,
        "success_pos_m": cfg.success_pos_m, "success_axis_deg": float(np.rad2deg(cfg.success_axis_rad)),
        "relative": cfg.relative, "relative_factor": cfg.relative_factor,
    }})
    scc = strongly_connected(graph)
    print(f"validated graph: largest strongly connected component = {scc.largest_size} frames, components>1: {int((scc.sizes > 1).sum())}")


def cmd_loops(args) -> None:
    run = _run_dir(args)
    graph = _load_graph(args)
    statuses = (JumpStatus.VALIDATED,) if not args.use_candidates else (JumpStatus.VALIDATED, JumpStatus.CANDIDATE)

    def search():
        found = find_loops(
            graph, statuses=statuses, max_jumps_considered=args.max_jumps_considered, max_loops=args.max_loops,
            min_frames=args.min_frames, max_frames=args.max_frames, jump_cost_offset=args.jump_cost_offset,
        )
        found = [L for L in found if np.rad2deg(L.pen_rotation_rad) >= args.min_rotation_deg]
        found.sort(key=lambda L: (-L.pen_rotation_rad / max(L.duration_s, 1e-6), L.n_jumps))
        return found

    loops = search()
    print(f"{len(loops)} loops (>= {args.min_rotation_deg} deg of pen rotation)")
    verification = None
    rounds_done = 0
    if args.verify_top > 0 and loops:
        from .sim import PenSpinScene
        from .sim.rollout import make_controller
        from .sim.verify import reject_jumps, verify_loops

        scene, ctrl = PenSpinScene(), make_controller(args.controller)
        for rnd in range(max(1, args.rounds)):
            if not loops:
                break
            print(f"round {rnd + 1}: replaying the top {min(args.verify_top, len(loops))} loops end to end ({args.verify_repeats}x, {args.controller}):")
            ver = verify_loops(graph, loops[: args.verify_top], scene, ctrl, repeats=args.verify_repeats)
            rounds_done = rnd + 1
            blamed = [v.blamed_jump for v in ver if v.blamed_jump is not None]
            n_ok = sum(v.completed for v in ver)
            print(f"  {n_ok}/{len(ver)} loops completed; {len(set(blamed))} distinct jumps blamed for failures")
            if args.prune_failed and blamed and rnd < args.rounds - 1:
                n = reject_jumps(graph, blamed)
                print(f"  rejected {n} jumps, searching loops again")
                loops = search()
                continue
            order = sorted(range(len(ver)), key=lambda i: (not ver[i].completed, -ver[i].survived_fraction, -loops[i].pen_rotation_rad / max(loops[i].duration_s, 1e-6)))
            rest = list(range(len(ver), len(loops)))
            loops = [loops[i] for i in order] + [loops[i] for i in rest]
            verification = [{"completed": ver[i].completed, "survived_fraction": ver[i].survived_fraction, "ticks": ver[i].ticks, "summary": ver[i].result.summary()} for i in order]
            break
        if args.prune_failed:
            graph.save(run / "graph")
    for L in loops[: args.show]:
        print("  " + L.describe(graph))
    scc = strongly_connected(graph, statuses=statuses)
    print(f"largest strongly connected component: {scc.largest_size} frames ({100 * scc.largest_size / graph.num_nodes:.1f}% of all frames)")
    _save_loops(run / "loops.json", loops)
    np.save(run / "scc_largest_nodes.npy", scc.nodes_in_largest())
    _write_json(run / "loops_summary.json", {
        "n_loops": len(loops), "largest_scc": scc.largest_size, "n_scc_gt1": int((scc.sizes > 1).sum()),
        "verification": verification, "verify_rounds": rounds_done, "jumps": graph.jumps.summary(),
        "best": None if not loops else {"frames": loops[0].n_frames, "jumps": loops[0].n_jumps, "rotation_deg": float(np.rad2deg(loops[0].pen_rotation_rad)), "episodes": [graph.episodes[e].episode_id for e in loops[0].episodes]},
    })


def cmd_replay(args) -> None:
    """Replay a found loop (or a clip) in simulation with the tracking controller."""
    from .graph.planner import loop_to_trajectory
    from .sim import PenSpinScene, rollout, track_episode
    from .sim.rollout import frame_state_from_episode, make_controller

    run = _run_dir(args)
    graph = _load_graph(args)
    scene = PenSpinScene()
    ctrl = make_controller(args.controller)
    if args.clip is not None:
        ep = next(e for e in graph.episodes if e.episode_id == args.clip)
        res = track_episode(scene, ctrl, ep, start=args.start, n_frames=args.frames)
        print(f"{ep.episode_id} [{args.controller}] {res.summary()}")
        return
    loops = _load_loops(run / "loops.json")
    if not loops:
        sys.exit("no loops found yet: run `penspin-graph loops` first")
    loop = loops[args.loop]
    print(loop.describe(graph))
    traj = loop_to_trajectory(graph, loop, repeats=args.repeats, blend_frames=args.blend_frames)
    e, t = graph.node_episode_frame(int(loop.nodes[0]))
    res = rollout(scene, ctrl, traj, frame_state_from_episode(graph.episodes[e], t))
    print(f"loop x{args.repeats} [{args.controller}] {res.summary()}  (max ref joint vel {traj.max_joint_velocity():.2f} rad/s)")
    if args.save:
        np.savez(run / f"loop_{args.loop}_reference.npz", qpos=traj.qpos, pen_pose_local=traj.pen_pose, nodes=traj.nodes, frame_dt=traj.frame_dt)
        print(f"saved reference to {run / f'loop_{args.loop}_reference.npz'}")


def cmd_eval(args) -> None:
    from .policy import GraphTrackingPolicy, evaluate_closed_loop
    from .sim.rollout import make_controller

    run = _run_dir(args)
    graph = _load_graph(args)
    restrict = None
    if args.restrict_scc and (run / "scc_largest_nodes.npy").is_file():
        nodes = np.load(run / "scc_largest_nodes.npy")
        restrict = np.zeros(graph.num_nodes, bool)
        restrict[nodes] = True
        print(f"matching restricted to the largest SCC ({nodes.size} frames)")
    policy = GraphTrackingPolicy(graph, make_controller(args.controller), restrict_to=restrict)
    report = evaluate_closed_loop(graph, policy, n_starts=args.n_starts, horizon=args.horizon, seed=args.seed)
    print("graph policy:", report.describe())
    _write_json(run / "eval_graph_policy.json", {
        "summary": report.describe(), "controller": args.controller, "horizon": args.horizon,
        "episodes": [e.__dict__ for e in report.episodes],
    })


def cmd_export(args) -> None:
    from .policy import build_export_trajectories, export_clip_group

    run = _run_dir(args)
    graph = _load_graph(args)
    loops = _load_loops(run / "loops.json")[: args.max_loops]
    trajs = build_export_trajectories(graph, loops=loops, n_walks=args.n_walks, target_seconds=args.seconds, blend_frames=args.blend_frames, seed=args.seed)
    if not trajs:
        sys.exit("nothing to export: validate jumps first")
    out_root = run / "export" / "clips"
    report = export_clip_group(graph, trajs, out_root=out_root, group_id=args.group_id, recipe_dir=run / "export" / "recipe")
    print(report.describe())
    print(
        "\nTo train PPO with wuji-mjlab, copy\n"
        f"  {report.group_dir}  ->  wuji-mjlab/src/wuji_mjlab/tasks/pen_spin/data/rawdata/clips/{args.group_id}\n"
        f"  {report.recipe_path} ->  wuji-mjlab/src/wuji_mjlab/tasks/pen_spin/data/recipe/{args.group_id}.json\n"
        "then: pixi run train --task WujiHand2_PenSpin env.commands.motion.recipe_path=src/wuji_mjlab/tasks/pen_spin/data/recipe/"
        f"{args.group_id}.json env.scene.num_envs=2048 agent.max_iterations=10000 agent.run_name=PenSpin_{args.group_id}"
    )


def cmd_train_bc(args) -> None:
    from .policy import BCDataset, BCPolicy, collect_bc_dataset, evaluate_closed_loop, train_bc

    run = _run_dir(args)
    graph = _load_graph(args)
    bc_dir = run / "bc"
    bc_dir.mkdir(exist_ok=True)
    ds_path = bc_dir / "dataset.npz"
    if ds_path.is_file() and not args.recollect:
        data = BCDataset.load(ds_path)
        print(f"loaded {data.ticks} samples from {ds_path}")
    else:
        data = collect_bc_dataset(graph, n_walks=args.n_walks, walk_frames=args.walk_frames, controller=args.controller, seed=args.seed)
        data.save(ds_path)
        print(f"collected {data.ticks} samples from {data.walks} walks")
    policy = train_bc(data, hidden=tuple(args.hidden), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    policy.save(bc_dir)
    report = evaluate_closed_loop(graph, policy, n_starts=args.n_starts, horizon=args.horizon, seed=args.seed + 1)
    print("BC policy:", report.describe())
    _write_json(bc_dir / "eval.json", {"summary": report.describe(), "samples": data.ticks, "walks": data.walks, "epochs": args.epochs, "hidden": list(args.hidden)})


def cmd_render(args) -> None:
    """Render a loop replay (or a clip) to MP4/GIF. Headless Linux: MUJOCO_GL=egl."""
    from .graph.planner import loop_to_trajectory
    from .sim import PenSpinScene
    from .sim.render import render_rollout
    from .sim.rollout import episode_reference, frame_state_from_episode, make_controller

    run = _run_dir(args)
    graph = _load_graph(args)
    scene = PenSpinScene()
    ctrl = make_controller(args.controller)
    out = Path(args.out) if args.out else run / "figures" / (f"loop_{args.loop}.{args.format}" if args.clip is None else "clip.%s" % args.format)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.clip is not None:
        ep = next(e for e in graph.episodes if e.episode_id == args.clip)
        ref = episode_reference(ep, 0, len(ep) if args.frames is None else args.frames)
        start = frame_state_from_episode(ep, 0)
    else:
        loops = _load_loops(run / "loops.json")
        if not loops:
            sys.exit("no loops found yet: run `penspin-graph loops` first")
        loop = loops[args.loop]
        ref = loop_to_trajectory(graph, loop, repeats=args.repeats, blend_frames=args.blend_frames)
        e, t = graph.node_episode_frame(int(loop.nodes[0]))
        start = frame_state_from_episode(graph.episodes[e], t)
    path = render_rollout(scene, ctrl, ref, start, out, width=args.width, height=args.height, fps=args.fps, max_steps=args.frames)
    print(f"wrote {path} ({len(ref)} reference frames)")


def cmd_viz(args) -> None:
    from .viz import plot_embedding, plot_jump_matrix, plot_loop

    run = _run_dir(args)
    graph = _load_graph(args)
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(plot_embedding(graph, fig_dir / "embedding_pca.png"))
    print(plot_jump_matrix(graph, fig_dir / "jump_matrix.png"))
    for i, loop in enumerate(_load_loops(run / "loops.json")[: args.max_loops]):
        print(plot_loop(graph, loop, fig_dir / f"loop_{i}.png"))


def cmd_report(args) -> None:
    run = _run_dir(args)
    lines = ["# penspin-graph run report", ""]
    for name in ["build.json", "validation.json", "loops_summary.json", "eval_graph_policy.json", "bc/eval.json"]:
        p = run / name
        if p.is_file():
            payload = json.loads(p.read_text())
            payload.pop("episodes", None)
            lines += [f"## {name}", "", "```json", json.dumps(payload, indent=2), "```", ""]
    (run / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="penspin-graph", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/default", help="run directory (graph, loops, exports)")
    p.add_argument("--clips", default=str(DEFAULT_CLIPS), help="clip catalog root (wuji-mjlab layout)")
    p.add_argument("--groups", nargs="*", default=None, help="restrict to these motion groups")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="dataset / graph summary")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("build", help="embed frames, propose jumps, kinematic filter")
    s.add_argument("--k", type=int, default=8, help="nearest neighbours per node")
    s.add_argument("--window", type=int, default=5, help="frames compared after the jump")
    s.add_argument("--max-cost", type=float, default=4.0, help="max window cost in step units")
    s.add_argument("--weights", default=None, help="block weights, e.g. 'qpos=1,pen_pos=3,links=0.5'")
    s.add_argument("--no-same-episode", action="store_true", help="do not propose jumps inside one episode (default: allowed with --min-frame-gap)")
    s.add_argument("--min-frame-gap", type=int, default=50, help="min |t_u - t_v| for same-episode jumps")
    s.add_argument("--min-blend", type=int, default=2, help="minimum blended frames per jump")
    s.add_argument("--max-blend", type=int, default=10, help="maximum blended frames per jump (longer jumps fail the kinematic filter)")
    s.add_argument("--max-joint-vel", type=float, default=None, help="rad/s (default: from the data)")
    s.add_argument("--verify-hashes", action="store_true")
    s.add_argument("--reuse-from", default=None, help="run dir whose validated/rejected jump outcomes are copied for identical pairs")
    s.add_argument("--no-rest-transitions", action="store_true", help="do not propose hold -> clip-start transitions")
    s.add_argument("--rest-blend", type=int, default=25, help="minimum blend frames for hold -> clip-start transitions")
    s.add_argument("--rest-pen-cm", type=float, default=6.0, help="max pen position gap for hold -> clip-start transitions")
    s.add_argument("--rest-axis-deg", type=float, default=60.0, help="max pen shaft-direction gap for hold -> clip-start transitions")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("validate", help="validate candidate jumps in MuJoCo")
    s.add_argument("--controller", choices=["onnx", "pd"], default="onnx")
    s.add_argument("--follow-frames", type=int, default=25)
    s.add_argument("--loop-closing-first", action="store_true", help="validate candidates that lie on cycles first")
    s.add_argument("--only-loop-closing", action="store_true", help="validate only candidates that lie on cycles")
    s.add_argument("--rest-only", action="store_true", help="validate only hold -> clip-start transitions")
    s.add_argument("--success-pos-cm", type=float, default=3.0)
    s.add_argument("--success-axis-deg", type=float, default=30.0)
    s.add_argument("--fail-pos-cm", type=float, default=5.0)
    s.add_argument("--fail-axis-deg", type=float, default=60.0)
    s.add_argument("--no-relative", action="store_true", help="skip the baseline comparison")
    s.add_argument("--relative-factor", type=float, default=1.5)
    s.add_argument("--max-jumps", type=int, default=None, help="only test the N cheapest candidates")
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("loops", help="find loops through validated jumps")
    s.add_argument("--use-candidates", action="store_true", help="also use unvalidated candidates")
    s.add_argument("--max-jumps-considered", type=int, default=5000)
    s.add_argument("--max-loops", type=int, default=300)
    s.add_argument("--min-frames", type=int, default=50)
    s.add_argument("--max-frames", type=int, default=3000)
    s.add_argument("--min-rotation-deg", type=float, default=180.0)
    s.add_argument("--jump-cost-offset", type=float, default=25.0)
    s.add_argument("--show", type=int, default=10)
    s.add_argument("--verify-top", type=int, default=0, help="replay the top-K loops end to end in simulation and rank completed loops first")
    s.add_argument("--verify-repeats", type=int, default=1)
    s.add_argument("--prune-failed", action="store_true", help="reject the jump blamed for each failed replay and search again (needs --rounds > 1)")
    s.add_argument("--rounds", type=int, default=1, help="verify/prune rounds")
    s.add_argument("--controller", choices=["onnx", "pd"], default="onnx")
    s.set_defaults(func=cmd_loops)

    s = sub.add_parser("replay", help="replay a loop or a clip with the tracking controller")
    s.add_argument("--controller", choices=["onnx", "pd"], default="onnx")
    s.add_argument("--loop", type=int, default=0)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--blend-frames", type=int, default=None, help="override the per-jump blend length")
    s.add_argument("--clip", default=None, help="episode id, e.g. 01_single_axis/clip_001")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--frames", type=int, default=None)
    s.add_argument("--save", action="store_true", help="save the loop reference (qpos @ 50 Hz) as NPZ")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("eval", help="closed-loop graph policy (nearest frame + tracking) from random states")
    s.add_argument("--controller", choices=["onnx", "pd"], default="onnx")
    s.add_argument("--n-starts", type=int, default=20)
    s.add_argument("--horizon", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--restrict-scc", action="store_true", help="only match into the largest SCC")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("export", help="export stitched clips for wuji-mjlab PPO training")
    s.add_argument("--group-id", default="06_graph_stitched")
    s.add_argument("--max-loops", type=int, default=10)
    s.add_argument("--n-walks", type=int, default=20)
    s.add_argument("--seconds", type=float, default=20.0)
    s.add_argument("--blend-frames", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("train-bc", help="behaviour-clone the graph policy into an MLP")
    s.add_argument("--controller", choices=["onnx", "pd"], default="onnx")
    s.add_argument("--n-walks", type=int, default=200)
    s.add_argument("--walk-frames", type=int, default=150)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch-size", type=int, default=512)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    s.add_argument("--n-starts", type=int, default=20)
    s.add_argument("--horizon", type=int, default=250)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--recollect", action="store_true")
    s.set_defaults(func=cmd_train_bc)

    s = sub.add_parser("render", help="render a loop replay or a clip to MP4/GIF (MUJOCO_GL=egl on headless Linux)")
    s.add_argument("--controller", choices=["onnx", "pd"], default="onnx")
    s.add_argument("--loop", type=int, default=0)
    s.add_argument("--repeats", type=int, default=2)
    s.add_argument("--blend-frames", type=int, default=None)
    s.add_argument("--clip", default=None)
    s.add_argument("--frames", type=int, default=None)
    s.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    s.add_argument("--out", default=None)
    s.add_argument("--width", type=int, default=480)
    s.add_argument("--height", type=int, default=360)
    s.add_argument("--fps", type=int, default=25)
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("viz", help="figures: embedding, jump matrix, loops")
    s.add_argument("--max-loops", type=int, default=3)
    s.set_defaults(func=cmd_viz)

    s = sub.add_parser("report", help="write runs/<name>/report.md")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
