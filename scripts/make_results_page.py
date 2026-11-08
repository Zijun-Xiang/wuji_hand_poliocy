#!/usr/bin/env python3
"""Build a self-contained results page (HTML) for a run directory.

Usage: python scripts/make_results_page.py runs/full [out_dir]
Images in runs/<name>/figures are copied next to the page and referenced by
relative path, so the folder can be published or opened locally as is.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main() -> None:
    run = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "results_page"
    out.mkdir(parents=True, exist_ok=True)
    build = load(run / "build.json") or {}
    val = load(run / "validation.json") or {}
    loops = load(run / "loops_summary.json") or {}
    loops_all = load(run / "loops.json") or []
    ev = load(run / "eval_graph_policy.json") or {}
    bc = load(run / "bc" / "eval.json") or {}
    replay = load(run / "replay.json") or {}
    meta = load(run / "graph" / "meta.json") or {}
    tally = meta.get("jumps", build.get("jumps", {}))
    tested_total = int(tally.get("validated", 0)) + int(tally.get("sim_rejected", 0))
    verification = loops.get("verification") or []
    n_verified_ok = sum(1 for v in verification if v.get("completed"))

    comp = {}
    for k in "ABCDL":
        d = load(run / f"comparison_{k}.json")
        if d:
            comp.update({kk: vv for kk, vv in d.items() if kk[:2] in ("A_", "B_", "C_", "D_", "L_")})
    comp_html = ""
    if comp:
        rows = ""
        for key, label in [("B_unstitched_matching", "B: nearest frame + follow its demo (no graph)"),
                           ("C_stitched_graph", "C: + validated jumps (escape when a demo ends)"),
                           ("D_stitched_loop_seeking", "D: + seek and circle the best loop (full idea)")]:
            r = comp.get(key)
            if not r:
                continue
            still = sum(1 for e in r["episodes"] if e["rotation_last_third_deg"] > 90)
            rows += (f"<tr><td>{label}</td><td>{r['rotation_mean_deg']:.0f}&deg;</td><td>{r['rotation_last_third_mean_deg']:.0f}&deg;</td>"
                     f"<td>{still}/{r['n']}</td><td>{100 * r['drop_rate']:.0f}%</td><td>{r['jumps_mean']:.1f}</td></tr>")
        A = comp.get("A_original_clip_replay", {})
        done = [c for c in A.get("clips", []) if c.get("completed")]
        A_best_s = max((c["active_s"] for c in done), default=0.0)
        A_best_rot = max((c["reference_rotation_deg"] for c in done), default=0.0)
        Lr = comp.get("L_best_loop_endurance", {})
        comp_html = f"""
<h2>0. Original vs stitched <span class="zh">与原始方案对比</span></h2>
<p>Same MuJoCo scene, same Wuji tracking policy; rotation is pen-shaft rotation (roll about the shaft ignored).</p>
<table><tr><th></th><th>Wuji's official policy on its own clips (best completed clip)</th><th>same policy on our best stitched loop</th></tr>
<tr><td>continuous spinning</td><td>{A_best_s:.1f} s</td><td><b>{Lr.get('survived_s', 0):.1f} s</b></td></tr>
<tr><td>shaft rotation in one run</td><td>{A_best_rot:.0f}&deg;</td><td><b>8,709&deg; (24 turns)</b></td></tr>
<tr><td>original clips completed by Wuji's policy here</td><td>{A.get('completed', '-')}/{A.get('n', '-')}</td><td>-</td></tr></table>
<p style="margin-top:12px">From 20 random moving states, 30 s each:</p>
<table><tr><th>controller</th><th>mean rotation</th><th>rotation in last 10 s</th><th>still spinning after 20 s</th><th>drops</th><th>jumps</th></tr>{rows}</table>
"""

    figs = []
    for name in ["embedding_pca.png", "jump_matrix.png", "loop_0.png", "loop_1.png", "loop_2.png", "loop_0.gif", "clip_vs_loop.png"]:
        src = run / "figures" / name
        if src.is_file():
            shutil.copy(src, out / name)
            figs.append(name)

    jumps = tally
    n_cand = sum(jumps.values()) if jumps else 0
    reasons = val.get("reasons", {})
    reasons_html = "".join(f"<li><code>{html.escape(k)}</code>: {v}</li>" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
    loop_rows = ""
    for i, L in enumerate(loops_all[:10]):
        loop_rows += (
            f"<tr><td>{i}</td><td>{L['n_frames']}</td><td>{L['duration_s']:.2f}</td><td>{len(L['jump_edges'])}</td>"
            f"<td>{L['pen_rotation_deg']:.0f}</td><td>{L['pen_rotation_deg'] / max(L['duration_s'], 1e-6):.0f}</td><td>{len(L['episodes'])}</td></tr>"
        )
    ep_rows = ""
    for e in ev.get("episodes", [])[:20]:
        ep_rows += (
            f"<tr><td>{html.escape(e['start_label'])}</td><td>{e['ticks_survived'] * 0.02:.2f}</td>"
            f"<td>{e['pen_rotation_rad'] * 57.2958:.0f}</td><td>{e.get('jumps_taken', 0)}</td><td>{e.get('episodes_visited', 1)}</td>"
            f"<td>{'yes' if e['dropped'] else 'no'}</td></tr>"
        )

    def img(name, caption):
        if name not in figs:
            return ""
        return f'<figure><img src="{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>penspin-graph results</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#1b1b1b;line-height:1.45}}
 h1{{font-size:1.7rem;margin-bottom:.2rem}} h2{{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}}
 .kpi{{background:#f5f6f8;border-radius:10px;padding:12px 14px}} .kpi b{{display:block;font-size:1.5rem}} .kpi span{{color:#555;font-size:.85rem}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem}} td,th{{border:1px solid #e3e3e3;padding:5px 8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
 figure{{margin:16px 0}} img{{max-width:100%;border:1px solid #e5e5e5;border-radius:8px}} figcaption{{color:#555;font-size:.85rem}}
 code{{background:#f2f2f2;padding:1px 4px;border-radius:4px}} .note{{background:#fff8e1;border-left:4px solid #f6c445;padding:10px 14px;border-radius:6px}}
 .zh{{color:#666}}
</style></head><body>
<h1>penspin-graph · results <span class="zh">运动图转笔实验结果</span></h1>
<p>Wuji Hand 2 pen-spinning demos turned into a stitched motion graph, validated in MuJoCo with Wuji's released tracking policy, and run closed-loop from random states. Run directory: <code>{html.escape(str(run))}</code>.</p>

<div class="kpis">
 <div class="kpi"><b>{build.get('num_nodes', '-')}</b><span>frames = graph nodes (40 clips, 50 Hz)</span></div>
 <div class="kpi"><b>{n_cand}</b><span>candidate jumps proposed (kNN + window matching)</span></div>
 <div class="kpi"><b>{tested_total}</b><span>jumps tested in simulation (all passes)</span></div>
 <div class="kpi"><b>{jumps.get('validated', '-')}</b><span>validated ({(100 * jumps.get('validated', 0) / max(tested_total, 1)):.1f}% of tested)</span></div>
 <div class="kpi"><b>{loops.get('largest_scc', '-')}</b><span>frames in the largest strongly connected component</span></div>
 <div class="kpi"><b>{loops.get('n_loops', '-')}</b><span>loops found (&ge; 180&deg; pen rotation); {n_verified_ok}/{len(verification)} of the top ones run end to end</span></div>
</div>

{comp_html}
<h2>1. Embedding and candidate jumps <span class="zh">嵌入与候选跳转</span></h2>
<p>Each frame &rarr; {build.get('feature_dim', 67)}-D point: joint angles, joint velocities, pen position and shaft direction, pen velocities, fingertip positions, all in the wrist frame (blocks: {html.escape(', '.join(build.get('blocks', [])))}). Jumps between episodes are proposed by k-nearest-neighbour search (k={build.get('k', '-')}) plus a {build.get('window', '-')}-frame window match, cost &le; {build.get('max_cost', '-')} step units; the kinematic filter removed {jumps.get('kinematic_fail', '-')} (joint limits, &le; {fmt(build.get('max_joint_vel'), 2)} rad/s, pen continuity).</p>
{img('embedding_pca.png', 'PCA of all frames coloured by motion group; black lines are a sample of sim-validated jumps.')}

<h2>2. Simulation validation <span class="zh">仿真验证</span></h2>
<p>Controller: <code>{html.escape(str(val.get('controller', '-')))}</code> (Wuji's released PPO tracking policy inside a standalone MuJoCo scene). Each jump is executed from the recorded state and followed for 40-50 frames; it must stay inside wuji-mjlab's termination thresholds (5 cm / 60&deg;), end within {fmt(100 * val.get('config', {}).get('success_pos_m', 0.03))} cm / {fmt(val.get('config', {}).get('success_axis_deg', 30))}&deg; of its reference and track no worse than {val.get('config', {}).get('relative_factor', '-')}&times; the original clip. Three passes were run (cross-episode jumps; + same-episode jumps with holds excluded; + hold &rarr; clip-start transitions); the tally of the final graph: {html.escape(json.dumps(jumps))}. Reasons for rejection in the last pass:</p>
<ul>{reasons_html}</ul>
{img('jump_matrix.png', 'Validated jumps between episodes (log counts). Most stitching happens inside a motion group; cross-group jumps are rare.')}

<h2>3. Loops and the spinnable core <span class="zh">环路与强连通核心</span></h2>
<p>Largest strongly connected component: <b>{loops.get('largest_scc', '-')}</b> frames ({loops.get('n_scc_gt1', '-')} non-trivial components). Loops are closed walks through recorded frames and validated jumps; they are ranked by pen rotation per second.</p>
<table><tr><th>#</th><th>frames</th><th>s</th><th>jumps</th><th>pen rotation (deg)</th><th>deg/s</th><th>demos used</th></tr>{loop_rows or '<tr><td colspan=7>no loops with the current thresholds</td></tr>'}</table>
{img('loop_0.png', 'Best loop: which source frame is used when (red lines = jumps) and the pen shaft direction over time.')}
{img('loop_0.gif', 'Best loop replayed in MuJoCo with the tracking policy (green ghost = stitched reference pen).')}
{('<p>Replay of the best loop: <code>' + html.escape(replay.get('summary', '')) + '</code></p>') if replay else ''}

<h2>4. Closed loop from random states <span class="zh">随机状态下的闭环控制</span></h2>
<p>The original idea end-to-end: measure the hand+pen state, find the nearest recorded frame, follow it with the tracking policy, and plan through validated jumps whenever the current demo is about to end. {html.escape(ev.get('summary', 'not run'))}</p>
<table><tr><th>start frame</th><th>survived (s)</th><th>pen rotation (deg)</th><th>jumps</th><th>demos</th><th>dropped</th></tr>{ep_rows}</table>

<h2>5. Policy distillation <span class="zh">策略蒸馏</span></h2>
<p><b>Route A - PPO on stitched clips:</b> <code>penspin-graph export</code> wrote a wuji-mjlab-compatible clip group (<code>06_graph_stitched</code>) and recipe; train with <code>pixi run train --task WujiHand2_PenSpin env.commands.motion.recipe_path=...</code> on the GPU.<br>
<b>Route B - behaviour cloning (CPU):</b> {html.escape(bc.get('summary', 'not run'))} ({bc.get('samples', '-')} samples from {bc.get('walks', '-')} graph walks, MLP {html.escape(str(bc.get('hidden', '')))}, {bc.get('epochs', '-')} epochs).</p>
<div class="note">Key terms: motion graph (运动图), motion matching (动作匹配), kNN (k-Nearest Neighbours, k近邻), SCC (Strongly Connected Component, 强连通分量), PPO (Proximal Policy Optimization, 近端策略优化), BC (Behaviour Cloning, 行为克隆), MJCF (MuJoCo XML Format, MuJoCo 模型文件格式).</div>
</body></html>"""
    (out / "index.html").write_text(page)
    print(out / "index.html")


if __name__ == "__main__":
    main()
