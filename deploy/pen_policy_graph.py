#!/usr/bin/env python3
"""Run a stitched motion-graph loop on the real Wuji Hand 2.

This is a thin layer over wuji-mjlab's official ``pen-policy`` deploy runtime
(``deploy/pen_spin/runtime``).  Everything that touches the hardware, the
camera stream, the safety gates (FROZEN -> HOLD -> POLICY, placement check,
Ctrl+C easing the hand open) and the action mapping (residual x 0.5, clip
+-0.6, EMA 0.5) is Wuji's unmodified code.  One thing is added:

    the released policy that produces each frame's action is the policy of the
    motion group that frame was taken from

because a stitched loop mixes groups (02 and 03 for the best loop) and one
policy for the whole loop drops the pen within 5 s (official-env measurement,
runs/official_env/90_loop_pol02.json / 90_loop_pol03.json).

Usage (from the wuji-mjlab repository root, inside its pen-spin-deploy env):

    python /path/to/penspin-graph/deploy/pen_policy_graph.py \
        --motion /path/to/penspin-graph/deploy/clips/graph_loop_20laps.npz \
        --policy-map 02_continuous_single_axis=src/wuji_mjlab/tasks/pen_spin/data/checkpoints/02_continuous_single_axis/policy.onnx \
                     03_continuous_multi_axis=src/wuji_mjlab/tasks/pen_spin/data/checkpoints/03_continuous_multi_axis/policy.onnx \
        [any pen-policy option: --no-sim --obj-from-ref --no-hardware --hand-sn ...]

The per-frame group list is read from ``<motion>.groups.json`` next to the
clip (written by scripts/official_env/export_loop_official.py) or from
``--frame-groups``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _find_runtime_root() -> Path:
    """Locate wuji-mjlab/deploy/pen_spin so ``runtime`` and ``lib`` import."""
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "deploy" / "pen_spin" / "runtime" / "loop.py").is_file():
            return cand / "deploy" / "pen_spin"
        if (cand / "runtime" / "loop.py").is_file() and (cand / "lib" / "onnx_policy.py").is_file():
            return cand
    sys.exit("run this from the wuji-mjlab repository root (deploy/pen_spin/runtime not found)")


class MultiPolicy:
    """Same interface as runtime.policy.Loaded, with one ONNX policy per motion group."""

    def __init__(self, group_to_path: dict[str, str], frame_groups: list[str]):
        from lib.onnx_policy import ONNXPolicy

        self.groups = list(group_to_path)
        self.policies = [ONNXPolicy(p) for p in group_to_path.values()]
        first = self.policies[0]
        for name, pol in zip(self.groups, self.policies):
            same = (pol.ctrl_dt == first.ctrl_dt and pol.action_scale == first.action_scale
                    and pol.action_residual_clip == first.action_residual_clip
                    and pol.action_ema_alpha == first.action_ema_alpha)
            if not same:
                raise ValueError(f"policy contract of {name} differs from {self.groups[0]}")
        unknown = sorted(set(frame_groups) - set(self.groups))
        if unknown:
            raise ValueError(f"clip frames come from groups without a policy: {unknown}")
        self.frame_policy = np.asarray([self.groups.index(g) for g in frame_groups], dtype=np.int64)
        self.ctrl_dt = first.ctrl_dt
        self.action_scale = first.action_scale
        self.clip_lo, self.clip_hi = first.action_residual_clip
        self.alpha = first.action_ema_alpha
        self.obs_dim = first.obs_dim
        self.policy = first  # what preflight inspects
        self._loop = None
        self.switches = 0
        self._last = -1

    def bind(self, loop) -> None:
        self._loop = loop

    def current_index(self) -> int:
        k = 0 if self._loop is None else int(self._loop.idle_frame)
        return int(self.frame_policy[min(max(k, 0), len(self.frame_policy) - 1)])

    def infer(self, obs: np.ndarray) -> np.ndarray:
        i = self.current_index()
        if i != self._last and self._last >= 0:
            self.switches += 1
        self._last = i
        return self.policies[i].infer(obs)

    def describe(self) -> str:
        counts = {g: int((self.frame_policy == i).sum()) for i, g in enumerate(self.groups)}
        return (f"motion-graph loop: per-frame policy switching over {len(self.groups)} released policies "
                f"{counts}  {1.0 / self.ctrl_dt:.0f}Hz scale={self.action_scale} "
                f"clip=[{self.clip_lo},{self.clip_hi}] alpha={self.alpha}")


def main() -> int:
    argv = sys.argv[1:]
    if "--policy-map" not in argv:
        sys.exit("--policy-map GROUP=policy.onnx [GROUP=policy.onnx ...] is required")
    i = argv.index("--policy-map")
    pairs, j = [], i + 1
    while j < len(argv) and not argv[j].startswith("--"):
        pairs.append(argv[j])
        j += 1
    del argv[i:j]
    group_to_path = dict(p.split("=", 1) for p in pairs)
    frame_groups_path = None
    if "--frame-groups" in argv:
        i = argv.index("--frame-groups")
        frame_groups_path = argv[i + 1]
        del argv[i:i + 2]
    if "--motion" not in argv:
        sys.exit("--motion <clip.npz> is required")
    motion = Path(argv[argv.index("--motion") + 1])
    if frame_groups_path is None:
        frame_groups_path = motion.with_suffix(".groups.json")
    frame_groups = json.loads(Path(frame_groups_path).read_text())
    if isinstance(frame_groups, dict):
        frame_groups = frame_groups["frame_groups"]
    with np.load(motion, allow_pickle=False) as data:
        n = int(data["reference_qpos"].shape[0])
    if len(frame_groups) != n:
        sys.exit(f"{frame_groups_path} lists {len(frame_groups)} frames but the clip has {n}")

    root = _find_runtime_root()
    sys.path.insert(0, str(root))
    from runtime import loop as loop_mod
    from runtime import main as main_mod
    from runtime import policy as policy_mod

    multi = MultiPolicy(group_to_path, frame_groups)

    def load_multi(policy_path, *, device="cpu"):  # noqa: ARG001 - same signature as runtime.policy.load
        return multi

    policy_mod.load = load_multi
    orig_init = loop_mod.ControlLoop.__init__

    def init_and_bind(self, session):
        orig_init(self, session)
        session.loaded.bind(self)

    loop_mod.ControlLoop.__init__ = init_and_bind
    # the official CLI insists on one --policy path; it is only displayed
    first_path = next(iter(group_to_path.values()))
    argv += ["--policy", first_path]
    print(f"[graph] {multi.describe()}", flush=True)
    code = main_mod.main(argv)
    print(f"[graph] policy switches during the run: {multi.switches}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
