"""The idea in closed loop: nearest frame -> follow its episode (through validated jumps).

:class:`GraphTrackingPolicy` glues the :class:`MotionMatchingController`
(which frame should the hand be tracking right now?) to a low-level tracking
controller (how do I get there?).  It needs no external reference signal: the
reference is looked up from the graph at every tick, so a perturbed hand
simply re-attaches to the closest recorded motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features.featurizer import FrameState
from ..graph.motion_graph import MotionGraph
from ..graph.planner import MotionMatchingController, nearest_nodes
from ..sim.controller import Controller, LiveState, ReferenceFrame


def _reference_frame(graph: MotionGraph, node: int) -> ReferenceFrame:
    e, t = graph.node_episode_frame(node)
    ep = graph.episodes[e]
    return ReferenceFrame(qpos=ep.qpos[t], pen_pose=ep.pen_pose[t], link_pos=ep.link_pos[t], group_id=ep.group_id)


def _frame_state(live: LiveState) -> FrameState:
    return FrameState(
        qpos=live.qpos, qvel=live.qvel, pen_pose=live.pen_pose,
        pen_linvel=np.zeros(3), pen_angvel=np.zeros(3), link_pos=live.link_pos,
    )


@dataclass
class GraphTrackingPolicy(Controller):
    graph: MotionGraph
    tracker: Controller  # e.g. OnnxTrackingController()
    matcher: MotionMatchingController | None = None
    restrict_to: np.ndarray | None = None  # boolean node mask (e.g. largest SCC)
    name: str = "graph"
    _initialised: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.matcher is None:
            self.matcher = MotionMatchingController(self.graph, restrict_to=self.restrict_to)

    def reset(self) -> None:
        self.tracker.reset()
        self._initialised = False

    def act(self, ref, live: LiveState) -> np.ndarray:
        state = _frame_state(live)
        if not self._initialised:
            self.matcher.reset(state)
            self._initialised = True
        self.matcher.step(state)  # advances / re-matches / blends; the frame to track is matcher.current_ref
        ref = self.matcher.current_ref
        frame = ReferenceFrame(qpos=ref.qpos, pen_pose=ref.pen_pose, link_pos=ref.link_pos, group_id=ref.group_id)
        return self.tracker.act(frame, live)

    @property
    def tracked_nodes(self) -> list[int]:
        return list(self.matcher.history)
