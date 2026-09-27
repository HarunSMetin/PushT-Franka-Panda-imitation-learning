"""Scripted pusher, used only to test the pipeline without a human (smoke test).

It starts the block displaced from the target along the T's symmetry axis and pushes the
stem end straight back, so the block translates onto the target without rotating.
"""

from __future__ import annotations

import numpy as np

TIP_RADIUS = 0.008


def symmetric_push_start(env, rng: np.random.Generator):
    """Block offset from the target along the stem axis; tip just behind the stem end.

    Returns (block_pose, tip_xy, final_tip_xy).
    """
    tx, ty, tyaw = env.target_pose
    u = np.array([np.sin(tyaw), -np.cos(tyaw)])          # world direction of local -y (stem side)
    stem_end = env.shape.centroid_offset                 # centroid -> stem end distance
    d = rng.uniform(0.05, 0.12)
    block = np.array([tx, ty]) + d * u
    tip = block + (stem_end + TIP_RADIUS + 0.03) * u
    final_tip = np.array([tx, ty]) + (stem_end + TIP_RADIUS - 0.002) * u
    return np.array([*block, tyaw]), tip, final_tip
