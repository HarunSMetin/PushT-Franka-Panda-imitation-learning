"""Measure how well each camera sees the T block over the recorded demos.

    .venv/bin/python scripts/07_camera_study.py [--every 25] [--res 96 96]

For sampled frames of every usable demo, each camera renders a segmentation image twice:
with the robot (what the policy sees) and without the robot arm. The ratio of visible block
pixels is the block's visibility; 1.0 = never occluded. Also reports the stick tip's
visibility and how many pixels the block covers at the given resolution. Then every camera
pair is scored by how often at least one of the two sees most of the block.
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import episode_attrs, list_episodes, load_episode  # noqa: E402
from pusht.render import ROBOT_VISUAL_GROUP  # noqa: E402

GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--every", type=int, default=25, help="sample every N control steps (50 Hz)")
    ap.add_argument("--res", type=int, nargs=2, default=[96, 96])
    ap.add_argument("--good", type=float, default=0.5, help="'seen' = at least this visible fraction")
    args = ap.parse_args()

    cams = load_config("cameras")["cameras"]
    names = list(cams)
    paths = [p for p in list_episodes(resolve(args.raw_dir or load_config("collect")["raw_dir"]))
             if not episode_attrs(p)["discard"]]
    h, w = args.res

    vis = {c: [] for c in names}     # block visible fraction per sampled frame
    tip = {c: [] for c in names}     # stick tip visible (bool)
    px = {c: [] for c in names}      # unoccluded block pixel count
    opt_full, opt_noarm = mujoco.MjvOption(), mujoco.MjvOption()
    opt_noarm.geomgroup[ROBOT_VISUAL_GROUP] = 0

    for p in paths:
        ep = load_episode(p)
        env = PushTEnv(cameras=cams, xml=ep["physics_xml"])
        m = env.model
        block_ids = [m.geom("t_block_bar").id, m.geom("t_block_stem").id]
        tip_ids = [m.geom("stick_tip").id]
        r = mujoco.Renderer(m, h, w)
        r.enable_segmentation_rendering()
        for t in range(0, len(ep["state"]), args.every):
            env.set_state(ep["state"][t])
            for c in names:
                segs = []
                for opt in (opt_full, opt_noarm):
                    r.update_scene(env.data, camera=c, scene_option=opt)
                    segs.append(r.render())
                full, ref = segs
                is_geom_full = full[..., 1] == GEOM
                n_ref = int(np.isin(ref[..., 0], block_ids).sum() * 1)
                n_vis = int((np.isin(full[..., 0], block_ids) & is_geom_full).sum())
                if n_ref > 0:
                    vis[c].append(n_vis / n_ref)
                    px[c].append(n_ref)
                else:
                    vis[c].append(0.0)  # block out of the field of view
                    px[c].append(0)
                tip[c].append(bool((np.isin(full[..., 0], tip_ids) & is_geom_full).any()))
        r.close()

    n = len(vis[names[0]])
    print(f"{len(paths)} demos, {n} sampled frames, {h}x{w} images\n")
    print(f"{'camera':<13}{'block visible':>14}{f'frames <{args.good:.0%} seen':>18}{'tip visible':>13}{'block px':>10}")
    for c in names:
        v = np.array(vis[c])
        print(f"{c:<13}{v.mean():>13.0%}{(v < args.good).mean():>18.0%}{np.mean(tip[c]):>13.0%}"
              f"{np.median(px[c]):>10.0f}")

    print(f"\nPairs: % of frames where NEITHER camera sees >= {args.good:.0%} of the block (lower is better)")
    scores = []
    for a, b in itertools.combinations(names, 2):
        both_bad = (np.array(vis[a]) < args.good) & (np.array(vis[b]) < args.good)
        scores.append((both_bad.mean(), a, b))
    for s, a, b in sorted(scores)[:8]:
        print(f"  {a:>12} + {b:<12} {s:6.1%}")


if __name__ == "__main__":
    main()
