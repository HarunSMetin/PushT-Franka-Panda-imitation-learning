"""How different are the evaluation start states from the training demos?

    .venv/bin/python training/analyze_eval.py outputs/dp_image --checkpoint 080000

For every evaluation seed, the start state (T pose + tip position) is compared with the
training data. Pose distance = mean displacement of the T's 8 corner points between two poses
(cm), which combines position and rotation in one physical number. Reported per eval start:
  - nearest training START state (the demo that began most similarly)
  - nearest T pose ANYWHERE in the training demos (every 10 Hz frame of every demo)
and, if evaluation results exist, success rate split by how novel the start was.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pusht  # noqa: E402,F401
import numpy as np  # noqa: E402

from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import episode_attrs, list_episodes, load_episode  # noqa: E402


def corners(shape, poses: np.ndarray) -> np.ndarray:
    """(N, 3) poses -> (N, 8, 2) world coordinates of the T's corner points."""
    local = np.array(shape.local_polygon().exterior.coords)[:-1]  # (8, 2)
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # (N, 2, 2)
    return np.einsum("nij,kj->nki", R, local) + poses[:, None, :2]


def pose_dist(shape, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean corner displacement (m) between every pose in a (N,3) and b (M,3) -> (N, M)."""
    ca, cb = corners(shape, a), corners(shape, b)
    return np.linalg.norm(ca[:, None] - cb[None], axis=-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--checkpoint", default="080000", help="eval folder whose episodes.csv to use")
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()

    run = Path(args.run)
    eval_dir = run / "eval" / args.checkpoint
    rows = list(csv.DictReader(open(eval_dir / "episodes.csv"))) if (eval_dir / "episodes.csv").exists() else []
    seeds = [int(r["seed"]) for r in rows] or list(range(100_000, 100_100))
    success = {int(r["seed"]): r["success"] == "True" for r in rows}

    env = PushTEnv()
    shape = env.shape

    # training data: start states + every 10 Hz T pose of every usable demo
    starts, tips, frames = [], [], []
    for p in list_episodes(PROJECT_ROOT / args.raw_dir):
        a = episode_attrs(p)
        if a["discard"] or not a["success"]:
            continue
        ep = load_episode(p)
        starts.append(ep["block_pose"][0])
        tips.append(ep["tip_xy"][0])
        frames.append(ep["block_pose"][::5])
    starts, tips, frames = np.array(starts), np.array(tips), np.concatenate(frames)

    # evaluation start states (same sampler + seeds as evaluate.py, without running physics)
    ev = [env.sample_initial(np.random.default_rng(s)) for s in seeds]
    ev_pose = np.array([e[0] for e in ev])
    ev_tip = np.array([e[1] for e in ev])

    d_start = pose_dist(shape, ev_pose, starts)
    near = d_start.argmin(1)
    near_start = d_start.min(1) * 100
    near_start_tip = np.linalg.norm(ev_tip - tips[near], axis=1) * 100
    near_any = pose_dist(shape, ev_pose, frames).min(1) * 100
    dd = pose_dist(shape, starts, starts)
    np.fill_diagonal(dd, np.inf)
    train_spacing = dd.min(1) * 100

    def stats(x):
        return f"median {np.median(x):5.1f} cm   (min {x.min():4.1f}, max {x.max():5.1f})"

    print(f"training: {len(starts)} demos, {len(frames)} T poses at 10 Hz")
    print(f"evaluation: {len(seeds)} starts (seeds {seeds[0]}..{seeds[-1]})\n")
    print("T pose distance = mean movement of the T's corners between two poses")
    print(f"  training start -> nearest OTHER training start : {stats(train_spacing)}")
    print(f"  eval start     -> nearest training start       : {stats(near_start)}")
    print(f"  eval start     -> nearest T pose in ANY demo    : {stats(near_any)}")
    print(f"  tip start: distance to the tip start of that nearest demo: median {np.median(near_start_tip):.1f} cm")
    exact = int((near_start < 0.1).sum())
    print(f"  eval starts identical to a training start (<1 mm): {exact}")

    if success:
        ok = np.array([success[s] for s in seeds])
        print(f"\nsuccess vs novelty of the start (nearest training start):")
        for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 100)):
            m = (near_start >= lo) & (near_start < hi)
            if m.any():
                print(f"  {lo:>2}-{hi:<3} cm: {ok[m].mean() * 100:5.1f}% success  ({m.sum()} starts)")
        fails = [s for s, o in zip(seeds, ok) if not o]
        print(f"  failed seeds: {fails}")
    out = {"near_start_cm": near_start.tolist(), "near_any_cm": near_any.tolist(), "seeds": seeds}
    (eval_dir / "novelty.json").write_text(json.dumps(out)) if eval_dir.exists() else None


if __name__ == "__main__":
    main()
