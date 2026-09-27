"""Render every camera in configs/cameras.yaml to PNGs for quick camera tuning.

    .venv/bin/python scripts/01_scene_preview.py [--seed 0] [--res 480 640] [--out data/preview]
    .venv/bin/python scripts/01_scene_preview.py --episode data/raw/episode_000000.hdf5 --step 100

Edit configs/cameras.yaml and re-run until the views look right. With --episode, the frame is
rendered from a recorded demo (proof that cameras can change after collection).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import load_episode  # noqa: E402
from pusht.render import CameraRenderer  # noqa: E402
from pusht.scene import all_cameras  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, nargs=2, default=[480, 480], metavar=("H", "W"))
    ap.add_argument("--out", default="data/preview")
    ap.add_argument("--episode", help="render from a recorded raw episode instead of a fresh reset")
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()

    cams = all_cameras(load_config("cameras"))
    if args.episode:
        ep = load_episode(args.episode)
        env = PushTEnv(cameras=cams, xml=ep["physics_xml"])
        env.set_state(ep["state"][args.step])
    else:
        env = PushTEnv(cameras=cams)
        env.reset(seed=args.seed)
    mujoco.mj_forward(env.model, env.data)

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = CameraRenderer(env.model, cams)
    for name in cams:
        img = renderer.render(env.data, name, args.res)
        Image.fromarray(img).save(out / f"{name}.png")
        print(f"saved {out / f'{name}.png'}")
    renderer.close()


if __name__ == "__main__":
    main()
