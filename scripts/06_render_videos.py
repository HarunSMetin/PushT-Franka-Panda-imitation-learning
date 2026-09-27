"""Render recorded demos to MP4 videos, several cameras side by side.

    .venv/bin/python scripts/06_render_videos.py                          # first 5 episodes
    .venv/bin/python scripts/06_render_videos.py --episodes 0 12 40 --cameras overhead corner wrist
    .venv/bin/python scripts/06_render_videos.py --all --res 240 240

Videos go to data/videos/<episode>.mp4. Cameras come from configs/cameras.yaml, so you can
try new camera positions on old demos: edit the YAML, re-run this script, watch the result.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import av  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import list_episodes, load_episode  # noqa: E402
from pusht.render import CameraRenderer  # noqa: E402


TITLE_H = 18  # even, so the yuv420p frame height stays even


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(text), 16), fill=(255, 255, 255))
    d.text((4, 2), text, fill=(0, 0, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--episodes", type=int, nargs="+", help="episode indices (default: first 5)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cameras", nargs="+", default=["overhead", "corner", "corner_left", "wrist"])
    ap.add_argument("--res", type=int, nargs=2, default=[240, 240], metavar=("H", "W"))
    ap.add_argument("--fps", type=int, default=25, help="video fps (demos are recorded at 50 Hz)")
    ap.add_argument("--out", default="data/videos")
    args = ap.parse_args()

    raw_dir = resolve(args.raw_dir or load_config("collect")["raw_dir"])
    paths = {int(p.stem.split("_")[1]): p for p in list_episodes(raw_dir)}
    if args.all:
        idxs = sorted(paths)
    else:
        idxs = args.episodes if args.episodes else sorted(paths)[:5]

    cams_cfg = load_config("cameras")
    all_cams = {**cams_cfg["cameras"], "teleop_view": cams_cfg["teleop_view"]}
    missing = [c for c in args.cameras if c not in all_cams]
    if missing:
        sys.exit(f"Unknown camera(s) {missing}; available: {list(all_cams)}")
    cams = {c: all_cams[c] for c in args.cameras}
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for i in idxs:
        if i not in paths:
            print(f"episode {i}: not found, skipped")
            continue
        ep = load_episode(paths[i])
        env = PushTEnv(cameras=cams, xml=ep["physics_xml"])
        renderer = CameraRenderer(env.model, cams)
        hz = float(ep["attrs"]["control_hz"])
        stride = max(1, int(round(hz / args.fps)))
        path = out / f"{paths[i].stem}.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=int(round(hz / stride)))
            stream.width = args.res[1] * len(cams)
            stream.height = args.res[0] + TITLE_H
            stream.pix_fmt = "yuv420p"
            for t in range(0, len(ep["state"]), stride):
                env.set_state(ep["state"][t])
                tiles = [label(renderer.render(env.data, c, args.res), c) for c in cams]
                row = np.concatenate(tiles, axis=1)
                title = np.full((TITLE_H, row.shape[1], 3), 255, np.uint8)
                title = label(title, f"{paths[i].stem}   t={t / hz:4.1f}s   coverage={ep['coverage'][t] * 100:4.1f}%")
                frame = np.concatenate([title, row], axis=0)
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        renderer.close()
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
