"""Export raw demos (data/raw/*.hdf5) to a LeRobot dataset (v3 format).

Images are RENDERED HERE from the recorded simulator states, so cameras, resolution and
fps can all be changed and the dataset re-exported without recollecting anything.

    .venv/bin/python scripts/04_export_lerobot.py                       # defaults from configs
    .venv/bin/python scripts/04_export_lerobot.py --cameras overhead corner wrist --res 96 96 \
        --fps 10 --out data/lerobot/pusht_sim --overwrite
    .venv/bin/python scripts/04_export_lerobot.py --no-images --out data/lerobot/pusht_sim_lowdim

Features (per frame, at --fps):
    observation.images.<cam>        video (H, W, 3)
    observation.state               [x, y]  measured stick-tip position (m, table frame)
    observation.environment_state   [x, y, cos(yaw), sin(yaw)]  T-block pose (low-dim policies)
    action                          [x, y]  commanded tip position at the END of the frame
                                            interval (= target for the next 1/fps s)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import logging  # noqa: E402

# torchcodec needs system FFmpeg (`sudo apt install ffmpeg`); without it LeRobot logs a
# 100-line warning and falls back to PyAV, which works fine. Keep the fallback, drop the noise.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

import av  # noqa: E402

av.logging.set_level(av.logging.ERROR)  # FFmpeg prints "Starting second pass..." per video
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import episode_attrs, list_episodes, load_episode  # noqa: E402
from pusht.render import CameraRenderer  # noqa: E402


def build_features(cams: list[str], res: dict[str, tuple[int, int]], env_state: bool) -> dict:
    feats = {
        f"observation.images.{c}": {
            "dtype": "video", "shape": (res[c][0], res[c][1], 3), "names": ["height", "width", "channels"],
        }
        for c in cams
    }
    feats["observation.state"] = {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}
    if env_state:
        feats["observation.environment_state"] = {
            "dtype": "float32", "shape": (4,), "names": ["x", "y", "cos_yaw", "sin_yaw"]}
    feats["action"] = {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}
    return feats


def randomize_appearance(model, rng: np.random.Generator) -> None:
    """Per-episode lighting / colour jitter (optional, for sim-to-real robustness)."""
    model.light_diffuse[:] = np.clip(model.light_diffuse * rng.uniform(0.7, 1.3), 0, 1)
    model.vis.headlight.diffuse[:] = np.clip(model.vis.headlight.diffuse * rng.uniform(0.7, 1.3), 0, 1)
    table = model.geom("table").id
    model.geom_rgba[table, :3] = np.clip(model.geom_rgba[table, :3] + rng.uniform(-0.12, 0.08, 3), 0, 1)
    for g in ("t_block_bar", "t_block_stem"):
        gid = model.geom(g).id
        model.geom_rgba[gid, :3] = np.clip(model.geom_rgba[gid, :3] + rng.uniform(-0.08, 0.08, 3), 0, 1)


def main():
    collect_cfg = load_config("collect")
    exp_cfg = collect_cfg["export"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=collect_cfg["raw_dir"])
    ap.add_argument("--out", default="data/lerobot/pusht_sim", help="dataset root directory")
    ap.add_argument("--repo-id", default=None, help="default: local/<out dir name>")
    ap.add_argument("--cameras", nargs="+", default=exp_cfg["cameras"])
    ap.add_argument("--no-images", action="store_true",
                    help="numbers only (state, block pose, action): for low-dim policies, trains much faster")
    ap.add_argument("--res", type=int, nargs=2, metavar=("H", "W"),
                    help="override resolution of all cameras (default: per-camera from cameras.yaml)")
    ap.add_argument("--fps", type=int, default=exp_cfg["fps"])
    ap.add_argument("--include-failed", action="store_true", help="also export unsuccessful episodes")
    ap.add_argument("--min-final-coverage", type=float, default=None,
                    help="select episodes by final coverage instead of the success flag recorded "
                         "at collection time (e.g. 0.92 = stricter, 0.85 = also keep near-misses)")
    ap.add_argument("--no-env-state", action="store_true", help="omit observation.environment_state")
    ap.add_argument("--light-randomize", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_dir = resolve(args.raw_dir)
    out = resolve(args.out)
    repo_id = args.repo_id or f"local/{out.name}"

    cams_cfg = load_config("cameras")["cameras"]
    missing = [c for c in args.cameras if c not in cams_cfg]
    if missing:
        sys.exit(f"Unknown camera(s) {missing}; defined in configs/cameras.yaml: {list(cams_cfg)}")
    cams = {} if args.no_images else {c: cams_cfg[c] for c in args.cameras}
    res = {c: tuple(args.res) if args.res else tuple(cams[c].get("resolution", (96, 96))) for c in cams}

    paths = []
    for p in list_episodes(raw_dir):
        a = episode_attrs(p)
        if a["discard"]:
            continue
        if args.min_final_coverage is not None:
            if a["final_coverage"] < args.min_final_coverage:
                continue
        elif not (a["success"] or args.include_failed):
            continue
        paths.append(p)
    if args.max_episodes:
        paths = paths[:args.max_episodes]
    if not paths:
        sys.exit(f"No exportable episodes in {raw_dir}")

    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=args.fps, root=out, robot_type="panda_stick",
        features=build_features(list(cams), res, not args.no_env_state),
        use_videos=True, image_writer_threads=8,
    )

    task = collect_cfg["task"]
    episode_map = []
    for ep_i, path in enumerate(tqdm(paths, desc="episodes")):
        ep = load_episode(path)
        hz = float(ep["attrs"]["control_hz"])
        stride = hz / args.fps
        if abs(stride - round(stride)) > 1e-9:
            sys.exit(f"control_hz {hz} is not a multiple of fps {args.fps}")
        stride = int(round(stride))

        env = PushTEnv(cameras=cams, xml=ep["physics_xml"])
        if args.light_randomize:
            randomize_appearance(env.model, np.random.default_rng(ep_i))
        renderer = CameraRenderer(env.model, cams)

        n = len(ep["state"])
        for t in range(0, n, stride):
            env.set_state(ep["state"][t])
            yaw = ep["block_pose"][t, 2]
            frame = {f"observation.images.{c}": renderer.render(env.data, c, res[c]) for c in cams}
            frame["observation.state"] = ep["tip_xy"][t].astype(np.float32)
            if not args.no_env_state:
                bx, by = ep["block_pose"][t, :2]
                frame["observation.environment_state"] = np.array(
                    [bx, by, np.cos(yaw), np.sin(yaw)], np.float32)
            # Command reached at the end of this frame's interval = target for the next 1/fps s.
            frame["action"] = ep["action_xy"][min(t + stride - 1, n - 1)].astype(np.float32)
            frame["task"] = task
            ds.add_frame(frame)
        ds.save_episode()
        renderer.close()
        episode_map.append({"episode_index": ep_i, "raw_file": path.name,
                            "seed": int(ep["attrs"]["seed"]), "success": bool(ep["attrs"]["success"]),
                            "raw_steps": n, "frames": len(range(0, n, stride))})

    ds.finalize()

    info = {
        "raw_dir": str(raw_dir), "fps": args.fps, "cameras": cams, "resolution": res,
        "light_randomize": args.light_randomize, "include_failed": args.include_failed,
        "min_final_coverage": args.min_final_coverage,
        "task": task, "episodes": episode_map,
    }
    (out / "meta" / "export_info.json").write_text(json.dumps(info, indent=2, default=list))
    print(f"\nWrote {len(paths)} episodes, {sum(e['frames'] for e in episode_map)} frames to {out}")
    print(f"repo_id={repo_id}  cameras={list(cams)}  fps={args.fps}")


if __name__ == "__main__":
    main()
