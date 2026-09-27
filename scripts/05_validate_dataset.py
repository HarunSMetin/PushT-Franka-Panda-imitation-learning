"""Validate an exported LeRobot dataset.

    .venv/bin/python scripts/05_validate_dataset.py --root data/lerobot/pusht_sim

Checks: loads with LeRobotDataset, shapes/dtypes/fps, normalization stats present, decoded
video frames look sane, action/state alignment, and an open-loop replay of the dataset
actions in the simulator from each episode's recorded initial state. Writes a contact sheet
of decoded frames to <root>/validation_frames.png.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import logging  # noqa: E402

# torchcodec needs system FFmpeg (`sudo apt install ffmpeg`); without it LeRobot logs a
# 100-line warning and falls back to PyAV, which works fine. Keep the fallback, drop the noise.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import load_episode  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/lerobot/pusht_sim")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--raw-dir", default=None, help="raw episodes (for the open-loop replay check)")
    ap.add_argument("--replay-episodes", type=int, default=3)
    ap.add_argument("--video-backend", default="pyav")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = resolve(args.root)
    info = json.loads((root / "meta" / "export_info.json").read_text())
    ds = LeRobotDataset(args.repo_id or f"local/{root.name}", root=root, video_backend=args.video_backend)
    cams = ds.meta.camera_keys
    print(f"dataset {root}\n  episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps} cameras={cams}")

    # --- structure
    check("fps matches export", ds.fps == info["fps"], f"({ds.fps})")
    check("episode count matches export", ds.num_episodes == len(info["episodes"]))
    check("frame count matches export", ds.num_frames == sum(e["frames"] for e in info["episodes"]))
    for k in ("observation.state", "action"):
        check(f"stats for {k}", k in ds.meta.stats and "mean" in ds.meta.stats[k])
    for c in cams:
        check(f"stats for {c}", c in ds.meta.stats)

    # --- item contents
    item = ds[0]
    for c in cams:
        img = item[c]
        h, w = info["resolution"][c.split(".")[-1]]
        check(f"{c} shape", tuple(img.shape) == (3, h, w), f"{tuple(img.shape)}")
        check(f"{c} range [0,1] and not blank", 0 <= float(img.min()) and float(img.max()) <= 1
              and float(img.std()) > 0.02, f"min={float(img.min()):.2f} max={float(img.max()):.2f}")
    check("state shape", tuple(item["observation.state"].shape) == (2,))
    check("action shape", tuple(item["action"].shape) == (2,))
    check("task string", item["task"] == info["task"], repr(item["task"]))

    # --- contact sheet of decoded frames (first episode, up to 8 frames per camera)
    if cams:
        ep0 = ds.meta.episodes[0]
        lo, hi = int(ep0["dataset_from_index"]), int(ep0["dataset_to_index"])
        idxs = np.linspace(lo, hi - 1, min(8, hi - lo)).astype(int)
        rows = []
        for c in cams:
            frames = [(ds[int(i)][c].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for i in idxs]
            rows.append(np.concatenate(frames, axis=1))
        width = max(r.shape[1] for r in rows)
        sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows], axis=0)
        Image.fromarray(sheet).save(root / "validation_frames.png")
        print(f"  contact sheet -> {root / 'validation_frames.png'}")

    # --- action / state alignment: action[k] should be close to state[k+1]
    hf = ds.hf_dataset.with_format("numpy")
    act = np.stack(hf["action"])
    st = np.stack(hf["observation.state"])
    epi = np.asarray(hf["episode_index"])
    same = epi[:-1] == epi[1:]
    gap = np.linalg.norm(act[:-1][same] - st[1:][same], axis=1)
    check("action[k] ~ state[k+1] (tracking)", np.median(gap) < 0.01,
          f"median {np.median(gap) * 1e3:.1f} mm, p95 {np.percentile(gap, 95) * 1e3:.1f} mm")
    check("actions inside workspace", True,
          f"x[{act[:, 0].min():.3f},{act[:, 0].max():.3f}] y[{act[:, 1].min():.3f},{act[:, 1].max():.3f}]")

    # --- open-loop replay of dataset actions in the sim, from each episode's recorded start
    raw_dir = resolve(args.raw_dir or info["raw_dir"])
    for e in info["episodes"][:args.replay_episodes]:
        raw = load_episode(raw_dir / e["raw_file"])
        env = PushTEnv(xml=raw["physics_xml"])
        env.set_state(raw["state"][0], cmd_xy=raw["attrs"]["init_cmd_xy"])
        stride = int(round(env.control_hz / ds.fps))
        mask = epi == e["episode_index"]
        for a in act[mask]:
            for _ in range(stride):
                obs = env.step(a)
        rec_cov = float(raw["coverage"][-1])
        check(f"open-loop replay ep {e['episode_index']}", abs(obs["coverage"] - rec_cov) < 0.15,
              f"replayed coverage {obs['coverage']:.3f} vs recorded {rec_cov:.3f} "
              f"(block pose err {np.linalg.norm(env.block_pose()[:2] - raw['final_block_pose'][:2]) * 1e3:.1f} mm)")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
