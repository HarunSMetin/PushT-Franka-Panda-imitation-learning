"""Closed-loop evaluation of a trained policy in the MuJoCo Push-T simulator.

    .venv/bin/python training/evaluate.py outputs/dp_image                     # newest checkpoint
    .venv/bin/python training/evaluate.py outputs/dp_image --checkpoint 050000
    .venv/bin/python training/evaluate.py outputs/dp_image --all               # every checkpoint
    .venv/bin/python training/evaluate.py outputs/dp_image --episodes 100 --videos 10

Every episode starts from a random (seeded) block pose + tip position; the SAME seeds are used
for every checkpoint, so results are directly comparable. The policy sees exactly what it saw in
training: the cameras / resolution / fps of its dataset (or pusht_meta.json of a packaged model)
and the physics of the recorded demos.

An episode ends on success (same rule as data collection: coverage >= success_coverage for
success_hold_time, see configs/collect.yaml) or after --max-time seconds.

All episodes of a checkpoint run side by side (--batch) and the policy plans for all of them
in one GPU call, which is ~30x faster than running them one after another.

Output: outputs/<run>/eval/<checkpoint>/results.json, episodes.csv, videos/*.mp4
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import pusht  # noqa: E402,F401  (GPU rendering env vars; must come before mujoco)
import av  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from policy_runner import PolicyRunner, list_checkpoints  # noqa: E402
from pusht.config import load_config  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.render import CameraRenderer  # noqa: E402

VIDEO_RES = (240, 240)
VIDEO_FPS = 25


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(text), 16), fill=(255, 255, 255))
    d.text((4, 2), text, fill=(0, 0, 0))
    return np.asarray(im)


class VideoWriter:
    """MP4 writer. The file is named <name>.part until it is complete (an MP4 is only playable
    after close() writes its index), then renamed to <name>."""

    def __init__(self, path: Path, width: int, height: int):
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self.container = av.open(str(self.part), "w", format="mp4")
        self.stream = self.container.add_stream("libx264", rate=VIDEO_FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"

    def add(self, frame: np.ndarray) -> None:
        for pkt in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self.container.mux(pkt)

    def close(self) -> None:
        for pkt in self.stream.encode():
            self.container.mux(pkt)
        self.container.close()
        self.part.rename(self.path)


def setup_env(runner: PolicyRunner):
    """Env with the dataset's physics, the policy cameras and the video cameras."""
    meta = runner.meta
    policy_cams = {c: meta["cameras"][c] for c in runner.cameras}
    policy_res = {c: tuple(meta["resolution"][c]) for c in runner.cameras}
    cams_cfg = load_config("cameras")["cameras"]
    # Videos: the policy's own cameras (higher res), or two overview cameras for low-dim policies.
    video_cams = dict(policy_cams) if policy_cams else {c: cams_cfg[c] for c in ("overhead", "corner_left")}
    env = PushTEnv(cameras={**video_cams, **policy_cams}, xml=meta["physics_xml"])  # physics of the demos
    return env, policy_cams, policy_res, video_cams, meta


def run_batch(envs, runner, renderer, policy_res, video_cams, seeds, max_time, video_paths):
    """Run len(seeds) episodes side by side; the policy plans for all of them in one GPU call.

    All episodes start together and tick in lockstep (the policy keeps one shared queue of
    planned actions). A finished episode is frozen: its physics stop, its last observation is
    re-fed to the policy, and its result is final.
    """
    n = len(seeds)
    env0 = envs[0]
    stride = int(round(env0.control_hz / runner.fps))
    hold = max(1, int(round(float(env0.collect_cfg["success_hold_time"]) * env0.control_hz)))
    video_every = max(1, int(round(env0.control_hz / VIDEO_FPS)))
    n_ticks = int(max_time * runner.fps)

    obs = [env.reset(seed=int(s)) for env, s in zip(envs, seeds)]
    runner.reset()
    done = np.zeros(n, bool)
    t_success = [None] * n
    max_cov = np.array([o["coverage"] for o in obs])
    above95 = np.zeros(n, int)
    success95 = np.zeros(n, bool)
    steps = np.zeros(n, int)
    last_images = {c: [None] * n for c in runner.cameras}
    writers = {i: VideoWriter(p, VIDEO_RES[1] * len(video_cams), VIDEO_RES[0]) for i, p in video_paths.items()}
    inference_s, plans = 0.0, 0

    for tick in range(n_ticks):
        # observations of every episode (finished ones: keep their last images)
        states, env_states = [], []
        for i, env in enumerate(envs):
            bx, by, yaw = env.block_pose()
            states.append(env.tip_xyz()[:2])
            env_states.append([bx, by, np.cos(yaw), np.sin(yaw)])
            for c in runner.cameras:
                if not done[i] or last_images[c][i] is None:
                    last_images[c][i] = renderer.render(env.data, c, policy_res[c])
        batch = {"observation.state": np.array(states), "observation.environment_state": np.array(env_states)}
        for c in runner.cameras:
            batch[f"observation.images.{c}"] = np.stack(last_images[c])

        t0 = time.perf_counter()
        actions = runner.act(batch)  # (n, 2)
        dt = time.perf_counter() - t0
        if tick % runner.cfg.n_action_steps == 0:
            inference_s, plans = inference_s + dt, plans + 1

        for i, env in enumerate(envs):
            if done[i]:
                continue
            for _ in range(stride):
                obs[i] = env.step(actions[i])
                steps[i] += 1
                max_cov[i] = max(max_cov[i], obs[i]["coverage"])
                above95[i] = above95[i] + 1 if obs[i]["coverage"] >= 0.95 else 0
                success95[i] |= above95[i] >= hold
                if i in writers and steps[i] % video_every == 0:
                    tiles = [label(renderer.render(env.data, c, VIDEO_RES), c) for c in video_cams]
                    writers[i].add(label(np.concatenate(tiles, axis=1),
                                         f"seed {seeds[i]}  t={steps[i] / env.control_hz:4.1f}s  "
                                         f"coverage={obs[i]['coverage'] * 100:4.1f}%"))
                if obs[i]["success"]:
                    t_success[i] = steps[i] / env.control_hz
                    done[i] = True
                    break
        if (tick + 1) % (5 * runner.fps) == 0:  # progress every 5 s of sim time
            print(f"    t={(tick + 1) / runner.fps:4.0f}s  finished {int(done.sum())}/{n}", flush=True)
        if done.all():
            break

    for w in writers.values():
        w.close()
    plan_ms = 1000 * inference_s / max(1, plans)
    return [{
        "seed": int(seeds[i]),
        "success": t_success[i] is not None,
        "success95": bool(success95[i]),
        "time_to_success": t_success[i],
        "final_coverage": float(obs[i]["coverage"]),
        "max_coverage": float(max_cov[i]),
        "sim_time": steps[i] / env0.control_hz,
        "plan_ms_batch": plan_ms,
    } for i in range(n)]


def evaluate_checkpoint(run_dir: Path, ckpt: str | None, args) -> dict:
    runner = PolicyRunner(run_dir, ckpt, inference_steps=args.inference_steps)
    ckpt_name = runner.path.parent.name
    if args.inference_steps:
        ckpt_name += f"_ddim{args.inference_steps}"  # keep results of both samplers apart
    env, policy_cams, policy_res, video_cams, export = setup_env(runner)
    renderer = CameraRenderer(env.model, {**video_cams, **policy_cams})
    out = run_dir / "eval" / ckpt_name
    (out / "videos").mkdir(parents=True, exist_ok=True)

    print(f"\n== {run_dir.name} / checkpoint {ckpt_name} | inputs {runner.input_keys} | "
          f"{args.episodes} episodes, seeds {args.seed}..{args.seed + args.episodes - 1}")
    # All envs share one compiled model, so a single renderer serves every episode.
    envs = [env] + [PushTEnv(cameras=None, xml=env.xml, model=env.model) for _ in range(args.batch - 1)]
    seeds = np.arange(args.seed, args.seed + args.episodes)
    results = []
    t_start = time.time()
    for chunk in range(0, args.episodes, args.batch):
        chunk_seeds = seeds[chunk:chunk + args.batch]
        vids = {i: out / "videos" / f"seed{s}.mp4" for i, s in enumerate(chunk_seeds) if chunk + i < args.videos}
        rs = run_batch(envs[:len(chunk_seeds)], runner, renderer, policy_res, video_cams,
                       chunk_seeds, args.max_time, vids)
        for i, r in enumerate(rs):
            if i in vids:
                vids[i].rename(vids[i].with_name(f"seed{r['seed']}_{'success' if r['success'] else 'fail'}.mp4"))
            print(f"  seed {r['seed']}  {'SUCCESS' if r['success'] else 'fail   '}"
                  f"  max cov {r['max_coverage'] * 100:5.1f}%  t={r['sim_time']:5.1f}s")
        results += rs
    print(f"  ({time.time() - t_start:.0f} s wall time)")
    renderer.close()

    ts = [r["time_to_success"] for r in results if r["success"]]
    summary = {
        "run": run_dir.name,
        "checkpoint": ckpt_name,
        "inputs": runner.input_keys,
        "episodes": len(results),
        "success_threshold": env.success_cov,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "success95_rate": float(np.mean([r["success95"] for r in results])),
        "mean_max_coverage": float(np.mean([r["max_coverage"] for r in results])),
        "mean_final_coverage": float(np.mean([r["final_coverage"] for r in results])),
        "mean_time_to_success": float(np.mean(ts)) if ts else None,
        "plan_ms_per_batch": float(np.mean([r["plan_ms_batch"] for r in results])),
        "seed_start": args.seed,
        "max_time": args.max_time,
        "sampler": f"DDIM {args.inference_steps}" if args.inference_steps else "DDPM (training default)",
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"  -> success {summary['success_rate'] * 100:.0f}%  (>=95%: {summary['success95_rate'] * 100:.0f}%)"
          f"  mean max coverage {summary['mean_max_coverage'] * 100:.1f}%   saved to {out}")
    return summary


def evaluate_in_parallel(run_dir: Path, ckpts: list[str], args) -> list[dict]:
    """One child process per checkpoint, `args.jobs` at a time. The physics of every episode
    runs on the CPU, so this spreads the work over several cores (and fills the GPU more)."""
    import subprocess

    passthrough = ["--episodes", str(args.episodes), "--seed", str(args.seed), "--max-time", str(args.max_time),
                   "--videos", str(args.videos), "--batch", str(args.batch)]
    if args.inference_steps:
        passthrough += ["--inference-steps", str(args.inference_steps)]
    suffix = f"_ddim{args.inference_steps}" if args.inference_steps else ""
    pending, running, t0 = list(ckpts), {}, time.time()
    print(f"Evaluating {len(ckpts)} checkpoints, {args.jobs} at a time; per-checkpoint logs in "
          f"{run_dir / 'eval' / '<checkpoint>' / 'log.txt'}")
    while pending or running:
        while pending and len(running) < args.jobs:
            c = pending.pop(0)
            out = run_dir / "eval" / (c + suffix)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / "log.txt", "w")
            cmd = [sys.executable, "-W", "ignore", __file__, str(run_dir), "--checkpoint", c, *passthrough]
            # Few CPU threads per job: many jobs x all-cores thread pools only fight each other.
            env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
            running[c] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), out)
        time.sleep(2)
        for c, (proc, out) in list(running.items()):
            if proc.poll() is not None:
                del running[c]
                if proc.returncode == 0:
                    r = json.loads((out / "results.json").read_text())
                    print(f"  {c}: success {r['success_rate'] * 100:3.0f}%  (>=95%: {r['success95_rate'] * 100:3.0f}%)"
                          f"  max cov {r['mean_max_coverage'] * 100:5.1f}%   [{time.time() - t0:.0f} s]", flush=True)
                else:
                    print(f"  {c}: FAILED (exit {proc.returncode}), see {out / 'log.txt'}", flush=True)
    summaries = []
    for c in ckpts:
        f = run_dir / "eval" / (c + suffix) / "results.json"
        if f.exists():
            summaries.append(json.loads(f.read_text()))
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="outputs/<run_name>")
    ap.add_argument("--checkpoint", default=None, help="e.g. 050000 (default: newest)")
    ap.add_argument("--all", action="store_true", help="evaluate every checkpoint of the run")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=100_000, help="first seed (demos used random 31-bit seeds)")
    ap.add_argument("--max-time", type=float, default=40.0, help="seconds per episode")
    ap.add_argument("--videos", type=int, default=6, help="save videos of the first N episodes")
    ap.add_argument("--batch", type=int, default=50, help="episodes run in parallel (one GPU call plans for all)")
    ap.add_argument("--jobs", type=int, default=3,
                    help="with --all: checkpoints evaluated at the same time. Each job needs ~2.3 GB RAM and "
                         "~1.1 GB GPU memory; more than 3 makes a 16 GB laptop swap")
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="use the faster DDIM sampler with N steps (e.g. 10) instead of the training DDPM")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    ckpts = list_checkpoints(run_dir) if args.all else [args.checkpoint]
    if len(ckpts) > 1 and args.jobs > 1:
        summaries = evaluate_in_parallel(run_dir, ckpts, args)
    else:
        summaries = [evaluate_checkpoint(run_dir, c, args) for c in ckpts]

    if len(summaries) > 1:
        print(f"\n{'checkpoint':<12}{'success':>9}{'>=95%':>8}{'max cov':>9}")
        for s in summaries:
            print(f"{s['checkpoint']:<12}{s['success_rate'] * 100:>8.0f}%{s['success95_rate'] * 100:>7.0f}%"
                  f"{s['mean_max_coverage'] * 100:>8.1f}%")
        (run_dir / "eval" / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
