"""Teleoperated demo collection.

    .venv/bin/python scripts/02_collect.py [--operator NAME] [--raw-dir data/raw]

Controls (top-down view, robot arm hidden, red dot = stick tip):
    Left mouse (hold)  stick tip follows the cursor. Recording starts on the first click.
    R                  discard the current attempt and reset to a new random start
    S                  stop and save the current attempt now (normally not needed)
    X                  mark the LAST SAVED episode as discarded (undo a bad save)
    P                  pause / resume
    Q / Esc            quit (an unfinished attempt is discarded)

An episode ends automatically (and is saved) once the T covers the target above the success
threshold (configs/collect.yaml, default 90%) for 0.5 s. Sessions can be stopped and resumed at any time; numbering continues.
"""

import argparse
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering/window env vars first)
import pygame  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.recorder import EpisodeRecorder, episode_attrs, list_episodes, set_attr  # noqa: E402
from pusht.scene import all_cameras  # noqa: E402
from pusht.teleop import GREEN, ORANGE, RED, TeleopView  # noqa: E402


def count_kept(raw_dir) -> int:
    n = 0
    for p in list_episodes(raw_dir):
        a = episode_attrs(p)
        n += bool(a.get("success")) and not bool(a.get("discard"))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", default="")
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()

    collect_cfg = load_config("collect")
    cams_cfg = load_config("cameras")
    raw_dir = resolve(args.raw_dir or collect_cfg["raw_dir"])
    env = PushTEnv(collect_cfg=collect_cfg, cameras=all_cameras(cams_cfg))
    rec = EpisodeRecorder(env, raw_dir)
    view = TeleopView(env, int(collect_cfg["teleop"]["window_size"]), all_cameras(cams_cfg))
    max_steps = int(float(collect_cfg["max_episode_time"]) * env.control_hz)
    thr = env.success_cov

    kept = count_kept(raw_dir)
    last_saved: Path | None = None
    banner, banner_color, banner_until = None, GREEN, 0.0

    def new_attempt():
        seed = secrets.randbits(31)
        env.reset(seed=seed)
        return seed, env.block_pose(), env.tip_xyz()[:2]

    seed, init_block, init_tip = new_attempt()
    paused, stopped, running = False, False, True
    clock = pygame.time.Clock()

    while running:
        now = time.time()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_r:
                    rec.discard()
                    stopped = False
                    seed, init_block, init_tip = new_attempt()
                    banner, banner_color, banner_until = "discarded", RED, now + 0.8
                elif ev.key == pygame.K_s and rec.active and len(rec) > 0:
                    ok = env.observe()["coverage"] >= thr
                    last_saved = rec.save(success=ok, note="manual stop")
                    kept += ok
                    stopped = False
                    seed, init_block, init_tip = new_attempt()
                    banner, banner_color, banner_until = f"saved {last_saved.stem}", ORANGE, now + 1.0
                elif ev.key == pygame.K_x and last_saved is not None:
                    a = episode_attrs(last_saved)
                    if not a.get("discard"):
                        set_attr(last_saved, "discard", True)
                        kept -= bool(a.get("success"))
                        banner, banner_color, banner_until = f"{last_saved.stem} discarded", RED, now + 1.2
                elif ev.key == pygame.K_p:
                    paused = not paused

        mouse_down = pygame.mouse.get_pressed()[0]
        cursor = view.mouse_to_table()

        if not paused and not stopped:
            if not rec.active and mouse_down:
                rec.start(seed, init_block, init_tip, operator=args.operator)
            if rec.active:
                state, obs = env.get_state(), env.observe()
                setpoint = cursor if mouse_down else env.ctrl.cmd_xy
                obs_next = env.step(setpoint)
                rec.record(state, obs, setpoint, mouse_down)
                if obs_next["success"]:
                    last_saved = rec.save(success=True)
                    kept += 1
                    banner, banner_color, banner_until = f"SUCCESS  ({kept} kept)", GREEN, now + 1.2
                    seed, init_block, init_tip = new_attempt()
                elif len(rec) >= max_steps:
                    stopped = True
                    banner, banner_color, banner_until = "time up: S=save R=discard", RED, now + 1e9

        cov = env.observe()["coverage"]
        status = "PAUSED" if paused else ("REC" if rec.active else "ready - click to start")
        hud = [
            f"{status}   t={len(rec) / env.control_hz:5.1f}s   coverage={cov * 100:5.1f}%   {clock.get_fps():4.0f} Hz",
            f"kept (successful): {kept}    last: {last_saved.stem if last_saved else '-'}",
            "hold LMB: move | R reset | S save | X undo last | P pause | Q quit",
        ]
        view.draw(hud, cursor_xy=cursor, cmd_xy=env.ctrl.cmd_xy if rec.active else None,
                  banner=banner if now < banner_until else None, banner_color=banner_color,
                  coverage=cov, threshold=thr)
        clock.tick(env.control_hz)

    rec.discard()
    view.close()
    print(f"Session ended. Successful kept episodes in {raw_dir}: {count_kept(raw_dir)}")


if __name__ == "__main__":
    main()
