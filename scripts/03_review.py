"""Review recorded demos: summary table and/or visual replay with keep/discard marking.

    .venv/bin/python scripts/03_review.py --summary            # table only, no window
    .venv/bin/python scripts/03_review.py                      # replay window
    .venv/bin/python scripts/03_review.py --start 12 --camera front

Replay controls:
    Space        play / pause           Left/Right   previous / next episode
    D            toggle discard flag    Up/Down      replay speed x2 / /2
    C            cycle camera           Home         restart episode
    Q / Esc      quit
Discarded episodes stay on disk but are skipped by the exporter.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pusht  # noqa: E402,F401  (sets rendering env vars before mujoco loads)
import numpy as np  # noqa: E402

from pusht.config import load_config, resolve  # noqa: E402
from pusht.recorder import episode_attrs, list_episodes, load_episode, set_attr  # noqa: E402


def summary(paths) -> None:
    print(f"{'episode':<16}{'steps':>7}{'secs':>7}{'success':>9}{'final_cov':>11}{'discard':>9}  note")
    tot = kept = 0
    secs = []
    for p in paths:
        a = episode_attrs(p)
        s = a["n_steps"] / a["control_hz"]
        ok = bool(a["success"]) and not bool(a["discard"])
        kept += ok
        tot += 1
        if ok:
            secs.append(s)
        print(f"{p.stem:<16}{a['n_steps']:>7}{s:>7.1f}{str(bool(a['success'])):>9}"
              f"{a['final_coverage']:>11.3f}{str(bool(a['discard'])):>9}  {a.get('note', '')}")
    print(f"\n{tot} episodes on disk, {kept} usable (successful and not discarded)", end="")
    if secs:
        print(f", mean length {np.mean(secs):.1f}s, total {sum(secs) / 60:.1f} min")
    else:
        print()


def replay(paths, start: int, camera: str | None, size: int) -> None:
    import pygame

    from pusht.env import PushTEnv
    from pusht.render import ThreadedCameraRenderer
    from pusht.scene import all_cameras

    cams = all_cameras(load_config("cameras"))
    cam_names = list(cams)
    cam_i = cam_names.index(camera) if camera else cam_names.index("teleop_view")

    pygame.init()
    screen = pygame.display.set_mode((size, size))
    font = pygame.font.SysFont("dejavusansmono,monospace", 18)
    clock = pygame.time.Clock()

    idx = max(0, min(start, len(paths) - 1))
    ep = env = renderer = None
    t, playing, speed = 0, True, 1.0

    def load(i):
        nonlocal ep, env, renderer, t
        if renderer:
            renderer.close()
        ep = load_episode(paths[i])
        env = PushTEnv(cameras=cams, xml=ep["physics_xml"])  # the episode's own physics
        renderer = ThreadedCameraRenderer(env.model, cams)
        t = 0

    load(idx)
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    playing = not playing
                elif k == pygame.K_RIGHT and idx < len(paths) - 1:
                    idx += 1
                    load(idx)
                elif k == pygame.K_LEFT and idx > 0:
                    idx -= 1
                    load(idx)
                elif k == pygame.K_HOME:
                    t = 0
                elif k == pygame.K_UP:
                    speed = min(speed * 2, 16)
                elif k == pygame.K_DOWN:
                    speed = max(speed / 2, 0.25)
                elif k == pygame.K_c:
                    cam_i = (cam_i + 1) % len(cam_names)
                elif k == pygame.K_d:
                    new = not bool(episode_attrs(paths[idx])["discard"])
                    set_attr(paths[idx], "discard", new)

        n = len(ep["state"])
        env.set_state(ep["state"][min(int(t), n - 1)])
        name = cam_names[cam_i]
        img = renderer.render(env.data, name, (size, size))
        screen.blit(pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2))), (0, 0))

        a = episode_attrs(paths[idx])
        step = min(int(t), n - 1)
        lines = [
            f"{paths[idx].stem} ({idx + 1}/{len(paths)})  cam={name}  x{speed:g}",
            f"t={step / a['control_hz']:5.1f}/{n / a['control_hz']:.1f}s  cov={ep['coverage'][step] * 100:5.1f}%"
            f"  success={bool(a['success'])}",
            "DISCARDED" if a["discard"] else "kept",
        ]
        y = 8
        for i, line in enumerate(lines):
            color = (200, 30, 30) if (i == 2 and a["discard"]) else (0, 0, 0)
            screen.blit(font.render(line, True, color, (255, 255, 255)), (10, y))
            y += 22
        pygame.display.flip()

        if playing:
            t += speed
            if t >= n:
                t = n - 1
                playing = False
        clock.tick(a["control_hz"])

    renderer.close()
    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--summary", action="store_true", help="print the table and exit")
    ap.add_argument("--start", type=int, default=0, help="episode position to start the replay at")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--size", type=int, default=720)
    args = ap.parse_args()

    raw_dir = resolve(args.raw_dir or load_config("collect")["raw_dir"])
    paths = list_episodes(raw_dir)
    if not paths:
        sys.exit(f"No episodes in {raw_dir}")
    summary(paths)
    if not args.summary:
        replay(paths, args.start, args.camera, args.size)


if __name__ == "__main__":
    main()
