"""Simulation demo: the trained policy pushes the T in MuJoCo, shown in an interactive window.

    .venv/bin/python demo/sim_demo.py                          # best evaluated checkpoint of outputs/dp_image
    .venv/bin/python demo/sim_demo.py --checkpoint 050000 --seed 100035
    .venv/bin/python demo/sim_demo.py --run outputs/dp_lowdim --ddpm

Left: live view (camera tabs at the bottom, or a 2x2 grid of cameras). Orange line = the
policy's plan (where it will move the stick tip next). Right: the model, what the policy
actually sees (its 96x96 input images), the episode (coverage bar + graph) and the session.

Mouse:  drag the T to move it (test how the policy reacts) · wheel over the T rotates it ·
        in the free camera, drag empty space to orbit and wheel to zoom · click a grid tile to open it
Keys:   Space pause · R new start · K kick the T · G grid · P plan · +/- speed · V record video ·
        S sampler (fast DDIM / training DDPM) · A auto-next · [ ] previous / next checkpoint ·
        1-8 camera · Q quit
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "demo"))

import pusht  # noqa: E402,F401  (GPU rendering env vars; must come before mujoco and pygame)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402
from shapely.geometry import Point  # noqa: E402

import ui  # noqa: E402
from evaluate import VideoWriter, setup_env  # noqa: E402
from policy_runner import (PolicyRunner, best_checkpoint, download_pretrained, eval_results,  # noqa: E402
                           list_checkpoints)
from pusht.config import load_config  # noqa: E402
from pusht.env import PushTEnv  # noqa: E402
from pusht.render import ThreadedCameraRenderer, pixel_to_plane, world_to_pixel  # noqa: E402

DEFAULT_RUN = "outputs/dp_image"
MAIN = 760                      # live view (square)
TAB_H = 40
PANEL_X, PANEL_W = MAIN + 12, 456
W, H = PANEL_X + PANEL_W + 12, MAIN + TAB_H
FREE = "free"
NO_ARM = "top (no arm)"
TAB_LABELS = {"overhead": "overhead", "corner": "corner R", "corner_left": "corner L", "top": "top",
              "side": "side", "wrist": "wrist", NO_ARM: "top no-arm", FREE: "free cam"}
GRID = ["overhead", "corner_left", "corner", NO_ARM]
FAST_STEPS = 10


class Demo:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run).resolve()
        self.ckpts = list_checkpoints(self.run_dir)
        if not self.ckpts and self.run_dir == (PROJECT_ROOT / DEFAULT_RUN).resolve():
            download_pretrained(self.run_dir)  # fresh clone: fetch the published model
            self.ckpts = list_checkpoints(self.run_dir)
        if not self.ckpts:
            sys.exit(f"No checkpoints in {self.run_dir}. Train first (training/train.py).")
        self.results = eval_results(self.run_dir)
        ckpt = args.checkpoint or best_checkpoint(self.run_dir) or self.ckpts[-1]
        self.fast = not args.ddpm
        self.load_policy(ckpt)

        # Env: physics of the recorded demos, every camera from configs/cameras.yaml + the policy cameras.
        env0, self.policy_cams, self.policy_res, _, _ = setup_env(self.runner)
        cams_cfg = load_config("cameras")
        self.view_cams = {**cams_cfg["cameras"], NO_ARM: {**cams_cfg["teleop_view"], "hide_robot": True}}
        all_cams = {**self.view_cams, **self.policy_cams}
        self.env = PushTEnv(cameras=all_cams, xml=env0.xml)
        self.renderer = ThreadedCameraRenderer(self.env.model, all_cams)
        self.stride = int(round(self.env.control_hz / self.runner.fps))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")

        self.views = [v for v in TAB_LABELS if v in self.view_cams or v == FREE]
        self.view = "overhead" if "overhead" in self.views else self.views[0]
        self.grid = False
        self.free = mujoco.MjvCamera()
        self.free.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.free.lookat[:] = [0.51, 0.0, 0.05]
        self.free.distance, self.free.azimuth, self.free.elevation = 1.4, 160.0, -35.0

        self.seed = args.seed
        self.history: list[bool] = []
        self.speed, self.paused, self.show_plan, self.auto_next = 1, False, True, True
        self.banner, self.banner_color, self.banner_until, self.next_episode_at = None, ui.GREEN, 0.0, None
        self.recorder, self.rec_path, self.frame_i = None, None, 0
        self.drag = None          # ("block", offset_xy, view, tile_rect) | ("orbit", last_pos)
        self.new_episode()

    # ---------------------------------------------------------------- policy
    def load_policy(self, ckpt: str):
        self.runner = PolicyRunner(self.run_dir, ckpt, inference_steps=FAST_STEPS if self.fast else None)
        self.ckpt = self.runner.path.parent.name

    def switch_checkpoint(self, delta: int):
        i = (self.ckpts.index(self.ckpt) + delta) % len(self.ckpts) if self.ckpt in self.ckpts else 0
        self.flash(f"loading {self.ckpts[i]} ...", ui.ACCENT, 0.1)
        self.draw()
        pygame.display.flip()
        self.load_policy(self.ckpts[i])
        self.history.clear()
        self.new_episode()

    def toggle_sampler(self):
        self.fast = not self.fast
        self.load_policy(self.ckpt)
        self.new_episode()

    # ---------------------------------------------------------------- episode
    def new_episode(self):
        self.env.reset(seed=self.seed)
        self.runner.reset()
        self.step, self.action, self.got_action, self.future = 0, None, False, None
        self.plan, self.cov_trace, self.plan_ms = [], [], 0.0
        self.interactive, self.done = False, False
        self.obs_images = {c: self.renderer.render(self.env.data, c, self.policy_res[c]) for c in self.runner.cameras}
        self.next_episode_at = None

    def observe(self) -> dict:
        bx, by, yaw = self.env.block_pose()
        obs = {"observation.state": self.env.tip_xyz()[:2],
               "observation.environment_state": np.array([bx, by, np.cos(yaw), np.sin(yaw)])}
        for c in self.runner.cameras:
            self.obs_images[c] = self.renderer.render(self.env.data, c, self.policy_res[c])
            obs[f"observation.images.{c}"] = self.obs_images[c]
        return obs

    def advance(self) -> bool:
        """One 50 Hz control step. The policy runs on its own thread, so the window stays smooth
        while it plans; the simulation simply waits for the next action. Returns False if waiting."""
        if self.step % self.stride == 0 and not self.got_action:
            if self.future is None:
                self.t_submit = time.perf_counter()
                self.future = self.pool.submit(self.runner.act, self.observe())
            if not self.future.done():
                return False
            self.action = self.future.result()
            ms = (time.perf_counter() - self.t_submit) * 1000
            if ms > 20:  # a re-plan (the other ticks just pop the next queued action)
                self.plan_ms = ms
            self.future, self.got_action = None, True
            self.plan = [self.action, *self.runner.planned_actions()]
        o = self.env.step(self.action)
        self.step += 1
        self.got_action = False
        if self.step % 5 == 0:
            self.cov_trace.append(o["coverage"])
        if o["success"] or self.step >= self.args.max_time * self.env.control_hz:
            self.finish(bool(o["success"]))
        return True

    def finish(self, success: bool):
        self.done = True
        self.history.append(success)
        t = self.step / self.env.control_hz
        self.flash(f"SUCCESS  in {t:.1f} s" if success else "time out", ui.GREEN if success else ui.RED, 2.0)
        self.next_episode_at = time.time() + 2.0 if self.auto_next else None
        self.paused = not self.auto_next

    def flash(self, msg, color, seconds):
        self.banner, self.banner_color, self.banner_until = msg, color, time.time() + seconds

    def kick(self):
        rng = np.random.default_rng()
        bx, by, yaw = self.env.block_pose()
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.03, 0.05)
        self.env.set_block_pose([bx + d * np.cos(a), by + d * np.sin(a), yaw + rng.choice([-1, 1]) * rng.uniform(0.26, 0.52)])
        mujoco.mj_forward(self.env.model, self.env.data)
        self.interactive = True
        self.flash("kicked!", ui.ORANGE, 0.6)

    # ---------------------------------------------------------------- geometry helpers
    def tiles(self) -> list[tuple[str, pygame.Rect]]:
        if not self.grid:
            return [(self.view, pygame.Rect(0, 0, MAIN, MAIN))]
        s = MAIN // 2
        names = [v for v in GRID if v in self.view_cams]
        return [(n, pygame.Rect((i % 2) * s, (i // 2) * s, s, s)) for i, n in enumerate(names)]

    def tile_at(self, pos):
        for name, rect in self.tiles():
            if rect.collidepoint(pos):
                return name, rect
        return None, None

    def table_point(self, name, rect, pos):
        if name == FREE:
            return None
        uv = (pos[0] - rect.x, pos[1] - rect.y)
        return pixel_to_plane(self.env.model, self.env.data, name, uv, rect.h, rect.w, z=self.env.shape.height / 2)

    def over_block(self, p) -> bool:
        return p is not None and self.env.shape.polygon(*self.env.block_pose()).distance(Point(p)) < 0.015

    # ---------------------------------------------------------------- input
    def build_buttons(self):
        y0, bw, bh, g = H - 3 * 46 - 10, (PANEL_W - 3 * 8) // 4, 42, 8
        spec = [
            [("pause", "Pause", "Space"), ("new", "New start", "R"), ("kick", "Kick T", "K"), ("grid", "Grid", "G")],
            [("plan", "Plan", "P"), ("speed", "Speed x1", "+ / -"), ("rec", "Record", "V"), ("auto", "Auto-next", "A")],
            [("sampler", "Sampler", "S"), ("prev", "◀ Ckpt", "["), ("next", "Ckpt ▶", "]"), ("quit", "Quit", "Q")],
        ]
        self.buttons = {}
        for r, row in enumerate(spec):
            for c, (key, label, hint) in enumerate(row):
                self.buttons[key] = ui.Button((PANEL_X + c * (bw + g), y0 + r * (bh + 4), bw, bh), label, hint)
        tw = (MAIN - 7 * 4) // len(self.views)
        self.tabs = [(v, ui.Tab((i * (tw + 4), MAIN + 4, tw, TAB_H - 8), TAB_LABELS[v])) for i, v in enumerate(self.views)]

    def press(self, key: str):
        if key == "pause":
            if self.done and self.next_episode_at is None:  # finished (auto-next off): Play = next start
                self.seed += 1
                self.new_episode()
                self.paused = False
            else:
                self.paused = not self.paused
        elif key == "new":
            self.seed += 1
            self.new_episode()
            self.paused = False
        elif key == "kick":
            self.kick()
        elif key == "grid":
            self.grid = not self.grid
        elif key == "plan":
            self.show_plan = not self.show_plan
        elif key == "speed":
            self.speed = self.speed * 2 if self.speed < 8 else 1
        elif key == "rec":
            self.toggle_recording()
        elif key == "auto":
            self.auto_next = not self.auto_next
        elif key == "sampler":
            self.toggle_sampler()
        elif key == "prev":
            self.switch_checkpoint(-1)
        elif key == "next":
            self.switch_checkpoint(+1)
        elif key == "quit":
            self.running = False

    def handle_events(self):
        keymap = {pygame.K_SPACE: "pause", pygame.K_r: "new", pygame.K_k: "kick", pygame.K_g: "grid",
                  pygame.K_p: "plan", pygame.K_v: "rec", pygame.K_a: "auto", pygame.K_s: "sampler",
                  pygame.K_LEFTBRACKET: "prev", pygame.K_RIGHTBRACKET: "next", pygame.K_q: "quit",
                  pygame.K_ESCAPE: "quit"}
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in keymap:
                    self.press(keymap[ev.key])
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.speed = min(self.speed * 2, 8)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.speed = max(self.speed // 2, 1)
                elif pygame.K_1 <= ev.key <= pygame.K_8 and ev.key - pygame.K_1 < len(self.views):
                    self.view, self.grid = self.views[ev.key - pygame.K_1], False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self.mouse_down(ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.drag = None
            elif ev.type == pygame.MOUSEMOTION and self.drag:
                self.mouse_drag(ev.pos)
            elif ev.type == pygame.MOUSEWHEEL:
                self.mouse_wheel(pygame.mouse.get_pos(), ev.y)

    def mouse_down(self, pos):
        for key, b in self.buttons.items():
            if b.hit(pos):
                return self.press(key)
        for v, tab in self.tabs:
            if tab.rect.collidepoint(pos):
                self.view, self.grid = v, False
                return
        name, rect = self.tile_at(pos)
        if name is None:
            return
        p = self.table_point(name, rect, pos)
        if self.over_block(p):
            self.drag = ("block", self.env.block_pose()[:2] - p, name, rect)
        elif self.grid:
            self.view, self.grid = name, False
        elif name == FREE:
            self.drag = ("orbit", pos)

    def mouse_drag(self, pos):
        if self.drag[0] == "orbit":
            last = self.drag[1]
            self.free.azimuth -= 0.3 * (pos[0] - last[0])
            self.free.elevation = float(np.clip(self.free.elevation - 0.3 * (pos[1] - last[1]), -89, -5))
            self.drag = ("orbit", pos)
        else:
            _, offset, name, rect = self.drag
            p = self.table_point(name, rect, pos)
            if p is not None:
                xy = p + offset
                self.env.set_block_pose([xy[0], xy[1], self.env.block_pose()[2]])
                mujoco.mj_forward(self.env.model, self.env.data)
                self.interactive = True

    def mouse_wheel(self, pos, dy):
        name, rect = self.tile_at(pos)
        p = self.table_point(name, rect, pos) if name else None
        if self.over_block(p):
            bx, by, yaw = self.env.block_pose()
            self.env.set_block_pose([bx, by, yaw + np.radians(5) * dy])
            mujoco.mj_forward(self.env.model, self.env.data)
            self.interactive = True
        elif name == FREE:
            self.free.distance = float(np.clip(self.free.distance * (0.9 if dy > 0 else 1.1), 0.3, 4.0))

    # ---------------------------------------------------------------- recording
    def toggle_recording(self):
        if self.recorder is None:
            out = self.run_dir / "demo_videos"
            out.mkdir(parents=True, exist_ok=True)
            self.rec_path = out / f"demo_{dt.datetime.now():%Y%m%d_%H%M%S}.mp4"
            self.recorder = VideoWriter(self.rec_path, W, H)
            self.frame_i = 0
        else:
            self.recorder.close()
            self.flash(f"saved {self.rec_path.name}", ui.ACCENT, 2.0)
            print(f"recorded {self.rec_path}")
            self.recorder = None

    # ---------------------------------------------------------------- drawing
    def draw(self):
        s, f, mouse = self.screen, self.fonts, pygame.mouse.get_pos()
        s.fill(ui.BG)
        data = self.env.data

        # live view(s)
        for name, rect in self.tiles():
            if name == FREE:
                img = self.renderer.render_free(data, self.free, (rect.h, rect.w))
            else:
                img = self.renderer.render(data, name, (rect.h, rect.w))
            s.blit(pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2))), rect.topleft)
            if self.show_plan and name != FREE and len(self.plan) > 1:
                pts = [world_to_pixel(self.env.model, data, name, [x, y, self.env.ctrl.tip_z], rect.h, rect.w)
                       for x, y in self.plan]
                pts = [(int(u) + rect.x, int(v) + rect.y) for u, v in pts]
                pygame.draw.lines(s, ui.ORANGE, False, pts, 3)
                for i, p in enumerate(pts):
                    pygame.draw.circle(s, ui.ORANGE, p, 6 if i == 0 else 3)
            label = TAB_LABELS.get(name, name)
            lw = f.bold.size(label)[0]
            pygame.draw.rect(s, (255, 255, 255), (rect.x + 8, rect.y + 8, lw + 14, 24), border_radius=5)
            ui.text(s, label, (rect.x + 15, rect.y + 11), f.bold)
            if self.grid:
                pygame.draw.rect(s, ui.BG, rect, width=2)
        # coverage strip over the live view
        cov = self.env.observe()["coverage"]
        thr = self.env.success_cov
        ui.bar(s, pygame.Rect(12, MAIN - 22, MAIN - 24, 10), cov, thr)

        for v, tab in self.tabs:
            tab.draw(s, f, v == self.view and not self.grid, mouse)

        # ---- right panel
        x, y = PANEL_X, 12
        # model card
        inner = ui.card(s, pygame.Rect(x, y, PANEL_W, 96), "Model", f)
        ui.text(s, f"{self.run_dir.name}  ·  checkpoint {self.ckpt}", inner.topleft, f.big)
        r_ddpm = self.results.get(self.ckpt)
        r_fast = self.results.get(f"{self.ckpt}_ddim{FAST_STEPS}")
        ev = [f"{r_ddpm['success_rate'] * 100:.0f}% (training sampler)" if r_ddpm else None,
              f"{r_fast['success_rate'] * 100:.0f}% (fast)" if r_fast else None]
        ev = "  ·  ".join(e for e in ev if e) or "not evaluated"
        ui.text(s, f"evaluated success: {ev}", (inner.x, inner.y + 30), f.body, ui.MUTED)
        sampler = f"fast DDIM {FAST_STEPS}" if self.fast else "training DDPM 100"
        ui.text(s, f"sampler: {sampler}   ·   last plan {self.plan_ms:.0f} ms", (inner.x, inner.y + 50), f.body, ui.MUTED)
        y += 104

        # policy input card
        n = max(1, len(self.runner.cameras))
        inner = ui.card(s, pygame.Rect(x, y, PANEL_W, 236), "What the policy sees (96×96)", f)
        if self.runner.cameras:
            tw = min(200, (inner.w - (n - 1) * 12) // n)
            for i, c in enumerate(self.runner.cameras):
                img = pygame.surfarray.make_surface(np.ascontiguousarray(self.obs_images[c].transpose(1, 0, 2)))
                s.blit(pygame.transform.scale(img, (tw, tw)), (inner.x + i * (tw + 12), inner.y))
                ui.text(s, c, (inner.x + i * (tw + 12), inner.y + tw + 2), f.small, ui.MUTED)
        else:
            ui.text(s, "block pose + tip position (no images)", inner.topleft, f.body)
        y += 244

        # episode card
        inner = ui.card(s, pygame.Rect(x, y, PANEL_W, 190), "Episode", f)
        t = self.step / self.env.control_hz
        status = ("finished" if self.done else "PAUSED" if self.paused else "planning…" if self.future is not None and not self.future.done()
                  else "running")
        ui.text(s, f"seed {self.seed}   t = {t:4.1f} s   {status}" + ("   (you moved the T)" if self.interactive else ""),
                inner.topleft, f.body)
        ui.text(s, f"coverage {cov * 100:5.1f}%", (inner.x, inner.y + 24), f.bold)
        ui.text(s, f"goal ≥ {thr * 100:.0f}% for {self.env.collect_cfg['success_hold_time']} s",
                (inner.right, inner.y + 24), f.small, ui.MUTED, right=True)
        ui.bar(s, pygame.Rect(inner.x, inner.y + 46, inner.w, 12), cov, thr)
        ui.graph(s, pygame.Rect(inner.x, inner.y + 66, inner.w, inner.h - 68), self.cov_trace, thr, f,
                 self.args.max_time, 5 / self.env.control_hz)
        y += 198

        # session card
        inner = ui.card(s, pygame.Rect(x, y, PANEL_W, 72), "Session", f)
        ok, tot = sum(self.history), len(self.history)
        ui.text(s, f"{ok}/{tot} successful" + (f"  ({100 * ok / tot:.0f}%)" if tot else ""), inner.topleft, f.bold)
        ui.text(s, f"speed x{self.speed}", (inner.right, inner.y), f.body, ui.MUTED, right=True)
        for i, res in enumerate(self.history[-26:]):
            pygame.draw.circle(s, ui.GREEN if res else ui.RED, (inner.x + 7 + i * 16, inner.y + 30), 6)

        # buttons (labels reflect state)
        self.buttons["pause"].label = "Play" if self.paused else "Pause"
        self.buttons["speed"].label = f"Speed x{self.speed}"
        self.buttons["rec"].label = "● Stop rec" if self.recorder else "Record"
        self.buttons["sampler"].label = "Fast" if self.fast else "Training"
        for key, active in (("grid", self.grid), ("plan", self.show_plan), ("auto", self.auto_next),
                            ("rec", self.recorder is not None)):
            self.buttons[key].active = active
        for b in self.buttons.values():
            b.draw(s, f, mouse)

        if self.recorder:
            pygame.draw.circle(s, ui.RED, (MAIN - 22, 22), 8)
            ui.text(s, "REC", (MAIN - 36, 32), f.bold, ui.RED, right=True)
        if self.banner and time.time() < self.banner_until:
            img = f.huge.render(self.banner, True, self.banner_color)
            box = img.get_rect(center=(MAIN // 2, MAIN // 2))
            pygame.draw.rect(s, (255, 255, 255), box.inflate(40, 24), border_radius=12)
            s.blit(img, box)

    # ---------------------------------------------------------------- main loop
    def run(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Push-T · Diffusion Policy · simulation demo")
        self.fonts = ui.Fonts()
        self.build_buttons()
        clock = pygame.time.Clock()
        self.running = True
        while self.running:
            self.handle_events()
            now = time.time()
            if self.next_episode_at and now >= self.next_episode_at:
                self.seed += 1
                self.new_episode()
            elif not self.paused and not self.done and self.drag is None:
                for _ in range(self.speed):
                    if not self.advance() or self.next_episode_at is not None or self.paused:
                        break
            self.draw()
            pygame.display.flip()
            if self.recorder:
                self.frame_i += 1
                if self.frame_i % 2 == 0:  # 50 Hz loop -> 25 fps video
                    frame = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
                    self.recorder.add(np.ascontiguousarray(frame))
            clock.tick(self.env.control_hz)
        if self.recorder:
            self.toggle_recording()
        self.pool.shutdown(wait=True)
        self.renderer.close()
        pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN, help="training run folder (default: downloads the "
                    "pretrained model if it has no checkpoints)")
    ap.add_argument("--checkpoint", default=None, help="default: best evaluated checkpoint")
    ap.add_argument("--seed", type=int, default=100_000, help="first random start (same seeds as evaluate.py)")
    ap.add_argument("--max-time", type=float, default=40.0, help="seconds before an attempt counts as failed")
    ap.add_argument("--ddpm", action="store_true", help="start with the training sampler (slower, ~1 s per plan)")
    args = ap.parse_args()
    Demo(args).run()


if __name__ == "__main__":
    main()
