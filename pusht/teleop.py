"""Top-down teleoperation view: renders the scene into a pygame window and maps the
mouse cursor to a tip position on the table."""

from __future__ import annotations

import numpy as np
import pygame

from .render import ThreadedCameraRenderer, pixel_to_plane, world_to_pixel

VIEW_CAM = "teleop_view"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
GREEN = (30, 170, 60)
ORANGE = (240, 150, 20)
BLUE = (40, 90, 220)


class TeleopView:
    def __init__(self, env, size: int, cameras: dict[str, dict], title: str = "Push-T teleop"):
        self.env = env
        self.size = size
        self.renderer = ThreadedCameraRenderer(env.model, cameras)
        pygame.init()
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 18)
        self.big = pygame.font.SysFont("dejavusans,sans", 42, bold=True)

    def mouse_to_table(self) -> np.ndarray:
        u, v = pygame.mouse.get_pos()
        return pixel_to_plane(self.env.model, self.env.data, VIEW_CAM, (u, v), self.size, self.size,
                              z=self.env.ctrl.tip_z)

    def to_px(self, xy) -> tuple[int, int]:
        p = world_to_pixel(self.env.model, self.env.data, VIEW_CAM,
                           [xy[0], xy[1], self.env.ctrl.tip_z], self.size, self.size)
        return int(p[0]), int(p[1])

    def draw(self, hud_lines: list[str], cursor_xy=None, cmd_xy=None, banner: str | None = None,
             banner_color=GREEN, coverage: float = 0.0, threshold: float = 0.95) -> None:
        img = self.renderer.render(self.env.data, VIEW_CAM, (self.size, self.size))
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
        self.screen.blit(surf, (0, 0))

        # Workspace boundary (where the tip is allowed to go).
        lo, hi = self.env.ctrl.ws_lo, self.env.ctrl.ws_hi
        corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
        pygame.draw.lines(self.screen, BLUE, True, [self.to_px(c) for c in corners], 1)

        tip = self.env.tip_xyz()[:2]
        pygame.draw.circle(self.screen, RED, self.to_px(tip), 7)
        if cmd_xy is not None:
            pygame.draw.circle(self.screen, ORANGE, self.to_px(cmd_xy), 4, 2)
        if cursor_xy is not None:
            c = self.to_px(cursor_xy)
            pygame.draw.circle(self.screen, BLACK, c, 10, 1)
            pygame.draw.line(self.screen, BLACK, (c[0] - 14, c[1]), (c[0] + 14, c[1]), 1)
            pygame.draw.line(self.screen, BLACK, (c[0], c[1] - 14), (c[0], c[1] + 14), 1)

        # Coverage bar.
        w = self.size - 20
        pygame.draw.rect(self.screen, (60, 60, 60), (10, self.size - 26, w, 16))
        pygame.draw.rect(self.screen, GREEN if coverage >= threshold else ORANGE,
                         (10, self.size - 26, int(w * min(coverage, 1.0)), 16))
        pygame.draw.line(self.screen, BLACK, (10 + int(w * threshold), self.size - 30),
                         (10 + int(w * threshold), self.size - 6), 2)

        y = 8
        for line in hud_lines:
            txt = self.font.render(line, True, BLACK)
            bg = pygame.Surface((txt.get_width() + 8, txt.get_height() + 2), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 170))
            self.screen.blit(bg, (6, y - 1))
            self.screen.blit(txt, (10, y))
            y += txt.get_height() + 3

        if banner:
            txt = self.big.render(banner, True, banner_color)
            self.screen.blit(txt, ((self.size - txt.get_width()) // 2, self.size // 2 - 30))
        pygame.display.flip()

    def close(self) -> None:
        self.renderer.close()
        pygame.quit()
