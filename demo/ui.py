"""Small pygame UI toolkit for the simulation demo: theme, buttons, cards, bars, graph."""

from __future__ import annotations

import numpy as np
import pygame

# ---------------------------------------------------------------- theme
BG = (236, 238, 242)
CARD = (255, 255, 255)
BORDER = (214, 218, 226)
TEXT = (28, 32, 40)
MUTED = (110, 118, 132)
ACCENT = (44, 110, 230)
GREEN = (34, 160, 84)
RED = (214, 56, 56)
ORANGE = (240, 140, 20)
DARK = (48, 52, 60)
WHITE = (255, 255, 255)


class Fonts:
    def __init__(self):
        self.small = pygame.font.SysFont("dejavusans,sans", 13)
        self.body = pygame.font.SysFont("dejavusans,sans", 15)
        self.bold = pygame.font.SysFont("dejavusans,sans", 15, bold=True)
        self.title = pygame.font.SysFont("dejavusans,sans", 12, bold=True)
        self.big = pygame.font.SysFont("dejavusans,sans", 22, bold=True)
        self.huge = pygame.font.SysFont("dejavusans,sans", 44, bold=True)
        self.mono = pygame.font.SysFont("dejavusansmono,monospace", 14)


def text(surf, s, pos, font, color=TEXT, center=False, right=False):
    img = font.render(s, True, color)
    x, y = pos
    if center:
        x -= img.get_width() // 2
    elif right:
        x -= img.get_width()
    surf.blit(img, (x, y))
    return img.get_width()


def card(surf, rect, title: str | None, fonts: Fonts) -> pygame.Rect:
    """White rounded panel with a small caps title; returns the inner content rect."""
    pygame.draw.rect(surf, CARD, rect, border_radius=8)
    pygame.draw.rect(surf, BORDER, rect, width=1, border_radius=8)
    top = rect.y + 8
    if title:
        text(surf, title.upper(), (rect.x + 12, rect.y + 8), fonts.title, MUTED)
        top = rect.y + 26
    return pygame.Rect(rect.x + 12, top, rect.w - 24, rect.bottom - top - 8)


def bar(surf, rect, value: float, threshold: float | None = None, color=None):
    pygame.draw.rect(surf, (225, 228, 234), rect, border_radius=5)
    if color is None:
        color = GREEN if threshold is not None and value >= threshold else ORANGE
    w = int(rect.w * float(np.clip(value, 0, 1)))
    if w > 0:
        pygame.draw.rect(surf, color, (rect.x, rect.y, w, rect.h), border_radius=5)
    if threshold is not None:
        x = rect.x + int(rect.w * threshold)
        pygame.draw.line(surf, DARK, (x, rect.y - 3), (x, rect.bottom + 2), 2)


def graph(surf, rect, values, threshold: float | None, fonts: Fonts, t_max: float, dt: float):
    """Coverage-over-time line graph (values in [0, 1])."""
    pygame.draw.rect(surf, (248, 249, 251), rect, border_radius=4)
    for frac in (0.25, 0.5, 0.75):
        y = rect.bottom - int(rect.h * frac)
        pygame.draw.line(surf, (232, 234, 238), (rect.x, y), (rect.right, y))
    if threshold is not None:
        y = rect.bottom - int(rect.h * threshold)
        pygame.draw.line(surf, GREEN, (rect.x, y), (rect.right, y), 1)
    if len(values) > 1:
        n_max = max(len(values), int(t_max / dt))
        pts = [(rect.x + int(rect.w * i / n_max), rect.bottom - int(rect.h * float(np.clip(v, 0, 1))))
               for i, v in enumerate(values)]
        pygame.draw.lines(surf, ACCENT, False, pts, 2)
    text(surf, "100%", (rect.x + 3, rect.y + 2), fonts.small, MUTED)
    text(surf, f"{t_max:.0f} s", (rect.right - 3, rect.bottom - 16), fonts.small, MUTED, right=True)


class Button:
    def __init__(self, rect, label: str, key: str = "", accent: bool = False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.key = key
        self.accent = accent
        self.active = False  # toggles: drawn highlighted when on

    def hit(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def draw(self, surf, fonts: Fonts, mouse_pos):
        hover = self.hit(mouse_pos)
        if self.active or self.accent:
            bg, fg = (ACCENT if not self.active else DARK), WHITE
            if hover:
                bg = tuple(min(255, c + 25) for c in bg)
        else:
            bg, fg = ((242, 244, 248) if not hover else (228, 233, 242)), TEXT
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, BORDER, self.rect, width=1, border_radius=6)
        text(surf, self.label, (self.rect.centerx, self.rect.y + 6), fonts.bold, fg, center=True)
        if self.key:
            text(surf, self.key, (self.rect.centerx, self.rect.y + 25), fonts.small,
                 (220, 226, 240) if fg == WHITE else MUTED, center=True)


class Tab:
    def __init__(self, rect, label: str):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, surf, fonts: Fonts, selected: bool, mouse_pos):
        hover = self.rect.collidepoint(mouse_pos)
        bg = ACCENT if selected else ((228, 233, 242) if hover else (245, 246, 249))
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        text(surf, self.label, (self.rect.centerx, self.rect.y + 8), fonts.body,
             WHITE if selected else TEXT, center=True)
