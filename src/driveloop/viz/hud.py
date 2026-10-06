"""pygame 화면 + 텍스트 오버레이."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pygame

WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
RED = (255, 70, 70)
YELLOW = (255, 210, 60)
GREEN = (80, 230, 110)
CYAN = (90, 210, 255)

HudLine = tuple[str, tuple[int, int, int]]


class Display:
    def __init__(self, width: int, height: int, title: str = "DriveLoop") -> None:
        pygame.init()
        self.size = (width, height)
        # SCALED: 렌더링 해상도는 그대로 두고 창 크기만 늘림 (창 테두리 드래그/최대화, F11 전체화면)
        self.screen = pygame.display.set_mode(self.size, pygame.SCALED | pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.clock = pygame.time.Clock()

    def poll_quit(self) -> bool:
        """창 닫기 또는 ESC면 True. F11은 전체화면 전환."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                pygame.display.toggle_fullscreen()
        return False

    def draw(self, rgb: np.ndarray, lines: Sequence[HudLine] = ()) -> None:
        surface = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))
        self.screen.blit(surface, (0, 0))
        if lines:
            line_h = self.font.get_linesize()
            panel = pygame.Surface((340, line_h * len(lines) + 16), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 150))
            self.screen.blit(panel, (8, 8))
            for i, (text, color) in enumerate(lines):
                self.screen.blit(self.font.render(text, True, color), (16, 16 + i * line_h))
        pygame.display.flip()
        self.clock.tick()

    @property
    def fps(self) -> float:
        return self.clock.get_fps()

    def close(self) -> None:
        pygame.quit()
