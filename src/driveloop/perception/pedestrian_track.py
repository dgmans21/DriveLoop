"""카메라 보행자 측정(월드 위치) → 여러 명 추적 + 속도 추정 (CARLA 비의존, 단위 테스트 대상).

앞차 추적(LeadTracker)은 한 대만 거리 1차원으로 따라가지만, 보행자는 여러 명이고 '차도 쪽으로 다가오는 옆 속도'가
양보 판단(planning/pedestrian.py)의 핵심이라 월드 평면 2차원으로 추적한다.

  - 연결: 예측 위치에서 가장 가까운 측정을 max_jump m 안에서 짝짓는다 (가까운 쌍부터, 한 측정은 한 트랙에만)
  - 필터: 축마다 알파-베타 (위치 p, 속도 v). 예측 p' = p + v·dt, 잔차 e, p = p' + a·e, v = v + (b/dt)·e
  - 확정: confirm_frames 연속으로 보여야 판단에 넘긴다 → 오검출 1프레임에 급제동하지 않게
  - 유지: max_missed 프레임까지 놓쳐도 예측 위치로 유지 → 순간 검출 누락에 출발하지 않게
    단, establish_frames 이상 꾸준히 본 보행자는 hold_missed 프레임(2초)까지 유지 — '놓친 보행자 = 사라진 보행자'가
    아니다 (ped_scen_model_v1: 비 오는 밤 차선 가운데 9m 앞 보행자를 1초 놓침 → 3프레임 만에 지움 → 양보 해제·재가속
    → 5.4m에서 다시 보고 최대 제동). 잠깐 보였다 사라진 오검출은 여전히 3프레임에 지운다 → 헛양보는 늘지 않음
    놓친 동안 속도는 coast_decay로 줄인다: 멈춘 사람을 계속 걸어가는 것으로 끌고 가지 않게
    (v2에서 1초 → v3에서 2초: 비 오는 밤 차선 가운데 보행자를 약 1.05초 놓쳐 1초로는 빠듯했음)
  - 새 트랙은 속도 0으로 시작: 서 있는 사람으로 보고, 걸어오면 몇 프레임 안에 속도가 붙는다
    (통로 안의 보행자는 속도와 무관하게 양보하므로 '모르는 속도'가 위험한 쪽으로 작용하지 않는다)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from driveloop.perception.lead import Obstacle


@dataclass
class _Track:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    seen: int = 1
    missed: int = 0

    def max_missed(self, tracker: "PedestrianTracker") -> int:
        return tracker.hold_missed if self.seen >= tracker.establish_frames else tracker.max_missed


class PedestrianTracker:
    def __init__(self, dt: float, confirm_frames: int = 2, max_missed: int = 3, max_jump: float = 1.5,
                 a: float = 0.4, b: float = 0.05, radius: float = 0.3, establish_frames: int = 10,
                 hold_missed: int = 40, coast_decay: float = 0.9) -> None:
        self.dt, self.confirm_frames, self.max_missed, self.max_jump = dt, confirm_frames, max_missed, max_jump
        self.a, self.b, self.radius = a, b, radius
        self.establish_frames, self.hold_missed, self.coast_decay = establish_frames, hold_missed, coast_decay
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks = []

    def update(self, measurements: list[tuple[float, float]]) -> tuple[Obstacle, ...]:
        """measurements: 이번 프레임 보행자 발 위치 (월드 x, y). → 확정된 보행자 (속도 포함)."""
        preds = [(t.x + t.vx * self.dt, t.y + t.vy * self.dt) for t in self._tracks]
        pairs = sorted((math.dist(p, m), i, j) for i, p in enumerate(preds) for j, m in enumerate(measurements))
        used_t, used_m = set(), set()
        for d, i, j in pairs:
            if d > self.max_jump or i in used_t or j in used_m:
                continue
            used_t.add(i)
            used_m.add(j)
            t, (px, py), (mx, my) = self._tracks[i], preds[i], measurements[j]
            ex, ey = mx - px, my - py
            t.x, t.y = px + self.a * ex, py + self.a * ey
            t.vx += (self.b / self.dt) * ex
            t.vy += (self.b / self.dt) * ey
            t.seen, t.missed = t.seen + 1, 0
        kept = []
        for i, t in enumerate(self._tracks):
            if i not in used_t:
                t.missed += 1
                if t.missed > t.max_missed(self):
                    continue
                t.x, t.y = preds[i]                           # 놓친 동안은 예측 위치로
                t.vx, t.vy = t.vx * self.coast_decay, t.vy * self.coast_decay
            kept.append(t)
        for j, (mx, my) in enumerate(measurements):
            if j not in used_m:
                kept.append(_Track(self._next_id, mx, my))
                self._next_id += 1
        self._tracks = kept
        return tuple(Obstacle(t.id, t.x, t.y, t.vx, t.vy, self.radius)
                     for t in self._tracks if t.seen >= self.confirm_frames)
