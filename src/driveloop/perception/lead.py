"""내 경로 위 '앞차' 찾기 (CARLA 비의존, 단위 테스트 대상).

경로(내 차선 중심선 waypoint 열)를 따라 잰 거리로 판단한다 — 직선거리로 재면 커브에서 옆 차선 차를
앞차로 오인하거나, 코너 너머 차를 가깝게 본다 (정지선 거리와 같은 이유).

  - 경로 중심선에서 옆으로 lane_half_width 이내에 차 중심이 있으면 내 차선의 차
  - 거리 = 경로를 따라 잰 위치 차이 − 내 앞 범퍼 오프셋 − 상대 차 길이의 절반  (범퍼 사이 간격)
  - 속도 = 상대 차 속도를 그 지점 경로 방향으로 투영 (마주 오는 차는 음수)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Point = tuple[float, float]


@dataclass(frozen=True)
class Obstacle:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    half_length: float = 2.4


@dataclass(frozen=True)
class Lead:
    id: int
    distance: float      # m, 범퍼 사이 간격
    speed: float         # m/s, 경로 방향 성분


def _cumulative(path: Sequence[Point]) -> list[float]:
    s = [0.0]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))
    return s


def project_on_path(path: Sequence[Point], cum: Sequence[float], p: Point) -> tuple[float, float, int]:
    """점 p를 경로에 투영 → (경로 위치 s, 경로에서 옆 거리, 구간 번호)."""
    best = (math.inf, 0.0, 0)
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(path, path[1:])):
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / seg2))
        qx, qy = x0 + t * dx, y0 + t * dy
        lat = math.hypot(p[0] - qx, p[1] - qy)
        if lat < best[0]:
            best = (lat, cum[i] + t * math.sqrt(seg2), i)
    lat, s, i = best
    return s, lat, i


class LeadTracker:
    """카메라 앞차 측정(거리만) → 확정된 앞차 + 속도 추정.

    - 확정: confirm_frames 연속으로 비슷한 거리(max_jump 이내)에 보여야 앞차로 인정 → 오검출 1프레임에 급제동하지 않게
    - 유지: max_missed 프레임까지 놓쳐도 마지막 추정을 (속도로 앞당겨) 유지 → 순간 검출 누락에 출발하지 않게
    - 속도: 앞차 속도 = 내 속도 + 거리 변화율, 지수 평활(alpha)로 흔들림 줄임
    - id: 새 앞차로 바뀔 때마다 증가 (같은 차를 계속 따라가는지 로그로 구분)
    """

    def __init__(self, dt: float, confirm_frames: int = 2, max_missed: int = 3, max_jump: float = 4.0,
                 alpha: float = 0.3) -> None:
        self.dt, self.confirm_frames, self.max_missed = dt, confirm_frames, max_missed
        self.max_jump, self.alpha = max_jump, alpha
        self._dist: float | None = None
        self._speed: float | None = None
        self._seen = 0
        self._missed = 0
        self._id = 0

    def reset(self) -> None:
        self._dist, self._speed, self._seen, self._missed = None, None, 0, 0

    def update(self, measured: float | None, ego_speed: float) -> Lead | None:
        if measured is None:
            if self._dist is None:
                return None
            self._missed += 1
            if self._missed > self.max_missed:
                self.reset()
                return None
            if self._speed is not None:                   # 놓친 동안은 상대 속도로 거리를 앞당겨 둔다
                self._dist = max(0.0, self._dist + (self._speed - ego_speed) * self.dt)
            return self._confirmed()
        if self._dist is None or abs(measured - self._dist) > self.max_jump:
            self._id += 1                                  # 새 앞차 (처음이거나 거리가 갑자기 바뀜)
            self._dist, self._speed, self._seen, self._missed = measured, None, 1, 0
            return self._confirmed()
        rate = (measured - self._dist) / self.dt           # 거리 변화율 = 앞차 속도 − 내 속도
        v_meas = ego_speed + rate
        self._speed = v_meas if self._speed is None else self._speed + self.alpha * (v_meas - self._speed)
        self._dist, self._seen, self._missed = measured, self._seen + 1, 0
        return self._confirmed()

    def _confirmed(self) -> Lead | None:
        if self._seen < self.confirm_frames:
            return None
        return Lead(self._id, self._dist, self._speed if self._speed is not None else 0.0)


def find_lead(path: Sequence[Point], ego: Point, ego_front_offset: float, obstacles: Sequence[Obstacle],
              lane_half_width: float = 1.75, max_distance: float = 50.0) -> Lead | None:
    if len(path) < 2:
        return None
    cum = _cumulative(path)
    s_ego, _, _ = project_on_path(path, cum, ego)
    best: Lead | None = None
    for o in obstacles:
        s, lat, i = project_on_path(path, cum, (o.x, o.y))
        if lat > lane_half_width or s <= s_ego:
            continue
        gap = s - s_ego - ego_front_offset - o.half_length
        if gap > max_distance:
            continue
        (x0, y0), (x1, y1) = path[i], path[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        speed = (o.vx * (x1 - x0) + o.vy * (y1 - y0)) / seg if seg > 1e-6 else 0.0
        if best is None or gap < best.distance:
            best = Lead(o.id, gap, speed)
    return best
