"""보행자 양보 판단 (CARLA 비의존, 단위 테스트 대상).

'멈출지'는 여기서, '얼마나 세게'는 ACC가 정한다: 양보할 보행자는 '서 있는 앞차'로 넘겨 acc_target을 재사용
(안전 속도는 즉시, 간격 항은 편안한 감속으로 — smooth_target).

양보 조건 (경로를 따라 잰 위치 s, 경로 중심선에서의 부호 있는 옆 거리 lat):
  1) 지금 내 통로 안 (|lat| ≤ corridor_half): 차 폭의 절반 + 여유. 걸어 나가는 중이어도 기다린다 (보수적)
  2) 통로 쪽으로 걸어오는 중 (옆 속도 ≥ min_lateral_speed)이고, 내가 그 지점을 다 지나가기 전에
     (+ buffer초) 통로에 들어올 것으로 예측되며, 진입까지 horizon초 이내
  보도에서 나란히 걷거나 서 있는 사람(옆 속도 ≈ 0), 이미 건너서 멀어지는 사람은 양보하지 않는다 → 불필요한 감속 0
유지(히스테리시스): 한 번 양보한 보행자는 통로 밖 release_margin을 더 벗어나거나, 통로 밖에서 멀어지는 중일 때만 해제
  → 통로 경계에서 정지/출발을 반복하지 않게.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from driveloop.perception.lead import Obstacle, Point, _cumulative, project_on_path


@dataclass(frozen=True)
class PedestrianYield:
    id: int
    distance: float      # m, 내 앞 범퍼 ~ 보행자 (경로를 따라 잼)
    lateral: float       # m, 경로 중심선에서의 옆 거리 (왼쪽 +)
    reason: str          # "in_path" | "entering" | "held"


@dataclass(frozen=True)
class PathCoords:
    gap: float           # m, 내 앞 범퍼 ~ 보행자 몸 앞 (경로를 따라 잼, 음수면 범퍼보다 뒤)
    lateral: float       # m, 경로 중심선에서의 부호 있는 옆 거리 (진행 방향 왼쪽이 +, 수학 좌표 기준)
    toward: float        # m/s, 통로 중심 쪽으로 다가오는 옆 속도 (+면 다가옴)


def path_coords(path: Sequence[Point], cum: Sequence[float], s_ego: float, ego_front_offset: float,
                o: Obstacle) -> PathCoords:
    """보행자를 경로 좌표로 (판단과 평가 로그가 같은 계산을 쓴다)."""
    s, _, i = project_on_path(path, cum, (o.x, o.y))
    (x0, y0), (x1, y1) = path[i], path[i + 1]
    seg = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / seg, (x1 - x0) / seg
    lat = (o.x - x0) * nx + (o.y - y0) * ny
    v_lat = o.vx * nx + o.vy * ny
    return PathCoords(s - s_ego - ego_front_offset - o.half_length, lat, -v_lat if lat > 0 else v_lat)


class PedestrianYielder:
    def __init__(self, corridor_half: float = 1.5, lookahead: float = 50.0, horizon: float = 4.0,
                 buffer: float = 1.0, min_lateral_speed: float = 0.3, release_margin: float = 0.5,
                 ego_length: float = 4.8, min_eval_speed: float = 2.0) -> None:
        self.corridor_half, self.lookahead, self.horizon, self.buffer = corridor_half, lookahead, horizon, buffer
        self.min_lateral_speed, self.release_margin = min_lateral_speed, release_margin
        self.ego_length, self.min_eval_speed = ego_length, min_eval_speed
        self._held: set[int] = set()

    def step(self, path: Sequence[Point], ego: Point, ego_front_offset: float,
             pedestrians: Sequence[Obstacle], speed: float) -> PedestrianYield | None:
        if len(path) < 2:
            self._held.clear()
            return None
        cum = _cumulative(path)
        s_ego, _, _ = project_on_path(path, cum, ego)
        v_eval = max(speed, self.min_eval_speed)     # 서 있을 때도 '출발하면 언제 도착하나'로 본다
        best: PedestrianYield | None = None
        held: set[int] = set()
        for o in pedestrians:
            c = path_coords(path, cum, s_ego, ego_front_offset, o)
            gap, lat, toward = c.gap, c.lateral, c.toward
            if gap < -ego_front_offset or gap > self.lookahead:   # 내 앞 범퍼보다 뒤(차 옆)는 이미 지나침
                continue
            outside = abs(lat) - self.corridor_half        # 통로 경계까지 남은 거리 (≤0이면 통로 안)
            reason = None
            if outside <= 0:
                reason = "in_path"
            elif toward >= self.min_lateral_speed:
                t_in = outside / toward
                t_pass = (max(gap, 0.0) + self.ego_length) / v_eval
                if t_in <= self.horizon and t_in <= t_pass + self.buffer:
                    reason = "entering"
            if reason is None and o.id in self._held and outside <= self.release_margin and toward > -0.1:
                reason = "held"
            if reason is None:
                continue
            held.add(o.id)
            gap = max(gap, 0.0)
            if best is None or gap < best.distance:
                best = PedestrianYield(o.id, gap, lat, reason)
        self._held = held
        return best
