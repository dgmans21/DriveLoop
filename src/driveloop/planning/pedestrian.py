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

주의 서행 (C-7 v3, 양보와 별개의 속도 상한 caution_speed):
  '차도 위'(보도 경계 안쪽)에서 내 통로 가까이(통로 밖 caution_width 안)에 있는 보행자 → 그 사람이 지금 뛰어들어도
  인지 지연 caution_latency초 뒤 caution_decel로 그 앞 caution_margin(= ACC 정지 간격 5m)에 설 수 있는 속도:
      v·t + v²/(2a) ≤ s   →   v = −a·t + sqrt((a·t)² + 2·a·s),  s = 간격 − margin,  최저 caution_min_speed (멈추지 않음)
  보도 위(연석 위)의 사람은 대상 아님 → 보도를 걷거나 서 있기만 해도 감속하지 않는다.
  근거 (ped_scen_model_v2): 18m 앞 차도 가장자리에서 3 m/s로 뛰어든 보행자를 카메라는 약 0.4초 늦게 알아챔
  (보이는 몸이 물리 몸체보다 늦게 움직임 + 추적 필터) → 4회 모두 최대 제동. 보도와 차도의 경계는 지도(road_edge)에서.
  margin을 정지 간격과 같게 둬야 한다: v3에서 1m로 두니 '1m 앞에 설 수 있는 속도'로 다가갔다가 뛰어들면 ACC는 '5m 앞에
  서라'고 해서 그 4m만큼 급제동 (1차원 모의: margin 1 → 4.4~8.2 m/s², margin 5 → 2.6~3.0 m/s²)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

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
    seg: int = 0         # 가장 가까운 경로 구간 번호 (지도 조회용: 그 지점의 보도 경계)


def path_coords(path: Sequence[Point], cum: Sequence[float], s_ego: float, ego_front_offset: float,
                o: Obstacle) -> PathCoords:
    """보행자를 경로 좌표로 (판단과 평가 로그가 같은 계산을 쓴다)."""
    s, _, i = project_on_path(path, cum, (o.x, o.y))
    (x0, y0), (x1, y1) = path[i], path[i + 1]
    seg = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / seg, (x1 - x0) / seg
    lat = (o.x - x0) * nx + (o.y - y0) * ny
    v_lat = o.vx * nx + o.vy * ny
    return PathCoords(s - s_ego - ego_front_offset - o.half_length, lat, -v_lat if lat > 0 else v_lat, i)


class PedestrianYielder:
    def __init__(self, corridor_half: float = 1.5, lookahead: float = 50.0, horizon: float = 4.0,
                 buffer: float = 1.0, min_lateral_speed: float = 0.3, release_margin: float = 0.5,
                 ego_length: float = 4.8, min_eval_speed: float = 2.0, caution_width: float = 2.0,
                 caution_decel: float = 3.0, caution_latency: float = 0.5, caution_margin: float = 5.0,
                 caution_min_speed: float = 3.0) -> None:
        self.corridor_half, self.lookahead, self.horizon, self.buffer = corridor_half, lookahead, horizon, buffer
        self.min_lateral_speed, self.release_margin = min_lateral_speed, release_margin
        self.ego_length, self.min_eval_speed = ego_length, min_eval_speed
        self.caution_width, self.caution_decel, self.caution_latency = caution_width, caution_decel, caution_latency
        self.caution_margin, self.caution_min_speed = caution_margin, caution_min_speed
        self._held: set[int] = set()
        self.caution_speed: float | None = None     # 마지막 step의 주의 서행 상한 (없으면 None)
        self.caution_id: int | None = None

    def caution_limit(self, gap: float) -> float:
        """지금 뛰어들어도 (인지 지연 t 뒤 감속 a로) 간격 − margin 안에 설 수 있는 최대 속도."""
        a, t = self.caution_decel, self.caution_latency
        s = max(0.0, gap - self.caution_margin)
        return max(self.caution_min_speed, -a * t + math.sqrt((a * t) ** 2 + 2 * a * s))

    def step(self, path: Sequence[Point], ego: Point, ego_front_offset: float,
             pedestrians: Sequence[Obstacle], speed: float,
             road_edge: Callable[[int, int], float | None] | None = None) -> PedestrianYield | None:
        """road_edge(구간 번호, 쪽 ±1) → 그쪽 보도 경계까지 옆 거리 (없거나 모르면 None = 차도로 본다)."""
        self.caution_speed, self.caution_id = None, None
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
                if 0 < outside <= self.caution_width and toward > -0.1 and gap > 0:
                    edge = road_edge(c.seg, 1 if lat > 0 else -1) if road_edge else None
                    if edge is None or abs(lat) < edge:          # 보도 경계 안쪽 = 차도 위
                        v = self.caution_limit(gap)
                        if self.caution_speed is None or v < self.caution_speed:
                            self.caution_speed, self.caution_id = v, o.id
                continue
            held.add(o.id)
            gap = max(gap, 0.0)
            if best is None or gap < best.distance:
                best = PedestrianYield(o.id, gap, lat, reason)
        self._held = held
        return best
