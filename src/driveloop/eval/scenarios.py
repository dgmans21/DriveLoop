"""앞차 시나리오 정의 (CARLA 비의존, 단위 테스트 대상) — B-6 정답값 vs 모델 인지 비교용.

무작위 교통에서는 '앞차를 따라가는 상황'이 120초 중 1초 안팎이라 평가가 안 됐다 (acc_model_pilot).
그래서 앞차를 직접 연출한다: 앞차는 내 차와 같은 경로를 따라가고, 시간에 따라 정해진 속도·제동·차선으로 움직인다.

  follow  : 25m 앞에서 출발, 20 km/h로 계속 → 간격 유지·거리 추정 오차
  brake   : 25m 앞에서 7 m/s, 15초에 최대 제동으로 정지 → 충돌·최소 간격·비상 제동
  stopped : 50m 앞에 서 있는 차 → 부드럽게 멈추는지 (놓치면 위험)
  cutin   : 옆 차선 40m 앞에서 6 m/s, 7초에 내 차선으로 3초 동안 이동 → 반응·헛봄·놓침
            (첫 파일럿은 15m 앞·8초 시작 → 내 차(30 km/h)가 앞차(실제 ~5 m/s)를 이미 따라잡아 옆에 있을 때
             끼어들어 앞차가 한 번도 안 잡혔다. 40m·7초면 끼어들기가 끝날 때 범퍼 간격 약 14m)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Point = tuple[float, float]


@dataclass(frozen=True)
class LeadCommand:
    target_speed: float        # m/s
    full_brake: bool = False   # True면 속도 제어 대신 브레이크 최대 (급정거)
    lane_blend: float = 0.0    # 0 = 출발 차선(옆 차선), 1 = 내 차선. 끼어들기 외에는 1


@dataclass(frozen=True)
class Scenario:
    name: str
    lead_offset: float         # m, 내 차 위치에서 경로를 따라 앞차를 놓을 거리 (차 중심 기준)
    adjacent: bool = False     # 옆 차선에서 시작 (끼어들기)
    seconds: float = 35.0

    def command(self, t: float) -> LeadCommand:
        if self.name == "follow":
            return LeadCommand(20 / 3.6, lane_blend=1.0)
        if self.name == "brake":
            return LeadCommand(0.0, full_brake=True, lane_blend=1.0) if t >= 15.0 else LeadCommand(7.0, lane_blend=1.0)
        if self.name == "stopped":
            return LeadCommand(0.0, full_brake=True, lane_blend=1.0)
        if self.name == "cutin":
            blend = 0.0 if t < 7.0 else min(1.0, (t - 7.0) / 3.0)
            return LeadCommand(6.0, lane_blend=_smooth(blend))
        raise ValueError(f"알 수 없는 시나리오: {self.name}")


SCENARIOS = {
    "follow": Scenario("follow", 25.0 + 4.8),
    "brake": Scenario("brake", 25.0 + 4.8),
    "stopped": Scenario("stopped", 50.0 + 4.8),
    "cutin": Scenario("cutin", 40.0 + 4.8, adjacent=True),
}


def _smooth(x: float) -> float:
    """0→1을 부드럽게 (차선 변경이 꺾이지 않게)."""
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def heading_change(path: Sequence[Point], length: float) -> float:
    """경로 앞쪽 length m 동안 진행 방향이 바뀐 최대 각도(도). 직선 구간 고르기용."""
    if len(path) < 3:
        return 0.0
    h0 = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    s, worst = 0.0, 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        s += seg
        if seg > 1e-6:
            d = math.degrees(math.atan2(y1 - y0, x1 - x0) - h0)
            d = (d + 180) % 360 - 180
            worst = max(worst, abs(d))
        if s >= length:
            break
    return worst


def blend_paths(a: Sequence[Point], b: Sequence[Point], w: float) -> list[Point]:
    """같은 길이의 두 경로를 w(0=a, 1=b) 비율로 섞는다 (옆 차선 → 내 차선)."""
    return [(ax + (bx - ax) * w, ay + (by - ay) * w) for (ax, ay), (bx, by) in zip(a, b)]
