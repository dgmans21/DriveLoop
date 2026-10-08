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


# ---- C-2 보행자 시나리오 ------------------------------------------------------------------------------
# 좌표: 보행자 출발점 = 내 경로에서 ahead m 앞 waypoint 기준, u = 차선 중심에서 연석(오른쪽) 쪽으로 잰 옆 거리.
# 출발 옆 위치 = 기준선 + offset. 기준선은 보도 경계("sidewalk") 또는 내 차선 끝("edge"). 출발 지점은 맨 오른쪽
# 차선이고 바로 옆에 보도가 있는 곳만 (sim/scenario_start.sidewalk_offset).
# 보행자는 내 앞 범퍼가 trigger_gap m 안으로 들어오면 움직이기 시작한다 (시간이 아니라 거리 기준 → 내 속도가
# 실행마다 조금 달라도 '몇 m 앞에서 뛰어드는지'가 같다).
#
#   cross    : 50m 앞 보도(경계 +1m)에서 30m 남았을 때 1.4 m/s로 건넘 → 감속 후 통과
#   stop     : 같은 출발, 내 차선 가운데(u=0)에서 8초 멈춘 뒤 마저 건넘 → 정지 간격·대기·재출발
#   dartout  : 60m 앞 차선 끝 바로 밖(+0.35m, 주차 차 사이 같은 위치)에서 18m 남았을 때 3 m/s로 뛰어듦
#   sidewalk : 40m 앞 보도(경계 +1.5m)를 나를 향해 1.4 m/s로 걸음 → 감속하면 안 됨
#   curb     : 40m 앞 차선 끝 +0.5m(갓길)에 계속 서 있음 → 감속하면 안 됨

CROSS_END_U = -5.0             # m, 건너는 보행자는 차선 중심 반대쪽 5m(옆 차선 너머)에서 멈춘다


@dataclass(frozen=True)
class PedCommand:
    u_speed: float             # m/s, 연석 쪽(+u)으로의 속도. 음수 = 건너는 방향
    along_speed: float = 0.0   # m/s, 내 진행 방향 성분 (음수 = 나를 향해)


@dataclass(frozen=True)
class PedScenario:
    name: str
    ahead: float               # m, 내 출발 위치에서 경로를 따라 보행자 출발점까지 (차 중심 기준)
    anchor: str                # "sidewalk" = 보도 경계, "edge" = 내 차선 끝
    offset: float              # m, 기준선에서 바깥(연석 쪽)으로
    trigger_gap: float | None  # m, 내 앞 범퍼 ~ 보행자 거리가 이 안이면 출발 (None = 처음부터)
    yield_expected: bool       # 양보해야 정상인 시나리오인가 (아니면 감속이 곧 헛감속)
    seconds: float = 30.0

    def start_u(self, lane_width: float, sidewalk_u: float) -> float:
        return (sidewalk_u if self.anchor == "sidewalk" else lane_width / 2) + self.offset

    def command(self, since: float | None, u: float, u_start: float) -> PedCommand:
        """since = 출발 후 경과 초 (아직 출발 전이면 None), u = 지금 옆 위치, u_start = 출발 옆 위치."""
        if self.name == "sidewalk":
            return PedCommand(0.0, -1.4)
        if since is None or self.name == "curb":
            return PedCommand(0.0)
        if self.name == "cross":
            return PedCommand(-1.4 if u > CROSS_END_U else 0.0)
        if self.name == "stop":
            t_mid = u_start / 1.4                          # 차선 가운데(u=0)에 닿는 시각
            if since < t_mid:
                return PedCommand(-1.4)
            if since < t_mid + 8.0:
                return PedCommand(0.0)
            return PedCommand(-1.4 if u > CROSS_END_U else 0.0)
        if self.name == "dartout":
            return PedCommand(-3.0 if u > CROSS_END_U else 0.0)
        raise ValueError(f"알 수 없는 보행자 시나리오: {self.name}")


PED_SCENARIOS = {
    "cross": PedScenario("cross", 50.0, "sidewalk", 1.0, 30.0, True),
    "stop": PedScenario("stop", 50.0, "sidewalk", 1.0, 30.0, True),
    "dartout": PedScenario("dartout", 60.0, "edge", 0.35, 18.0, True),
    "sidewalk": PedScenario("sidewalk", 40.0, "sidewalk", 1.5, None, False),
    "curb": PedScenario("curb", 40.0, "edge", 0.5, None, False),
}
