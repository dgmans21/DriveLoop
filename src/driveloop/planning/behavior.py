"""신호등 대응 판단 상태 머신 (CARLA 비의존, 단위 테스트 대상).

상태:
  CRUISE          순항 (신호 없음 / 초록불)
  STOPPING        정지선 앞 정지를 향해 감속 중 (빨간불, 또는 설 수 있는 노란불)
  STOPPED         정지선 앞 정지 완료, 초록불 대기
  PROCEED_YELLOW  노란불이지만 안전하게 설 수 없어 통과하기로 결정

노란불 판단: 정지 지점까지 남은 거리 gap에서 필요한 감속도 a = v^2 / (2*gap)가
max_stop_decel 이하이면 정지, 아니면 통과 (딜레마 존).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from driveloop.config import DrivingConfig
from driveloop.perception.types import PerceptionOutput, TLState

STOPPED_SPEED = 0.3   # m/s 이하면 정지로 간주
STOPPED_GAP = 1.0     # m, 정지 지점과 이만큼 이내


class BehaviorState(str, Enum):
    CRUISE = "CRUISE"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    PROCEED_YELLOW = "PROCEED_YELLOW"


@dataclass(frozen=True)
class Decision:
    state: BehaviorState
    target_speed: float   # m/s
    reason: str


def required_decel(speed: float, gap: float) -> float:
    """거리 gap 안에 속도 speed에서 정지하는 데 필요한 등감속도."""
    if gap <= 0:
        return math.inf if speed > STOPPED_SPEED else 0.0
    return speed * speed / (2.0 * gap)


def stopping_target_speed(gap: float, decel: float, cap: float) -> float:
    """남은 거리에서 감속도 decel로 정확히 멈출 수 있는 속도 v = sqrt(2*a*gap)."""
    return min(cap, math.sqrt(2.0 * decel * max(gap, 0.0)))


class TrafficLightBehavior:
    def __init__(self, cfg: DrivingConfig) -> None:
        self._cfg = cfg
        self.state = BehaviorState.CRUISE

    @property
    def cruise_speed(self) -> float:
        return self._cfg.cruise_speed_kmh / 3.6

    def step(self, p: PerceptionOutput, speed: float) -> Decision:
        reason = self._transition(p, speed)
        return Decision(self.state, self._target_speed(p), reason)

    def _transition(self, p: PerceptionOutput, speed: float) -> str:
        S = BehaviorState
        if p.tl_state is None or p.stop_distance is None:
            self.state = S.CRUISE
            return "no light ahead"

        tl = p.tl_state
        gap = p.stop_distance - self._cfg.stop_margin

        if tl is TLState.GREEN:
            self.state = S.CRUISE
            return "green"
        if tl is TLState.UNKNOWN:
            return "light unknown: keep decision"

        # 이하 RED / YELLOW
        if self.state in (S.STOPPING, S.STOPPED):
            if speed < STOPPED_SPEED and gap < STOPPED_GAP:
                self.state = S.STOPPED
                return f"{tl.value}: stopped, waiting"
            return f"{tl.value}: braking to line"

        if self.state is S.PROCEED_YELLOW:
            return f"{tl.value}: committed to pass"

        # CRUISE 상태에서 빨간불/노란불을 새로 만남
        if p.stop_distance < 0:
            return f"{tl.value}: already past line"
        need = required_decel(speed, gap)
        if tl is TLState.RED or need <= self._cfg.max_stop_decel:
            self.state = S.STOPPING
            return f"{tl.value}: stop (need {need:.1f} m/s2)"
        self.state = S.PROCEED_YELLOW
        return f"YELLOW: pass (need {need:.1f} m/s2)"

    def _target_speed(self, p: PerceptionOutput) -> float:
        S = BehaviorState
        if self.state is S.STOPPED:
            return 0.0
        if self.state is S.STOPPING:
            assert p.stop_distance is not None
            gap = p.stop_distance - self._cfg.stop_margin
            return stopping_target_speed(gap, self._cfg.comfort_decel, self.cruise_speed)
        return self.cruise_speed
