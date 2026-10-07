"""신호등 대응 판단 상태 머신 (CARLA 비의존, 단위 테스트 대상).

상태:
  CRUISE          순항 (신호 없음 / 초록불)
  CAUTION         신호가 있는데 색을 모름(UNKNOWN) → 정지선에서 편안하게 설 수 있는 속도로 제한
  STOPPING        정지선 앞 정지를 향해 감속 중 (빨간불, 또는 설 수 있는 노란불)
  STOPPED         정지선 앞 정지 완료, 초록불 대기 (CAUTION으로 정지선까지 와서 선 경우 포함)
  PROCEED_YELLOW  노란불이지만 안전하게 설 수 없어 통과하기로 결정

노란불 판단: 정지 지점까지 남은 거리 gap에서 필요한 감속도 a = v^2 / (2*gap)가
max_stop_decel 이하이면 정지, 아니면 통과 (딜레마 존).

안전장치 (3단계 비교에서 찾은 위험 장면, 2026-10-07):
  1) 색을 모르는 채로 딜레마 존에 들어가지 않는다 — UNKNOWN이면 CAUTION: 목표 속도를
     sqrt(2 · comfort_decel · gap)로 제한. 이후 빨강·노랑이 보여도 필요한 감속도가 항상 comfort_decel 이하.
     끝까지 모르면 정지선에서 선다 (안전 쪽). 신호가 멀면 제한 속도 > 순항 속도라 영향 없음.
  2) 초록 → 빨강은 실제 신호 순서에 없다 (초록 → 노랑 → 빨강). 초록을 본 지 dilemma_memory_steps 이내에
     빨강이 보였는데 이미 설 수 없는 거리면 노란불 딜레마로 보고 통과 (1프레임 오인으로 급제동하지 않게).
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
    CAUTION = "CAUTION"
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
        self._steps_since_green: int | None = None   # 같은 신호에서 마지막으로 초록을 본 뒤 지난 step 수
        self._prev_dist: float | None = None

    @property
    def cruise_speed(self) -> float:
        return self._cfg.cruise_speed_kmh / 3.6

    def step(self, p: PerceptionOutput, speed: float) -> Decision:
        self._track_green(p)
        reason = self._transition(p, speed)
        return Decision(self.state, self._target_speed(p), reason)

    def _track_green(self, p: PerceptionOutput) -> None:
        """신호가 바뀌면(없어짐 / 정지선 거리가 크게 늘어남) 초록 기억을 지운다."""
        d = p.stop_distance
        if d is None or (self._prev_dist is not None and d > self._prev_dist + 5.0):
            self._steps_since_green = None
        elif self._steps_since_green is not None:
            self._steps_since_green += 1
        if p.tl_state is TLState.GREEN:
            self._steps_since_green = 0
        self._prev_dist = d

    def _recent_green(self) -> bool:
        return self._steps_since_green is not None and self._steps_since_green <= self._cfg.dilemma_memory_steps

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
            # 방금(dilemma_memory_steps 이내) 초록을 봤으면 유지: 정지선 바로 앞에서 신호가 화면 밖으로 나가 UNKNOWN이 되는 경우.
            # 초록 → 빨강 사이에는 노란불(약 3초)이 있으므로 1초 기억은 신호 위반으로 이어지지 않는다.
            if self.state in (S.CRUISE, S.CAUTION) and p.stop_distance >= 0 and not self._recent_green():
                if self.state is S.CAUTION and speed < STOPPED_SPEED and gap < STOPPED_GAP:
                    self.state = S.STOPPED
                    return "light unknown: stopped at line (fail-safe)"
                self.state = S.CAUTION
                return "light unknown: limit speed to stop comfortably"
            return "light unknown: keep decision"

        # 이하 RED / YELLOW
        if self.state in (S.STOPPING, S.STOPPED):
            if speed < STOPPED_SPEED and gap < STOPPED_GAP:
                self.state = S.STOPPED
                return f"{tl.value}: stopped, waiting"
            return f"{tl.value}: braking to line"

        if self.state is S.PROCEED_YELLOW:
            return f"{tl.value}: committed to pass"

        # CRUISE / CAUTION 상태에서 빨간불/노란불을 새로 만남
        if p.stop_distance < 0:
            self.state = S.CRUISE
            return f"{tl.value}: already past line"
        need = required_decel(speed, gap)
        if need <= self._cfg.max_stop_decel:
            self.state = S.STOPPING
            return f"{tl.value}: stop (need {need:.1f} m/s2)"
        if tl is TLState.YELLOW:
            self.state = S.PROCEED_YELLOW
            return f"YELLOW: pass (need {need:.1f} m/s2)"
        # RED인데 설 수 없는 거리
        if self._recent_green():
            self.state = S.PROCEED_YELLOW
            return f"RED right after GREEN (impossible order): treat as yellow, pass (need {need:.1f} m/s2)"
        self.state = S.STOPPING
        return f"RED: stop (need {need:.1f} m/s2)"

    def _target_speed(self, p: PerceptionOutput) -> float:
        S = BehaviorState
        if self.state is S.STOPPED:
            return 0.0
        if self.state in (S.STOPPING, S.CAUTION):
            assert p.stop_distance is not None
            gap = p.stop_distance - self._cfg.stop_margin
            return stopping_target_speed(gap, self._cfg.comfort_decel, self.cruise_speed)
        return self.cruise_speed
