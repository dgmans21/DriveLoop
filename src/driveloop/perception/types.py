"""인지 모듈의 출력 형식.

판단(planning)은 이 형식만 보고 동작한다. 1단계는 시뮬레이터 정답값(ground_truth),
2단계는 카메라 기반 내 모델이 같은 형식을 채운다 → 판단/제어 코드는 그대로 재사용.
CARLA에 의존하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class TLState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"   # 꺼짐 / 모델이 확신 못 함


@dataclass(frozen=True)
class PerceptionOutput:
    tl_state: TLState | None = None      # None = 전방 경로에 신호등 없음
    stop_distance: float | None = None   # m, 앞 범퍼 ~ 정지선 (음수면 이미 넘음)
    # TODO(1단계 다음): lead_distance / lead_speed (전방 장애물·차량)


class Perception(Protocol):
    def perceive(self, image=None) -> PerceptionOutput: ...
