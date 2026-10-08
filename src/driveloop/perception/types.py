"""인지 모듈의 출력 형식.

판단(planning)은 이 형식만 보고 동작한다. 1단계는 시뮬레이터 정답값(ground_truth),
2단계는 카메라 기반 내 모델이 같은 형식을 채운다 → 판단/제어 코드는 그대로 재사용.
CARLA에 의존하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from driveloop.perception.lead import Obstacle


class TLState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"   # 꺼짐 / 모델이 확신 못 함


@dataclass(frozen=True)
class PerceptionOutput:
    tl_state: TLState | None = None      # None = 전방 경로에 신호등 없음
    stop_distance: float | None = None   # m, 앞 범퍼 ~ 정지선 (음수면 이미 넘음)
    lead_distance: float | None = None   # m, 내 경로 위 앞차와 범퍼 사이 간격 (None = 앞차 없음)
    lead_speed: float | None = None      # m/s, 앞차 속도의 경로 방향 성분
    lead_id: int | None = None           # 앞차 식별자 (같은 차를 계속 따라가는지 / 잠깐 가로지른 차인지 구분용)
    pedestrians: tuple[Obstacle, ...] = ()   # 주변 보행자 (월드 좌표·속도). 양보 여부는 판단(planning/pedestrian.py)이 정한다


class Perception(Protocol):
    def perceive(self, image=None) -> PerceptionOutput: ...
