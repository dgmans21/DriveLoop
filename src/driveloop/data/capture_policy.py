"""수집 저장 시점 결정 (CARLA 비의존).

평소엔 regular 간격, 화면에 노란불이 보이면 fast 간격으로 저장한다.
노란불은 신호 주기의 일부(기본 3초)라 0.5초 간격으로는 거의 안 찍힌다 (v1: 학습용 93개).
"""
from __future__ import annotations

import math

import numpy as np

from driveloop.data.autolabel import facing_angle
from driveloop.data.geometry import box_corners, box_to_2d


class CaptureScheduler:
    def __init__(self, regular_ticks: int, fast_ticks: int | None = None) -> None:
        self.regular_ticks = regular_ticks
        self.fast_ticks = fast_ticks
        self._last: int | None = None

    def wants_fast_check(self, tick: int) -> bool:
        """fast 저장이 가능한 시점인지 (노란불 판정 비용을 아끼려고 먼저 묻는다)."""
        return (self.fast_ticks is not None and self._last is not None
                and tick - self._last >= self.fast_ticks)

    def decide(self, tick: int, yellow_visible: bool = False) -> str | None:
        """tick: 워밍업 이후 0부터. 저장하면 사유('regular'/'yellow'), 아니면 None."""
        if self._last is None or tick - self._last >= self.regular_ticks:
            self._last = tick
            return "regular"
        if yellow_visible and self.wants_fast_check(tick):
            self._last = tick
            return "yellow"
        return None


def yellow_in_view(tl_states: dict[str, str], catalog: dict, camera_tf: dict, K: np.ndarray,
                   width: int, height: int, max_angle: float, max_dist: float) -> bool:
    """노란불 헤드가 카메라 앞, 화면 안, 정면 각도 이내에 있으면 True (가림은 보지 않음)."""
    for tl_id, state in tl_states.items():
        if state != "yellow":
            continue
        for head in catalog[tl_id]["light_boxes"]:
            dist = math.dist((head["x"], head["y"], head["z"]), (camera_tf["x"], camera_tf["y"], camera_tf["z"]))
            if dist > max_dist or facing_angle(head, camera_tf) > max_angle:
                continue
            if box_to_2d(box_corners(head), camera_tf, K, width, height) is not None:
                return True
    return False
