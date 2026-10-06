"""차선을 따라가는 단순 경로: 전방 waypoint를 일정 길이만큼 유지한다.

교차로 분기에서는 무작위로 하나를 고른다 (목적지 기반 경로 탐색은 이후 필요 시).
"""
from __future__ import annotations

import random
from collections import deque

import carla


class RoutePlanner:
    def __init__(self, world_map: carla.Map, start: carla.Location, spacing: float,
                 horizon: float, rng: random.Random) -> None:
        self._spacing = spacing
        self._max_len = max(2, int(horizon / spacing))
        self._rng = rng
        self._wps: deque[carla.Waypoint] = deque([world_map.get_waypoint(start)])
        self._extend()

    def _extend(self) -> None:
        while len(self._wps) < self._max_len:
            candidates = self._wps[-1].next(self._spacing)
            if not candidates:
                break  # 막다른 길
            self._wps.append(self._rng.choice(candidates))

    def update(self, ego_location: carla.Location) -> None:
        """지나간 waypoint를 버리고 앞쪽을 다시 채운다."""
        while len(self._wps) > 1:
            d0 = self._wps[0].transform.location.distance(ego_location)
            d1 = self._wps[1].transform.location.distance(ego_location)
            if d1 <= d0:
                self._wps.popleft()
            else:
                break
        self._extend()

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def waypoints(self) -> list[carla.Waypoint]:
        return list(self._wps)

    def points_xy(self) -> list[tuple[float, float]]:
        return [(wp.transform.location.x, wp.transform.location.y) for wp in self._wps]
