"""시뮬레이터 API 정답값으로 만든 인지 (1단계 기준선)."""
from __future__ import annotations

from collections import defaultdict

import carla

from driveloop.perception.lead import Lead, Obstacle, find_lead
from driveloop.perception.types import PerceptionOutput, TLState
from driveloop.planning.route import RoutePlanner

_TL_STATE = {
    carla.TrafficLightState.Red: TLState.RED,
    carla.TrafficLightState.Yellow: TLState.YELLOW,
    carla.TrafficLightState.Green: TLState.GREEN,
}

# 정지선을 이만큼 넘어간 신호등은 '지나간 것'으로 보고 무시
PASSED_TOLERANCE = 2.0


class GroundTruthPerception:
    """자차 경로 위에 있는 가장 가까운 신호등의 상태와 정지선까지 거리를 돌려준다.

    시작 시 맵의 모든 신호등 정지선 waypoint를 (road_id, lane_id)로 색인해 두고,
    매 프레임 경로가 지나는 차선과 매칭한다.
    """

    def __init__(self, world: carla.World, ego: carla.Vehicle, route: RoutePlanner,
                 lookahead: float, lead_lookahead: float = 50.0, lane_half_width: float = 1.75) -> None:
        self._world = world
        self._lead_lookahead = lead_lookahead
        self._lane_half_width = lane_half_width
        self._ego = ego
        self._route = route
        self._lookahead = lookahead
        self._front_offset = ego.bounding_box.extent.x  # 차량 중심 → 앞 범퍼
        self._stops: dict[tuple[int, int], list[tuple[carla.TrafficLight, carla.Waypoint]]] = defaultdict(list)
        for tl in world.get_actors().filter("traffic.traffic_light"):
            for wp in tl.get_stop_waypoints():
                self._stops[(wp.road_id, wp.lane_id)].append((tl, wp))

    @property
    def num_traffic_lights(self) -> int:
        return len({tl.id for items in self._stops.values() for tl, _ in items})

    def perceive(self, image=None) -> PerceptionOutput:
        lead = self.lead_vehicle()
        ld, ls, lid = (lead.distance, lead.speed, lead.id) if lead else (None, None, None)
        nearest = self.nearest_light()
        if nearest is None:
            return PerceptionOutput(lead_distance=ld, lead_speed=ls, lead_id=lid)
        tl, dist = nearest
        return PerceptionOutput(_TL_STATE.get(tl.get_state(), TLState.UNKNOWN), dist, ld, ls, lid)

    def lead_vehicle(self) -> Lead | None:
        """정답값 앞차: 시뮬레이터의 모든 차량 위치·속도 → 내 경로 위 가장 가까운 차 (perception/lead.py)."""
        obstacles = []
        for v in self._world.get_actors().filter("vehicle.*"):
            if v.id == self._ego.id:
                continue
            loc, vel = v.get_location(), v.get_velocity()
            obstacles.append(Obstacle(v.id, loc.x, loc.y, vel.x, vel.y, v.bounding_box.extent.x))
        loc = self._ego.get_location()
        return find_lead(self._route.points_xy(), (loc.x, loc.y), self._front_offset, obstacles,
                         self._lane_half_width, self._lead_lookahead)

    def nearest_light(self) -> tuple[carla.TrafficLight, float] | None:
        """지도 정보만으로 '내 차선을 통제하는 가장 가까운 신호등'과 정지선까지 거리 (상태는 읽지 않음).

        모델 인지(ModelPerception)도 이 부분을 재사용한다 — 위치는 지도, 색은 모델.
        정지선까지 거리는 직선이 아니라 경로를 따라 잰다 (코너 너머 신호를 가깝게 오인하지 않도록).
        """
        tf = self._ego.get_transform()
        fwd = tf.get_forward_vector()
        wps = self._route.waypoints
        match_radius = self._route.spacing
        best: tuple[carla.TrafficLight, float] | None = None

        # 자차 → 첫 waypoint까지는 진행 방향 투영, 이후는 waypoint 사이 거리 누적
        first = wps[0].transform.location
        arc = (first.x - tf.location.x) * fwd.x + (first.y - tf.location.y) * fwd.y
        for i, wp in enumerate(wps):
            wl = wp.transform.location
            if i > 0:
                arc += wps[i - 1].transform.location.distance(wl)
            if arc - self._front_offset > self._lookahead:
                break
            for tl, stop_wp in self._stops.get((wp.road_id, wp.lane_id), ()):
                sl = stop_wp.transform.location
                if sl.distance(wl) > match_radius:
                    continue
                wf = wp.transform.get_forward_vector()
                offset = (sl.x - wl.x) * wf.x + (sl.y - wl.y) * wf.y  # waypoint → 정지선 (진행방향)
                dist = arc + offset - self._front_offset
                if dist < -PASSED_TOLERANCE or dist > self._lookahead:
                    continue
                if best is None or dist < best[1]:
                    best = (tl, dist)

        return best
