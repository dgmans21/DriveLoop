"""시나리오 출발 지점 고르기 (앞차 52 · 보행자 53 공용).

공정한 비교를 위해 실행마다 같은 출발 지점: 시드 순서로 스폰 지점을 돌며 앞쪽 경로가 곧은 곳(방향 변화 < 기준)을
고른다. 경로는 같은 시드의 RoutePlanner로 만든다 → 내 차(DrivingAgent)와 교차로 분기 선택이 같다.
"""
from __future__ import annotations

import random

import carla

from driveloop.eval.scenarios import heading_change
from driveloop.planning.route import RoutePlanner


def same_dir_neighbor(wp: carla.Waypoint) -> carla.Waypoint | None:
    """같은 방향 옆 차선 (끼어들기 출발 차선)."""
    for n in (wp.get_left_lane(), wp.get_right_lane()):
        if n is not None and n.lane_type == carla.LaneType.Driving and n.lane_id * wp.lane_id > 0:
            return n
    return None


def sidewalk_offset(wp: carla.Waypoint, max_shoulder: float = 1.5) -> float | None:
    """차선 중심 → 오른쪽 보도 경계까지 거리. 내 차선이 맨 오른쪽 차선이고 사이 갓길 합이 max_shoulder 이하일 때만.

    ped_scen 첫 실행에서 고른 곳: 다리 위 3.5m 갓길(보도 없음 → 갓길에 선 사람), 고속도로 왼쪽 차선
    ('연석에 선 사람'이 실은 옆 차선 위) → 보행자 장면이 현실적이지 않았다.
    """
    off, n = wp.lane_width / 2, wp
    for _ in range(4):
        n = n.get_right_lane()
        if n is None or n.lane_type == carla.LaneType.Driving:
            return None
        if n.lane_type == carla.LaneType.Sidewalk:
            return off if off - wp.lane_width / 2 <= max_shoulder else None
        off += n.lane_width
    return None


def choose_start(world_map, points, seed: int, route_spacing: float, straight_length: float,
                 max_heading_change: float, need_adjacent: bool = False, name: str = "",
                 need_sidewalk: tuple[float, float] | None = None, distinct: bool = False,
                 route_seed: int | None = None):
    """→ (스폰 지점 번호, 경로 waypoint 열). 앞쪽이 곧고, (need_adjacent면) 같은 방향 옆 차선이 있고,
    (need_sidewalk=(시작 m, 끝 m)면) 경로의 그 구간 내내 바로 옆에 보도가 있는 곳.

    distinct=True: 시드 순서로 섞지 않고 '조건에 맞는 곳 중 seed번째'(1부터, 스폰 번호 순) → 후보가 적어도
    시드마다 다른 곳 (보행자 조건은 Town05에서 시나리오마다 3~5곳뿐이라 섞으면 시드끼리 겹쳤다).
    route_seed: 경로 분기 선택 시드 (기본 = seed). distinct면 고정해야 후보 목록이 시드마다 같다 → 내 차도 같은 값으로
    """
    rs = seed if route_seed is None else route_seed
    order = range(len(points)) if distinct else random.Random(seed).sample(range(len(points)), len(points))
    skip = seed - 1 if distinct else 0
    for idx in order:
        sp = points[idx]
        path = RoutePlanner(world_map, sp.location, route_spacing, straight_length + 60,
                            random.Random(rs)).waypoints
        if len(path) * route_spacing < straight_length:
            continue
        xy = [(w.transform.location.x, w.transform.location.y) for w in path]
        if heading_change(xy, straight_length) > max_heading_change:
            continue
        if need_adjacent:
            adj = [same_dir_neighbor(w) for w in path[: int(straight_length / route_spacing)]]
            if any(a is None for a in adj):
                continue
        if need_sidewalk is not None:
            a, b = (max(0, round(v / route_spacing)) for v in need_sidewalk)
            if b >= len(path) or any(sidewalk_offset(w) is None for w in path[a:b + 1]):
                continue
        if skip:
            skip -= 1
            continue
        return idx, path
    raise RuntimeError(f"{name}: 조건에 맞는 출발 지점 없음")
