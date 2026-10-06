"""Pure pursuit 횡방향 제어 (CARLA 비의존).

CARLA 좌표계: x 전방, y 오른쪽, yaw는 x→y 방향(시계방향)으로 증가, 단위 degree.
VehicleControl.steer > 0 은 우회전.
"""
from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


def rear_axle(ego_xy: Point, ego_yaw_deg: float, wheelbase: float) -> Point:
    """차량 중심에서 wheelbase/2 뒤 = 후륜축. pure pursuit의 주시 거리는 여기서 재야 한다
    (차량 중심에서 재면 실제 주시 거리가 길어져 코너 안쪽을 더 깎는다 → 연석 접촉)."""
    yaw = math.radians(ego_yaw_deg)
    return ego_xy[0] - math.cos(yaw) * wheelbase / 2, ego_xy[1] - math.sin(yaw) * wheelbase / 2


def pick_lookahead_point(points: Sequence[Point], origin: Point, distance: float) -> Point:
    """경로 위에서 origin으로부터 distance 이상 떨어진 첫 점."""
    for p in points:
        if math.hypot(p[0] - origin[0], p[1] - origin[1]) >= distance:
            return p
    return points[-1]


def pure_pursuit_steer(ego_xy: Point, ego_yaw_deg: float, target: Point,
                       wheelbase: float, max_steer_rad: float) -> float:
    """목표점을 지나는 원호의 조향각을 [-1, 1] 범위 steer 명령으로 변환."""
    yaw = math.radians(ego_yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rx, ry = rear_axle(ego_xy, ego_yaw_deg, wheelbase)
    dx, dy = target[0] - rx, target[1] - ry
    local_x = cos_y * dx + sin_y * dy
    local_y = -sin_y * dx + cos_y * dy
    ld = math.hypot(local_x, local_y)
    if ld < 1e-3:
        return 0.0
    alpha = math.atan2(local_y, local_x)
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), ld)
    return max(-1.0, min(1.0, delta / max_steer_rad))
