"""경로 곡률 기반 커브 속도 제한 (CARLA 비의존).

곡률 κ인 지점의 안전 속도 v = sqrt(a_lat / κ).
앞쪽 거리 s에 있는 커브에 맞추려면 지금 속도는 sqrt(v^2 + 2·a_dec·s) 이하여야 한다
(그 거리 동안 a_dec로 감속해 커브 진입 속도를 맞출 수 있는 상한).
"""
from __future__ import annotations

import math
from typing import Sequence

Point = tuple[float, float]


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def path_curvatures(points: Sequence[Point]) -> list[float]:
    """각 점의 곡률 근사: 앞뒤 선분의 방향 변화 / 평균 선분 길이. 양 끝은 0."""
    n = len(points)
    curv = [0.0] * n
    for i in range(1, n - 1):
        (x0, y0), (x1, y1), (x2, y2) = points[i - 1], points[i], points[i + 1]
        l1 = math.hypot(x1 - x0, y1 - y0)
        l2 = math.hypot(x2 - x1, y2 - y1)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        dh = _wrap(math.atan2(y2 - y1, x2 - x1) - math.atan2(y1 - y0, x1 - x0))
        curv[i] = abs(dh) / ((l1 + l2) / 2)
    return curv


def curve_speed_limit(points: Sequence[Point], lat_accel: float, decel: float, cap: float) -> float:
    """points[0](현재 위치 부근)에서 낼 수 있는 최대 속도."""
    limit = cap
    s = 0.0
    curv = path_curvatures(points)
    for i in range(1, len(points)):
        s += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        if curv[i] < 1e-6:
            continue
        v_curve = math.sqrt(lat_accel / curv[i])
        limit = min(limit, math.sqrt(v_curve * v_curve + 2.0 * decel * s))
    return limit
