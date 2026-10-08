"""카메라 한 대로 앞차까지 거리 추정 (CARLA 비의존, 단위 테스트 대상).

핀홀 카메라, 카메라 광축이 도로와 평행(pitch 0)하다고 가정한다.
  1) 바닥선 방법: 차가 도로에 닿는 점(박스 아래 변 y2)은 멀수록 지평선(cy)에 가까워진다
       Z = f · H / (y2 − cy)       H = 카메라 높이
     → 도로가 평평하면 정확, 오르막·내리막·차체 흔들림(pitch)에 약하다. 박스 아래가 화면 밖이면 못 쓴다
  2) 폭 방법: 승용차 폭은 대략 일정 (약 1.8m)
       Z = f · W_real / box_w
     → 도로 기울기와 무관, 차종(트럭·이륜차) 폭 차이와 비스듬히 보이는 차(옆면이 보여 박스가 넓어짐)에 약하다
Z는 카메라에서 차 뒷면까지의 앞쪽 거리. 범퍼 간격 = Z − (내 앞 범퍼가 카메라보다 앞선 거리).
"""
from __future__ import annotations

import math


def ground_distance(y_bottom: float, fy: float, cy: float, cam_height: float) -> float | None:
    """박스 아래 변이 지평선 아래 있어야 계산 가능 (위면 도로 위에 닿은 점이 아님)."""
    dy = y_bottom - cy
    if dy <= 1.0:
        return None
    return fy * cam_height / dy


def ray_road_hit(origin: tuple[float, float, float], direction: tuple[float, float, float],
                 road: list[tuple[float, float, float]], max_dist: float = 80.0, step: float = 0.5,
                 ground_offset: float = 0.0) -> tuple[float, float, float] | None:
    """카메라 광선이 '지도의 도로 면'과 처음 만나는 점 (경사 보정, B-7b).

    평면 가정(ground_distance) 대신 내 경로 waypoint들의 높이(z)를 도로 면으로 쓴다:
    광선을 앞으로 조금씩 나아가며, 그 지점에서 가장 가까운 경로 점의 높이(+ground_offset)보다
    광선이 처음 낮아지는 곳을 찾고 직전 점과 선형 보간한다.
    ground_offset: 박스 아래 변이 실제 접지점보다 살짝 위에 잡히는 만큼 (평면 보정 H=1.556의 1.7과 차이)
    도로를 못 만나면(광선이 지평선 위 / max_dist 밖) None.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    horiz = math.hypot(dx, dy)
    if horiz < 1e-6 or not road:
        return None
    ux, uy, uz = dx / horiz, dy / horiz, dz / horiz          # 수평 1m당 이동량
    prev = None
    d = step
    while d <= max_dist:
        px, py, pz = ox + ux * d, oy + uy * d, oz + uz * d
        gz = min(road, key=lambda w: (w[0] - px) ** 2 + (w[1] - py) ** 2)[2] + ground_offset
        above = pz - gz
        if above <= 0:
            if prev is None:
                return (px, py, gz)
            d0, a0 = prev
            f = a0 / (a0 - above)                             # 위(+)에서 아래(−)로 바뀌는 지점 보간
            dd = d0 + f * (d - d0)
            return (ox + ux * dd, oy + uy * dd, oz + uz * dd)
        prev = (d, above)
        d += step
    return None


def width_distance(box_w: float, fx: float, real_width: float = 1.8) -> float | None:
    if box_w <= 1.0:
        return None
    return fx * real_width / box_w
