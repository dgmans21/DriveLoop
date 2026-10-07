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


def ground_distance(y_bottom: float, fy: float, cy: float, cam_height: float) -> float | None:
    """박스 아래 변이 지평선 아래 있어야 계산 가능 (위면 도로 위에 닿은 점이 아님)."""
    dy = y_bottom - cy
    if dy <= 1.0:
        return None
    return fy * cam_height / dy


def width_distance(box_w: float, fx: float, real_width: float = 1.8) -> float | None:
    if box_w <= 1.0:
        return None
    return fx * real_width / box_w
