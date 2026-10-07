"""앞차 따라가기 (ACC) — 속도 상한만 계산한다 (CARLA 비의존, 단위 테스트 대상).

신호등 판단(behavior.py)과 독립: 최종 목표 속도 = min(신호 판단, 커브 제한, ACC 상한).
→ 빨간불 정지와 앞차 정지가 겹쳐도 더 보수적인 쪽을 따른다.

두 가지 상한 중 낮은 값:
  1) 안전 속도  v_safe = sqrt(v_lead² + 2·a·(gap − d0))
     앞차가 지금 a로 제동해도 내가 같은 a로 앞차 뒤 d0에 멈출 수 있는 최대 속도 (Gipps 모델의 핵심 아이디어)
  2) 간격 유지  v_gap  = v_lead + (gap − (d0 + T·v)) / tau
     원하는 간격(정지 간격 d0 + 시간 간격 T)보다 가까우면 줄이고 멀면 천천히 따라붙는다
앞차가 마주 오거나 후진 중(속도 < 0)이면 정지한 장애물로 본다.
비상(emergency): 지금 속도에서 앞차 뒤 d0에 서려면 max_decel보다 세게 밟아야 함 → 로그·지표용 표시.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AccOutput:
    target_speed: float | None      # None = 앞차 없음 (제약 없음)
    desired_gap: float | None
    emergency: bool = False


def acc_target(lead_distance: float | None, lead_speed: float | None, speed: float, *,
               time_gap: float = 1.8, standstill_gap: float = 5.0, tau: float = 1.5,
               comfort_decel: float = 2.0, max_decel: float = 4.0) -> AccOutput:
    if lead_distance is None:
        return AccOutput(None, None)
    vl = max(lead_speed or 0.0, 0.0)
    room = lead_distance - standstill_gap
    desired = standstill_gap + time_gap * speed
    v_safe = math.sqrt(max(0.0, vl * vl + 2.0 * comfort_decel * room))
    v_gap = vl + (lead_distance - desired) / tau
    target = max(0.0, min(v_safe, v_gap))
    # 앞차가 지금 서 있다고 보고(가장 나쁜 경우의 근사) 앞차 뒤 d0까지 서는 데 필요한 감속도
    closing = speed - vl
    need = (closing * closing) / (2.0 * room) if room > 0.05 and closing > 0 else (math.inf if closing > 0.3 else 0.0)
    return AccOutput(target, desired, need > max_decel)
