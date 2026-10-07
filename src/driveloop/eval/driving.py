"""주행 로그 → 주행 지표 (CARLA 비의존). 3단계 '정답값 인지 vs 모델 인지' 비교에 쓴다.

로그 한 줄 = 시뮬레이션 1 tick. 필요한 열:
  t, speed(m/s), state(CRUISE/STOPPING/STOPPED/PROCEED_YELLOW), tl_state(판단에 쓴 신호), gt_state, gt_dist, gt_tl
gt_* 는 정답값 기준 '내 신호등'의 상태 / 앞 범퍼~정지선 거리(m, 음수 = 넘음) / actor id. 신호가 없으면 비어 있다.

지표 정의
- 정지선 통과(crossing): 같은 신호등에서 gt_dist가 양수 → 0 이하로 바뀐 tick. 그때 gt_state가 RED면 **신호 위반**
  (노란불 통과는 딜레마 규칙상 허용 → 따로 센다)
- 정지(stop): state가 STOPPED로 바뀐 tick. 그때 정답 신호가 RED/YELLOW면 정상 정지, 아니면 **불필요한 정지**
  정지 위치 = gt_dist (판단은 정지선 stop_margin m 앞을 목표로 한다)
- 출발 지연: 정지 중 정답 신호가 GREEN으로 바뀐 순간 → STOPPED를 벗어난 순간
- 불필요한 감속: 정답 신호가 GREEN(또는 없음)인데 STOPPING에 들어간 횟수
"""
from __future__ import annotations

import math

import pandas as pd

STOP_STATES = {"RED", "YELLOW"}


def _same_light(a, b) -> bool:
    return pd.notna(a) and pd.notna(b) and a == b


def crossings(log: pd.DataFrame) -> list[dict]:
    out = []
    prev = None
    for r in log.itertuples(index=False):
        if prev is not None and _same_light(prev.gt_tl, r.gt_tl) and pd.notna(prev.gt_dist) and pd.notna(r.gt_dist) \
                and prev.gt_dist > 0 >= r.gt_dist:
            out.append({"t": r.t, "gt_state": r.gt_state, "speed": r.speed, "tl": r.gt_tl})
        prev = r
    return out


def stops(log: pd.DataFrame) -> list[dict]:
    """정지 이벤트 + 출발 지연."""
    rows = log.reset_index(drop=True)
    out = []
    for i in range(len(rows)):
        if rows.state[i] != "STOPPED" or (i > 0 and rows.state[i - 1] == "STOPPED"):
            continue
        ev = {"t": rows.t[i], "gt_state": rows.gt_state[i], "gt_dist": rows.gt_dist[i], "tl": rows.gt_tl[i],
              "proper": rows.gt_state[i] in STOP_STATES, "start_delay": None}
        j = i
        while j < len(rows) and rows.state[j] == "STOPPED" and rows.gt_state[j] != "GREEN":
            j += 1
        if j < len(rows) and rows.gt_state[j] == "GREEN":          # 정지 중 초록불로 바뀜 (같은 tick 출발 = 지연 0)
            k = j
            while k < len(rows) and rows.state[k] == "STOPPED":
                k += 1
            if k < len(rows):                                      # 실행 시간 안에 출발
                ev["start_delay"] = round(float(rows.t[k] - rows.t[j]), 2)
        out.append(ev)
    return out


def false_brakes(log: pd.DataFrame) -> int:
    s = log.state
    entering = (s == "STOPPING") & (s.shift() != "STOPPING")
    return int((entering & ~log.gt_state.isin(STOP_STATES)).sum())


def summarize_run(log: pd.DataFrame, dt: float, stop_margin: float) -> dict:
    cr = crossings(log)
    st = stops(log)
    proper = [e for e in st if e["proper"]]
    delays = [e["start_delay"] for e in proper if e["start_delay"] is not None]
    pos = [e["gt_dist"] for e in proper if pd.notna(e["gt_dist"])]
    seen = log[log.gt_state.notna()]
    return {
        "seconds": round(float(log.t.iloc[-1]), 1) if len(log) else 0.0,
        "distance_m": round(float(log.speed.sum() * dt), 1),
        "crossings": len(cr),
        "red_violations": sum(e["gt_state"] == "RED" for e in cr),
        "yellow_crossings": sum(e["gt_state"] == "YELLOW" for e in cr),
        "stops": len(st),
        "proper_stops": len(proper),
        "false_stops": len(st) - len(proper),
        "false_brakes": false_brakes(log),
        "stop_err_mean": round(sum(abs(p - stop_margin) for p in pos) / len(pos), 2) if pos else None,
        "stop_over_line": sum(p < 0 for p in pos),                 # 정지선을 넘어서 섰음
        "start_delay_mean": round(sum(delays) / len(delays), 2) if delays else None,
        "decision_agree": round(float((seen.tl_state == seen.gt_state).mean()), 3) if len(seen) else None,
    }


def nan_to_none(d: dict) -> dict:
    """JSON 저장용: numpy 숫자 → 파이썬 숫자, NaN → None."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):          # numpy int64 / float64 / bool_
            v = v.item()
        out[k] = None if isinstance(v, float) and math.isnan(v) else v
    return out
