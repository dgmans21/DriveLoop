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
- 주의 감속 시간(caution_s): CAUTION(색을 몰라 속도 제한)의 제한 속도 sqrt(2·comfort_decel·gap)가
  순항 속도보다 낮았던 시간 (= 안전장치가 실제로 속도를 깎은 시간. 커브 감속은 제외)
앞차 지표 (B단계, 로그에 정답값 lead_dist / lead_speed 열이 있을 때만):
- 따라가는 앞차만 (lead_id가 1초 이상 같은 차). 잠깐 경로를 가로지르는 차는 crossing_events로 따로 센다
- 최소 차간 시간(min_time_gap): 앞차 간격 / 내 속도 (내 속도 > 2 m/s일 때). 사람 운전 권장 ~2초, 1초 미만은 위험
- 최소 TTC(min_ttc): 앞차 간격 / 접근 속도 (접근 중일 때만). 충돌까지 남은 시간, 2초 미만이면 위험 신호
- 최소 정지 간격(min_lead_gap): 앞차와 가장 가까웠던 범퍼 간격
- 급제동(hard_brakes): 실제 감속도(속도 변화, 0.25초 평활)가 4 m/s²를 넘은 횟수 (신호·앞차 원인 구분 없이)
- 제동 강도(brake_need_max): STOPPING에 들어간 순간 정지 지점(정지선 − stop_margin)까지 서는 데 필요한 감속도
  v²/(2·gap)의 최댓값. 늦게 알아챌수록 커진다 (판단의 max_stop_decel = 4 m/s²가 '설 수 있음'의 한계)
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
    """정답 신호가 초록(또는 없음)인데 STOPPING에 들어가 **실제로 감속을 지시한** 횟수.

    멀리서(정지 프로파일 속도 > 순항 속도) 잠깐 STOPPING이 됐다 풀리면 차는 감속하지 않으므로 세지 않는다
    (compare_v2에서 31m 앞 1~2프레임 오인이 '판단만' 바뀌고 속도는 그대로였던 경우).
    """
    s = log.state.reset_index(drop=True)
    gt = log.gt_state.reset_index(drop=True)
    target = log.target_speed.reset_index(drop=True) if "target_speed" in log else None
    cruise = target.max() if target is not None else None
    n = 0
    for i in range(len(s)):
        if s[i] != "STOPPING" or (i > 0 and s[i - 1] == "STOPPING") or gt[i] in STOP_STATES:
            continue
        if target is None:
            n += 1
            continue
        j = i
        while j < len(s) and s[j] == "STOPPING":
            if target[j] < cruise - 0.05:
                n += 1
                break
            j += 1
    return n


def brake_needs(log: pd.DataFrame, stop_margin: float) -> list[float]:
    s = log.state
    entering = (s == "STOPPING") & (s.shift() != "STOPPING") & log.gt_dist.notna()
    out = []
    for v, d in zip(log.speed[entering], log.gt_dist[entering]):
        gap = d - stop_margin
        out.append(round(v * v / (2 * gap), 2) if gap > 0.05 else float("inf"))
    return out


def lead_metrics(log: pd.DataFrame, dt: float, hard_decel: float = 4.0, min_follow_s: float = 1.0,
                 hard_min_speed: float = 3.0) -> dict:
    """앞차 관련 지표. 앞차 열이 없거나 앞차가 한 번도 없으면 None."""
    out = {"min_time_gap": None, "min_ttc": None, "min_lead_gap": None, "crossing_events": 0}
    if "lead_dist" in log and log.lead_dist.notna().any():
        has = log.lead_dist.notna()
        if "lead_id" in log:
            # 같은 차가 min_follow_s 이상 계속 앞에 있어야 '따라가는 앞차' — 교차로를 가로지르는 차는
            # 잠깐(0.05~0.3초) 경로 위에 나타났다 사라져 TTC가 비정상적으로 작게 나온다 (acc_pilot에서 확인)
            run_id = (log.lead_id != log.lead_id.shift()).cumsum()
            run_len = log.groupby(run_id).lead_id.transform("size")
            follow = has & (run_len >= round(min_follow_s / dt))
            out["crossing_events"] = int((has & ~follow & (run_id != run_id.shift())).sum())
            has = follow
        moving = has & (log.speed > 2.0)
        if moving.any():
            out["min_time_gap"] = round(float((log.lead_dist[moving] / log.speed[moving]).min()), 2)
        closing = has & ((log.speed - log.lead_speed) > 0.5)
        if closing.any():
            out["min_ttc"] = round(float((log.lead_dist[closing] / (log.speed - log.lead_speed)[closing]).min()), 2)
        if has.any():
            out["min_lead_gap"] = round(float(log.lead_dist[has].min()), 2)
    # 실제 감속도: 0.25초 이동 평균 속도의 변화율. 거의 선 상태(< hard_min_speed)에서 0으로 떨어지는 순간은
    # 감속도가 크게 계산되지만 체감 급제동이 아니다 (acc_pilot: 출발 직후 1 m/s → 0) → 제외
    k = max(1, round(0.25 / dt))
    v = log.speed.rolling(k, min_periods=1).mean()
    hard = ((-(v.diff() / dt)) > hard_decel) & (v.shift() > hard_min_speed)
    out["hard_brakes"] = int((hard & ~hard.shift(fill_value=False)).sum())
    return out


def lead_perception_metrics(log: pd.DataFrame, near: float = 30.0) -> dict:
    """판단에 쓴 앞차(p_lead_*) vs 정답값 앞차(lead_*) — 모델 인지 실행에서만 의미가 있다.

    - 거리 오차: 둘 다 앞차를 본 tick의 |추정 − 정답| 중앙값·90%
    - 놓침: 정답값 앞차가 near m 안에 있는데 인지가 앞차 없음 (위험한 쪽)
    - 헛봄: 인지는 near m 안에 앞차가 있다는데 정답값은 없음 (불필요한 감속 쪽)
    """
    out = {"lead_err_med": None, "lead_err_p90": None, "lead_miss_rate": None, "lead_phantom_rate": None}
    if "p_lead_dist" not in log or "lead_dist" not in log:
        return out
    both = log.lead_dist.notna() & log.p_lead_dist.notna()
    if both.any():
        err = (log.p_lead_dist[both] - log.lead_dist[both]).abs()
        out["lead_err_med"] = round(float(err.median()), 2)
        out["lead_err_p90"] = round(float(err.quantile(0.9)), 2)
    gt_near = log.lead_dist.notna() & (log.lead_dist <= near)
    if gt_near.any():
        out["lead_miss_rate"] = round(float((gt_near & log.p_lead_dist.isna()).sum() / gt_near.sum()), 3)
    p_near = log.p_lead_dist.notna() & (log.p_lead_dist <= near)
    if p_near.any():
        out["lead_phantom_rate"] = round(float((p_near & log.lead_dist.isna()).sum() / p_near.sum()), 3)
    return out


def summarize_run(log: pd.DataFrame, dt: float, stop_margin: float, comfort_decel: float = 2.0,
                  cruise_speed: float = 30 / 3.6) -> dict:
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
        "brake_need_max": (lambda n: min(max(n), 99.0) if n else None)(brake_needs(log, stop_margin)),
        # 기록된 target_speed는 커브 제한까지 적용된 값이라 CAUTION 효과와 섞인다 → CAUTION 규칙 자체의 제한 속도로 판정
        "caution_s": round(float(((log.state == "CAUTION") &
                                  ((2 * comfort_decel * (log.gt_dist - stop_margin).clip(lower=0)) ** 0.5
                                   < cruise_speed - 0.05)).sum() * dt), 2),
        "stop_err_mean": round(sum(abs(p - stop_margin) for p in pos) / len(pos), 2) if pos else None,
        "stop_over_line": sum(p < 0 for p in pos),                 # 정지선을 넘어서 섰음
        "start_delay_mean": round(sum(delays) / len(delays), 2) if delays else None,
        "decision_agree": round(float((seen.tl_state == seen.gt_state).mean()), 3) if len(seen) else None,
        **lead_metrics(log, dt),
        **lead_perception_metrics(log),
    }


def nan_to_none(d: dict) -> dict:
    """JSON 저장용: numpy 숫자 → 파이썬 숫자, NaN → None."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):          # numpy int64 / float64 / bool_
            v = v.item()
        out[k] = None if isinstance(v, float) and math.isnan(v) else v
    return out
