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
    out["hard_brakes"] = hard_brakes(log, dt, hard_decel, hard_min_speed)[0]
    return out


def hard_brakes(log: pd.DataFrame, dt: float, hard_decel: float = 4.0, hard_min_speed: float = 3.0) -> tuple[int, float]:
    """(급제동 횟수, 최대 감속도 m/s²).

    실제 감속도: 0.25초 이동 평균 속도의 변화율. 거의 선 상태(< hard_min_speed)에서 0으로 떨어지는 순간은
    감속도가 크게 계산되지만 체감 급제동이 아니다 (acc_pilot: 출발 직후 1 m/s → 0) → 제외
    """
    k = max(1, round(0.25 / dt))
    v = log.speed.rolling(k, min_periods=1).mean()
    decel = (-(v.diff() / dt)).where(v.shift() > hard_min_speed)
    hard = decel > hard_decel
    peak = float(decel.max()) if decel.notna().any() else 0.0
    return int((hard & ~hard.shift(fill_value=False)).sum()), round(max(peak, 0.0), 2)


def ped_metrics(log: pd.DataFrame, dt: float, body_half: float = 1.3, corridor_half: float = 1.5,
                cruise_speed: float = 30 / 3.6, after_s: float = 3.0) -> dict:
    """보행자 시나리오 지표 (C-2). 로그 열: ped_gap(경로를 따라 잰 범퍼~보행자), ped_lat(경로 중심선에서 옆 거리),
    yield_id(양보 중인 보행자, 없으면 비어 있음).

    - min_ped_gap: 보행자가 내 차체 폭(body_half) 안에 있는 동안 가장 가까웠던 범퍼 간격 (0 이하면 부딪힘)
    - yield_s: 양보(감속 상한) 중이던 시간 — 양보가 필요 없는 시나리오(보도·연석)에선 곧 헛감속 시간
    - min_speed_after_cruise: 처음 순항 속도에 닿은 뒤 최저 속도 (헛감속의 크기)
    - stop_s: 출발 후 선 시간, restart_delay: 내가 서 있는 동안 보행자가 통로를 벗어난 순간 → 1 m/s 넘을 때까지
    """
    # 평가 구간: 보행자를 지나간 뒤 after_s초까지 — 그 뒤 커브 감속 등은 보행자 판단과 무관 (ped_scen_v1 시드 3)
    passed = log.index[log.ped_gap.notna() & (log.ped_gap <= -2.0)]
    if len(passed):
        log = log.loc[: passed[0] + round(after_s / dt)]
    lat = log.ped_lat.abs()
    ahead = log.ped_gap.notna() & (log.ped_gap > -2.0)
    inside = ahead & (lat <= body_half)
    hb, peak = hard_brakes(log, dt)
    cruise_idx = log.index[log.speed >= 0.9 * cruise_speed]   # PID 정상 상태 오차로 순항이 목표보다 ~0.6 m/s 낮다
    moved = log.index[log.speed > 3.0]                         # 스폰 직후 차가 떨어지며 생기는 속도는 출발이 아님
    out = {
        "min_ped_gap": round(float(log.ped_gap[inside].min()), 2) if inside.any() else None,
        "yield_s": round(float(log.yield_id.notna().sum() * dt), 2),
        "caution_ped_s": round(float(log.ped_caution.notna().sum() * dt), 2) if "ped_caution" in log else None,
        "min_speed_after_cruise": round(float(log.speed[cruise_idx[0]:].min()), 2) if len(cruise_idx) else None,
        "stop_s": round(float((log.speed[moved[0]:] < 0.1).sum() * dt), 2) if len(moved) else 0.0,
        "restart_delay": None,
        "hard_brakes": hb,
        "max_decel": peak,
    }
    in_corr = ahead & (lat <= corridor_half)
    left = log.index[(in_corr.shift(fill_value=False)) & ~in_corr]
    for i in left:
        if log.speed[i] < 0.5:
            go = log.index[(log.index > i) & (log.speed > 1.0)]
            if len(go):
                out["restart_delay"] = round(float(log.t[go[0]] - log.t[i]), 2)
            break
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



def ped_perception_metrics(log: pd.DataFrame, dt: float, near: float = 30.0, corridor_half: float = 1.5,
                           margin: float = 1.0) -> dict:
    """모델이 본 보행자(p_ped_*, 실제 보행자와 2m 안에서 짝지은 것) vs 정답값(ped_*) — C-7.

    - 관련 구간: 실제 보행자가 near m 앞 안이고 통로 근처(통로 반폭 + margin)에 있을 때 = 양보 판단에 쓰일 수 있는 때
    - 놓침률: 관련 구간에서 모델이 그 보행자를 못 본(짝 없음) 비율
    - 첫 확정 거리: 모델이 처음 그 보행자를 확정한 순간의 실제 범퍼 간격 (클수록 일찍 봄)
    - 오차: 둘 다 있을 때 거리·옆 위치·다가오는 속도의 |추정 − 정답| 중앙값
    - 헛양보: 실제 보행자가 통로에서 멀리(반폭 + 2m 밖) 있는데 양보 중이던 시간
    """
    out = {"ped_miss_rate": None, "first_seen_gap": None, "ped_gap_err_med": None, "ped_lat_err_med": None,
           "ped_toward_err_med": None, "phantom_yield_s": None}
    if "p_ped_gap" not in log:
        return out
    gt_ok = log.ped_gap.notna() & (log.ped_gap > 0)
    rel = gt_ok & (log.ped_gap <= near) & (log.ped_lat.abs() <= corridor_half + margin)
    if rel.any():
        out["ped_miss_rate"] = round(float((rel & log.p_ped_gap.isna()).sum() / rel.sum()), 3)
    seen = gt_ok & log.p_ped_gap.notna()
    if seen.any():
        out["first_seen_gap"] = round(float(log.ped_gap[seen].iloc[0]), 1)
        both = log[seen]
        out["ped_gap_err_med"] = round(float((both.p_ped_gap - both.ped_gap).abs().median()), 2)
        out["ped_lat_err_med"] = round(float((both.p_ped_lat - both.ped_lat).abs().median()), 2)
        out["ped_toward_err_med"] = round(float((both.p_ped_toward - both.ped_toward).abs().median()), 2)
    far = gt_ok & (log.ped_lat.abs() > corridor_half + 2.0)
    out["phantom_yield_s"] = round(float((far & log.yield_id.notna()).sum() * dt), 2)
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
