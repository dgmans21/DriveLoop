import pandas as pd

from driveloop.eval.driving import crossings, false_brakes, stops, summarize_run

DT = 0.5


def make_log(rows):
    """rows: (state, gt_state, gt_dist, speed) — gt_tl은 dist가 있으면 1."""
    return pd.DataFrame([{"t": i * DT, "state": s, "tl_state": g, "gt_state": g, "gt_dist": d,
                          "gt_tl": 1 if d is not None else None, "speed": v}
                         for i, (s, g, d, v) in enumerate(rows)])


def test_red_crossing_is_violation_yellow_is_not():
    red = make_log([("CRUISE", "RED", 3.0, 8), ("CRUISE", "RED", -0.5, 8)])
    yellow = make_log([("PROCEED_YELLOW", "YELLOW", 1.0, 8), ("PROCEED_YELLOW", "YELLOW", -1.0, 8)])
    assert [c["gt_state"] for c in crossings(red)] == ["RED"]
    s = summarize_run(yellow, DT, stop_margin=2.0)
    assert s["red_violations"] == 0 and s["yellow_crossings"] == 1


def test_crossing_needs_same_light():
    log = make_log([("CRUISE", "RED", 3.0, 8), ("CRUISE", "RED", -0.5, 8)])
    log.loc[1, "gt_tl"] = 2            # 다음 신호등으로 바뀐 것 → 통과가 아님
    assert crossings(log) == []


def test_stop_position_and_start_delay():
    log = make_log([
        ("STOPPING", "RED", 6.0, 3),
        ("STOPPED", "RED", 2.5, 0),
        ("STOPPED", "RED", 2.5, 0),
        ("STOPPED", "GREEN", 2.5, 0),     # 초록불로 바뀜 (t=1.5)
        ("STOPPED", "GREEN", 2.5, 0),
        ("CRUISE", "GREEN", 2.4, 1),      # 출발 (t=2.5) → 지연 1.0초
    ])
    ev = stops(log)
    assert len(ev) == 1 and ev[0]["proper"] and ev[0]["start_delay"] == 1.0
    s = summarize_run(log, DT, stop_margin=2.0)
    assert s["proper_stops"] == 1 and s["false_stops"] == 0
    assert s["stop_err_mean"] == 0.5 and s["start_delay_mean"] == 1.0


def test_departing_on_the_same_tick_as_green_is_zero_delay():
    # 정답값 인지는 초록불이 된 tick에 바로 출발한다 (파일럿에서 None으로 나오던 경우)
    log = make_log([("STOPPED", "RED", 2.0, 0), ("STOPPED", "RED", 2.0, 0), ("CRUISE", "GREEN", 2.0, 0)])
    assert stops(log)[0]["start_delay"] == 0.0


def test_stop_on_green_is_false_stop_and_false_brake():
    log = make_log([("CRUISE", "GREEN", 20.0, 8), ("STOPPING", "GREEN", 15.0, 6), ("STOPPED", "GREEN", 3.0, 0)])
    s = summarize_run(log, DT, stop_margin=2.0)
    assert s["false_stops"] == 1 and s["false_brakes"] == 1


def test_far_stopping_decision_without_slowdown_is_not_false_brake():
    # 31m 앞에서 1~2프레임 STOPPING이 됐지만 목표 속도가 순항 그대로 → 차는 감속하지 않음
    log = make_log([("CRUISE", "GREEN", 32.0, 8), ("STOPPING", "GREEN", 31.0, 8), ("CRUISE", "GREEN", 30.0, 8)])
    log["target_speed"] = [8.33, 8.33, 8.33]
    assert false_brakes(log) == 0
    log["target_speed"] = [8.33, 6.0, 8.33]          # 실제로 감속 지시가 나가면 센다
    assert false_brakes(log) == 1


def test_braking_for_red_is_not_false_brake():
    log = make_log([("CRUISE", "RED", 30.0, 8), ("STOPPING", "RED", 25.0, 7)])
    assert false_brakes(log) == 0


def test_summary_is_json_serializable():
    import json
    from driveloop.eval.driving import nan_to_none
    log = make_log([("CRUISE", "RED", 3.0, 8), ("CRUISE", "RED", -0.5, 8), ("STOPPED", None, None, 0)])
    json.dumps(nan_to_none(summarize_run(log, DT, stop_margin=2.0)))


def test_caution_time_counts_only_when_the_rule_limits_speed():
    # 30m: 제한 sqrt(2·2·28)=10.6 > 순항 8.33 → 안 셈 / 12m: sqrt(2·2·10)=6.3 → 셈
    log = make_log([("CRUISE", "GREEN", 40.0, 8), ("CAUTION", None, 30.0, 8), ("CAUTION", None, 12.0, 7)])
    log["target_speed"] = [8.33, 5.0, 6.3]        # 30m의 5.0은 커브 제한 → CAUTION 효과로 세지 않는다
    assert summarize_run(log, DT, stop_margin=2.0)["caution_s"] == DT


def test_brake_need_is_required_decel_when_stopping_starts():
    # 정지선 12m (정지 지점 10m) 앞, 8 m/s에서 정지 시작 → 8²/(2·10) = 3.2 m/s²
    log = make_log([("CRUISE", "YELLOW", 14.0, 8), ("STOPPING", "YELLOW", 12.0, 8), ("STOPPING", "YELLOW", 8.0, 5)])
    assert summarize_run(log, DT, stop_margin=2.0)["brake_need_max"] == 3.2


def test_lead_metrics_time_gap_ttc_and_gap():
    from driveloop.eval.driving import lead_metrics
    log = pd.DataFrame({"speed": [10.0, 10.0, 8.0, 1.0],
                        "lead_dist": [30.0, 20.0, 12.0, 5.0],
                        "lead_speed": [5.0, 5.0, 8.0, 0.0]})
    m = lead_metrics(log, dt=0.05)
    assert m["min_time_gap"] == 1.5           # 12 / 8 (속도 2 m/s 이하인 마지막 줄은 제외)
    assert m["min_ttc"] == 4.0                # 접근 중인 줄: 30/5=6, 20/5=4, 5/1=5 → 4 (같은 속도인 셋째 줄은 제외)
    assert m["min_lead_gap"] == 5.0


def test_lead_metrics_without_lead_columns_are_none():
    from driveloop.eval.driving import lead_metrics
    m = lead_metrics(pd.DataFrame({"speed": [5.0, 5.0]}), dt=0.05)
    assert m["min_time_gap"] is None and m["min_ttc"] is None and m["hard_brakes"] == 0


def test_crossing_car_is_not_a_following_lead():
    from driveloop.eval.driving import lead_metrics
    n = 60
    log = pd.DataFrame({"speed": [8.0] * n, "lead_dist": [None] * n, "lead_speed": [None] * n,
                        "lead_id": [None] * n})
    log.loc[10:30, ["lead_dist", "lead_speed", "lead_id"]] = [20.0, 6.0, 1]   # 1초 이상 따라감
    log.loc[45:46, ["lead_dist", "lead_speed", "lead_id"]] = [3.0, 0.0, 2]    # 0.1초 가로지름
    m = lead_metrics(log.astype({"lead_dist": float, "lead_speed": float}), dt=0.05)
    assert m["crossing_events"] == 1
    assert m["min_ttc"] == 10.0                   # 20 / (8 - 6): 가로지른 차(3m)는 TTC에서 빠짐


def test_lead_perception_error_miss_and_phantom():
    from driveloop.eval.driving import lead_perception_metrics
    nan = float("nan")
    log = pd.DataFrame({"lead_dist":   [20.0, 20.0, 15.0, nan, nan],
                        "p_lead_dist": [21.0, 19.0, nan, 10.0, nan]})
    m = lead_perception_metrics(log)
    assert m["lead_err_med"] == 1.0
    assert m["lead_miss_rate"] == round(1 / 3, 3)      # 정답 앞차 3 tick 중 1 tick 못 봄
    assert m["lead_phantom_rate"] == round(1 / 3, 3)   # 본 앞차 3 tick 중 1 tick은 없는 차


def test_near_standstill_stop_is_not_hard_brake():
    from driveloop.eval.driving import lead_metrics
    speed = [1.0] * 10 + [0.0] * 10                # 1 m/s에서 한 번에 정지
    assert lead_metrics(pd.DataFrame({"speed": speed}), dt=0.05)["hard_brakes"] == 0


def test_hard_brake_counts_events_not_ticks():
    from driveloop.eval.driving import lead_metrics
    speed = [8.0] * 10 + [8.0 - 0.3 * i for i in range(1, 21)] + [2.0] * 10   # 0.05초마다 0.3 m/s → 6 m/s²
    m = lead_metrics(pd.DataFrame({"speed": speed}), dt=0.05)
    assert m["hard_brakes"] == 1


def test_stop_without_green_has_no_delay():
    log = make_log([("STOPPED", "RED", 2.0, 0), ("STOPPED", "RED", 2.0, 0)])
    assert stops(log)[0]["start_delay"] is None
