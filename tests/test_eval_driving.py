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


def test_braking_for_red_is_not_false_brake():
    log = make_log([("CRUISE", "RED", 30.0, 8), ("STOPPING", "RED", 25.0, 7)])
    assert false_brakes(log) == 0


def test_summary_is_json_serializable():
    import json
    from driveloop.eval.driving import nan_to_none
    log = make_log([("CRUISE", "RED", 3.0, 8), ("CRUISE", "RED", -0.5, 8), ("STOPPED", None, None, 0)])
    json.dumps(nan_to_none(summarize_run(log, DT, stop_margin=2.0)))


def test_stop_without_green_has_no_delay():
    log = make_log([("STOPPED", "RED", 2.0, 0), ("STOPPED", "RED", 2.0, 0)])
    assert stops(log)[0]["start_delay"] is None
