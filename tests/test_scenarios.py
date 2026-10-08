import math

import pytest

from driveloop.eval.scenarios import SCENARIOS, blend_paths, heading_change


def test_follow_keeps_constant_speed_in_my_lane():
    c = SCENARIOS["follow"].command(20.0)
    assert c.target_speed == pytest.approx(20 / 3.6) and not c.full_brake and c.lane_blend == 1.0


def test_brake_switches_to_full_brake_at_15s():
    s = SCENARIOS["brake"]
    assert not s.command(14.9).full_brake
    assert s.command(15.0).full_brake


def test_stopped_car_never_moves():
    assert SCENARIOS["stopped"].command(0.0).full_brake


def test_cutin_moves_from_adjacent_lane_smoothly():
    s = SCENARIOS["cutin"]
    assert s.adjacent
    assert s.command(5.0).lane_blend == 0.0
    mid = s.command(9.5).lane_blend
    assert 0.0 < mid < 1.0
    assert s.command(11.0).lane_blend == pytest.approx(1.0)


def test_heading_change_straight_vs_turn():
    straight = [(float(x), 0.0) for x in range(0, 130, 2)]
    turn = [(float(x), 0.0) for x in range(0, 40, 2)] + \
           [(40 + 20 * math.sin(a / 10), -20 + 20 * math.cos(a / 10)) for a in range(1, 16)]
    assert heading_change(straight, 120) < 1.0
    assert heading_change(turn, 120) > 60.0


def test_blend_paths():
    a, b = [(0.0, 0.0), (10.0, 0.0)], [(0.0, 3.5), (10.0, 3.5)]
    assert blend_paths(a, b, 0.5) == [(0.0, 1.75), (10.0, 1.75)]


def test_ped_waits_until_triggered_then_crosses():
    from driveloop.eval.scenarios import PED_SCENARIOS
    s = PED_SCENARIOS["cross"]
    assert s.command(None, 3.3, 3.3).u_speed == 0.0
    assert s.command(0.5, 3.0, 3.3).u_speed == pytest.approx(-1.4)
    assert s.command(9.0, -5.1, 3.3).u_speed == 0.0              # 다 건너면 멈춤


def test_ped_stop_scenario_waits_in_lane_center_then_continues():
    from driveloop.eval.scenarios import PED_SCENARIOS
    s = PED_SCENARIOS["stop"]
    assert s.command(1.0, 1.9, 2.8).u_speed < 0
    assert s.command(5.0, 0.0, 2.8).u_speed == 0.0               # 2.8/1.4 = 2초 뒤부터 8초 멈춤
    assert s.command(10.5, 0.0, 2.8).u_speed < 0


def test_ped_start_position_is_relative_to_sidewalk_or_lane_edge():
    from driveloop.eval.scenarios import PED_SCENARIOS
    assert PED_SCENARIOS["cross"].start_u(3.5, 2.88) == pytest.approx(3.88)     # 보도 경계 + 1m
    assert PED_SCENARIOS["curb"].start_u(3.5, 2.88) == pytest.approx(2.25)      # 차선 끝 + 0.5m


def test_sidewalk_and_curb_never_enter_the_road():
    from driveloop.eval.scenarios import PED_SCENARIOS
    for name in ("sidewalk", "curb"):
        for since in (None, 0.0, 10.0):
            assert PED_SCENARIOS[name].command(since, 3.0, 3.0).u_speed == 0.0
