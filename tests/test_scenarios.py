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
