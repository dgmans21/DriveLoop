import math

import pytest

from driveloop.config import DrivingConfig
from driveloop.control.lateral import pick_lookahead_point, pure_pursuit_steer, rear_axle
from driveloop.control.pid import PID, LongitudinalController
from driveloop.perception.types import PerceptionOutput, TLState
from driveloop.planning.behavior import BehaviorState, TrafficLightBehavior

MAX_STEER = math.radians(70)


def test_pid_proportional():
    assert PID(1.0, 0.0, 0.0).step(2.0, 0.05) == pytest.approx(2.0)


def test_pid_integral_is_clamped():
    pid = PID(0.0, 1.0, 0.0, integral_limit=1.0)
    for _ in range(1000):
        out = pid.step(10.0, 0.05)
    assert out == pytest.approx(1.0)


def test_longitudinal_holds_brake_when_stopped():
    assert LongitudinalController(DrivingConfig()).step(0.0, 0.1, 0.05) == (0.0, 1.0)


def test_longitudinal_throttle_and_brake_are_exclusive():
    lon = LongitudinalController(DrivingConfig())
    thr, brk = lon.step(8.0, 2.0, 0.05)
    assert thr > 0 and brk == 0
    thr, brk = lon.step(0.0, 8.0, 0.05)
    assert thr == 0 and brk > 0


@pytest.mark.parametrize("yaw, target, sign", [
    (0.0, (10.0, 2.0), +1),    # +y = 오른쪽 → 우회전
    (0.0, (10.0, -2.0), -1),
    (90.0, (-2.0, 10.0), +1),  # yaw 90° = +y 방향, 오른쪽은 -x
    (0.0, (10.0, 0.0), 0),
])
def test_pure_pursuit_steer_direction(yaw, target, sign):
    steer = pure_pursuit_steer((0.0, 0.0), yaw, target, 2.9, MAX_STEER)
    if sign == 0:
        assert steer == pytest.approx(0.0, abs=1e-9)
    else:
        assert math.copysign(1, steer) == sign


def test_rear_axle_is_half_wheelbase_behind():
    assert rear_axle((10.0, 0.0), 0.0, 2.0) == pytest.approx((9.0, 0.0))
    assert rear_axle((0.0, 0.0), 90.0, 2.0) == pytest.approx((0.0, -1.0))


def test_pick_lookahead_point():
    pts = [(float(i), 0.0) for i in range(10)]
    assert pick_lookahead_point(pts, (0.0, 0.0), 4.5) == (5.0, 0.0)
    assert pick_lookahead_point(pts, (0.0, 0.0), 100.0) == (9.0, 0.0)


def test_closed_loop_red_light_stop_point_mass():
    """판단+PID를 단순 질점 모델로 돌려, 빨간불 정지선 앞에 넘지 않고 서는지 확인."""
    cfg = DrivingConfig()
    behavior = TrafficLightBehavior(cfg)
    lon = LongitudinalController(cfg)
    dt, line, x, v = 0.05, 60.0, 0.0, 30 / 3.6
    max_decel_seen = 0.0
    for _ in range(int(40 / dt)):
        dist = line - x
        d = behavior.step(PerceptionOutput(TLState.RED, dist) if dist < cfg.tl_lookahead else PerceptionOutput(), v)
        thr, brk = lon.step(d.target_speed, v, dt)
        a = 4.0 * thr - 8.0 * brk - 0.05 * v  # 대략적인 가감속 특성
        v_new = max(0.0, v + a * dt)
        if v_new < v:
            max_decel_seen = max(max_decel_seen, (v - v_new) / dt)
        x += (v + v_new) / 2 * dt
        v = v_new
    final_gap = line - x
    assert behavior.state is BehaviorState.STOPPED
    assert 0.0 < final_gap <= cfg.stop_margin + 1.0, final_gap
    assert v == 0.0
