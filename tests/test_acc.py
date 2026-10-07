import pytest

from driveloop.planning.acc import acc_target

CRUISE = 30 / 3.6


def test_no_lead_means_no_limit():
    out = acc_target(None, None, CRUISE)
    assert out.target_speed is None and not out.emergency


def test_far_and_faster_lead_does_not_limit_cruise():
    out = acc_target(45.0, 12.0, CRUISE)
    assert out.target_speed > CRUISE


def test_slows_down_for_stopped_car_ahead():
    out = acc_target(30.0, 0.0, CRUISE)
    assert 0 < out.target_speed < CRUISE
    assert not out.emergency                    # 30m면 편안하게 설 수 있음


def test_stops_behind_stopped_car_at_standstill_gap():
    assert acc_target(5.0, 0.0, 0.5).target_speed == pytest.approx(0.0)
    assert acc_target(4.0, 0.0, 0.0).target_speed == 0.0      # 더 가까우면 0 (후진하지 않음)


def test_holds_speed_when_following_at_desired_gap():
    v = 8.0
    out = acc_target(5.0 + 1.8 * v, v, v)
    assert out.target_speed == pytest.approx(v)


def test_too_close_to_same_speed_lead_slows_down():
    v = 8.0
    assert acc_target(10.0, v, v).target_speed < v


def test_closing_fast_on_stopped_car_is_emergency():
    # 10 m/s로 달리다 10m 앞 정지 차: 5m 안에 서려면 10 m/s² 필요
    assert acc_target(10.0, 0.0, 10.0).emergency


def test_oncoming_car_is_treated_as_stopped():
    assert acc_target(20.0, -5.0, CRUISE).target_speed == pytest.approx(acc_target(20.0, 0.0, CRUISE).target_speed)


def test_safe_speed_lets_me_stop_if_lead_brakes():
    # 같은 감속도로 둘 다 제동할 때 내 정지 거리 ≤ 앞차 정지 거리 + (간격 − d0)
    gap, vl = 20.0, 6.0
    v = acc_target(gap, vl, 20.0, time_gap=0.0, tau=1e-6).target_speed   # 간격 유지 항을 끄고 안전 속도만
    a = 2.0
    assert v * v / (2 * a) <= vl * vl / (2 * a) + (gap - 5.0) + 1e-6
