import math

import pytest

from driveloop.perception.lead import Obstacle, find_lead

STRAIGHT = [(float(x), 0.0) for x in range(0, 62, 2)]          # x축을 따라 60m
# 20m 직진 후 반경 20m로 왼쪽(y 음수 방향)으로 90도 도는 경로
CURVE = [(float(x), 0.0) for x in range(0, 21, 2)] + \
        [(20 + 20 * math.sin(a / 10), -20 + 20 * math.cos(a / 10)) for a in range(1, 16)]


def test_car_ahead_in_my_lane_is_lead_with_bumper_gap():
    lead = find_lead(STRAIGHT, (0.0, 0.0), 2.4, [Obstacle(7, 20.0, 0.3, vx=5.0, half_length=2.4)])
    assert lead.id == 7
    assert lead.distance == pytest.approx(20 - 2.4 - 2.4)
    assert lead.speed == pytest.approx(5.0)


def test_adjacent_lane_and_behind_are_ignored():
    obs = [Obstacle(1, 15.0, 3.5),            # 옆 차선
           Obstacle(2, -8.0, 0.0),            # 뒤
           Obstacle(3, 30.0, -0.4)]           # 내 차선 앞
    assert find_lead(STRAIGHT, (0.0, 0.0), 2.4, obs).id == 3


def test_nearest_of_several_cars_is_lead():
    obs = [Obstacle(1, 40.0, 0.0), Obstacle(2, 18.0, 0.0), Obstacle(3, 30.0, 0.0)]
    assert find_lead(STRAIGHT, (0.0, 0.0), 2.4, obs).id == 2


def test_oncoming_car_has_negative_speed():
    lead = find_lead(STRAIGHT, (0.0, 0.0), 2.4, [Obstacle(1, 25.0, 0.0, vx=-6.0)])
    assert lead.speed == pytest.approx(-6.0)


def test_far_car_beyond_max_distance_is_ignored():
    assert find_lead(STRAIGHT, (0.0, 0.0), 2.4, [Obstacle(1, 58.0, 0.0)], max_distance=40.0) is None


def test_car_around_the_curve_uses_path_distance_not_straight_line():
    # 커브 너머 경로 위의 차: 직선거리는 짧지만 경로를 따라가면 더 멀다
    a = 1.2
    car = (20 + 20 * math.sin(a), -20 + 20 * math.cos(a))
    lead = find_lead(CURVE, (0.0, 0.0), 2.4, [Obstacle(9, *car)])
    path_dist = 20 + 20 * a - 2.4 - 2.4
    assert lead is not None and lead.distance == pytest.approx(path_dist, abs=0.6)
    assert lead.distance > math.hypot(*car) - 4.8


def test_tracker_needs_two_frames_to_confirm():
    from driveloop.perception.lead import LeadTracker
    t = LeadTracker(dt=0.05)
    assert t.update(20.0, 8.0) is None                  # 1프레임 오검출로는 앞차가 생기지 않는다
    assert t.update(19.8, 8.0) is not None


def test_tracker_converges_to_lead_speed_from_distance_change():
    from driveloop.perception.lead import LeadTracker
    t = LeadTracker(dt=0.05)
    lead = None
    for k in range(80):                                 # 0.05초에 0.1m씩 가까워짐 → 상대 −2 m/s → 앞차 6 m/s
        lead = t.update(30.0 - 0.1 * k, 8.0)
    assert lead.speed == pytest.approx(6.0, abs=0.2)


def test_new_lead_starts_as_stopped_conservatively():
    from driveloop.perception.lead import LeadTracker
    t = LeadTracker(dt=0.05)
    t.update(20.0, 8.0)
    lead = t.update(19.6, 8.0)                          # 정말 서 있는 차라면 0.05초에 0.4m 가까워짐
    assert lead.speed < 1.0                             # 처음엔 '서 있다'에 가깝게 → 늦게 제동하지 않음


def test_tracker_holds_through_short_misses_then_drops():
    from driveloop.perception.lead import LeadTracker
    t = LeadTracker(dt=0.05, max_missed=3)
    t.update(20.0, 8.0)
    t.update(20.0, 8.0)
    for _ in range(3):
        assert t.update(None, 8.0) is not None          # 3프레임까지는 유지
    assert t.update(None, 8.0) is None                  # 넘으면 놓아준다


def test_tracker_new_car_gets_new_id():
    from driveloop.perception.lead import LeadTracker
    t = LeadTracker(dt=0.05)
    a = [t.update(d, 8.0) for d in (30.0, 29.8)][-1]
    b = [t.update(d, 8.0) for d in (12.0, 11.9)][-1]    # 끼어들기: 거리가 갑자기 바뀜
    assert a.id != b.id


def test_car_off_the_curve_is_not_lead():
    # 커브를 돌지 않고 직진 방향에 있는 차 (다른 길) → 내 앞차가 아님
    assert find_lead(CURVE, (0.0, 0.0), 2.4, [Obstacle(1, 34.0, 0.0)]) is None
