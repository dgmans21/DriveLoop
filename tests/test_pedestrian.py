import pytest

from driveloop.perception.lead import Obstacle
from driveloop.planning.acc import acc_target, smooth_target
from driveloop.planning.pedestrian import PedestrianYielder

STRAIGHT = [(float(x), 0.0) for x in range(0, 202, 2)]          # x축을 따라 200m, 왼쪽이 +y
FRONT = 2.4                                                       # 차량 중심 → 앞 범퍼
CRUISE = 30 / 3.6


def ped(x, y, vx=0.0, vy=0.0, id=1):
    return Obstacle(id, x, y, vx, vy, half_length=0.3)


def yields(o, ego_x=0.0, speed=CRUISE, y=None):
    y = y or PedestrianYielder()
    return y.step(STRAIGHT, (ego_x, 0.0), FRONT, [o], speed)


def test_pedestrian_in_my_path_is_yielded_with_bumper_gap():
    r = yields(ped(30.0, 0.5))
    assert r.reason == "in_path" and r.distance == pytest.approx(30 - FRONT - 0.3)


def test_standing_on_curb_or_walking_along_sidewalk_is_ignored():
    assert yields(ped(20.0, 2.5)) is None                     # 차도 옆에 서 있음
    assert yields(ped(20.0, -3.0, vx=1.4)) is None            # 보도에서 나란히 걸음


def test_walking_toward_my_path_is_yielded_before_entering():
    r = yields(ped(25.0, -3.0, vy=1.4))                       # 1.1초 뒤 통로 진입, 나는 ~3초 뒤 도착
    assert r is not None and r.reason == "entering"


def test_walking_away_after_crossing_is_not_yielded():
    assert yields(ped(20.0, 2.0, vy=1.4)) is None


def test_slow_pedestrian_that_arrives_after_i_pass_is_not_yielded():
    # 10m 앞, 통로까지 3.5m를 1 m/s로 → 3.5초 뒤 진입. 나는 1.4초면 지나감 (+여유 1초)
    assert yields(ped(12.4, -5.0, vy=1.0)) is None


def test_pedestrian_behind_my_front_bumper_is_ignored():
    assert yields(ped(-3.0, 0.0), ego_x=0.0) is None


def test_yield_is_held_near_corridor_edge_until_clearly_out():
    y = PedestrianYielder()
    assert yields(ped(20.0, 1.2), y=y).reason == "in_path"
    assert yields(ped(20.0, 1.7), y=y).reason == "held"       # 경계 바로 밖에 멈춤 → 계속 기다림
    assert yields(ped(20.0, 2.2), y=y) is None                # 여유 0.5m를 벗어나면 해제


def test_nearest_of_several_yielded_pedestrians_is_returned():
    r = PedestrianYielder().step(STRAIGHT, (0.0, 0.0), FRONT,
                                 [ped(35.0, 0.0, id=1), ped(18.0, -0.5, id=2), ped(10.0, 4.0, id=3)], CRUISE)
    assert r.id == 2


# ---- 1차원 모의 주행: 자차는 목표 속도를 가속 +2 / 감속 −6 m/s² 한계로 따라간다 (에이전트와 같은 목표 속도 계산) ----

def simulate(walk, seconds=20.0, dt=0.05, ego_x=0.0, lag=0.0, edge=None):
    """walk(t) -> (x, y, vx, vy). 반환: 매 tick (t, ego_x, v, 보행자 위치, 양보 여부, 가속도).

    lag: 인지 지연(초) — 카메라는 보이는 몸이 늦게 움직이고 추적 필터도 지연을 더한다 (모델 실측 약 0.4초)
    edge: 오른쪽 보도 경계 옆 거리 (주의 서행용 지도 정보). 판단은 에이전트와 같다
    """
    from collections import deque
    yielder, v, prev, log = PedestrianYielder(), CRUISE, None, []
    seen = deque(maxlen=round(lag / dt) + 1)
    road_edge = (lambda seg, side: edge if side > 0 else None) if edge is not None else None
    for k in range(int(seconds / dt)):
        t = k * dt
        seen.append(walk(t))
        px, py, vx, vy = seen[0]
        r = yielder.step(STRAIGHT, (ego_x, 0.0), FRONT, [ped(px, py, vx, vy)], v, road_edge)
        acc = acc_target(r.distance if r else None, 0.0, v)
        limit = smooth_target(acc, prev, 2.0, dt)
        caution = yielder.caution_speed if yielder.caution_speed is not None else float("inf")
        target = min(CRUISE, limit if limit is not None else float("inf"), caution)
        px, py, vx, vy = walk(t)                                  # 충돌 판정은 실제 위치로
        prev = target
        a = max(-6.0, min(2.0, (target - v) / 0.25))
        v = max(0.0, v + a * dt)
        ego_x += v * dt
        log.append((t, ego_x, v, px, py, r is not None, a))
    return log


def collided(log):
    # 차체(길이 4.8, 폭 2.0)와 보행자(반경 0.3)가 겹침
    return any(ex - FRONT - 0.3 <= px <= ex + FRONT + 0.3 and abs(py) <= 1.3 for _, ex, _, px, py, _, _ in log)


def crossing(x, y0, speed, start=0.0, y1=6.0):
    def walk(t):
        if t < start:
            return x, y0, 0.0, 0.0
        y = min(y0 + speed * (t - start), y1)
        return x, y, 0.0, (speed if y < y1 else 0.0)
    return walk


def test_sim_crossing_pedestrian_slow_down_and_pass_after():
    log = simulate(crossing(45.0, -4.0, 1.4))                 # 45m 앞 횡단 (보통 걸음, 통로 3m를 약 2초에)
    assert not collided(log)
    assert min(v for _, _, v, *_ in log) < CRUISE - 2.0       # 건너는 동안 속도를 줄였고
    assert min(a for *_, a in log) >= -3.0                    # 멀리서 봤으니 급제동 없이
    assert log[-1][2] > 6.0                                   # 다 건넌 뒤 다시 속도를 낸다


def test_sim_pedestrian_stopping_in_my_lane_stop_wait_and_go():
    # 45m 앞에서 건너다 통로 한가운데 12초 멈춤 → 그 뒤 마저 건넘
    def walk(t):
        if t < 4.0:
            return 45.0, -4.0 + 1.0 * t, 0.0, 1.0
        if t < 16.0:
            return 45.0, 0.0, 0.0, 0.0
        y = min(1.0 * (t - 16.0), 6.0)
        return 45.0, y, 0.0, (1.0 if y < 6.0 else 0.0)
    log = simulate(walk, seconds=30)
    assert not collided(log)
    stopped = [(ex, px) for t, ex, v, px, *_ in log if v < 0.1]
    assert stopped, "멈추지 않음"
    gap = stopped[-1][1] - 0.3 - (stopped[-1][0] + FRONT)
    assert 3.0 <= gap <= 7.0                                  # 정지 간격 5m 근처에서 기다림
    assert min(a for *_, a in log) >= -3.0
    assert log[-1][2] > 6.0


def test_sim_sidewalk_walker_causes_no_slowdown():
    # 오른쪽(+, CARLA 진행 방향 오른쪽) 보도 위 (경계 2.88 밖)를 나란히 걸음
    log = simulate(lambda t: (30.0 + 1.4 * t, 3.3, 1.4, 0.0), seconds=10, edge=2.88)
    assert min(v for _, _, v, *_ in log) == pytest.approx(CRUISE)


def test_sim_pedestrian_standing_on_road_edge_slows_but_never_stops():
    log = simulate(lambda t: (40.0, 2.1, 0.0, 0.0), seconds=10, edge=2.88)   # 차도 위 가장자리에 서 있음
    vmin = min(v for _, _, v, *_ in log)
    assert 3.0 - 1e-6 <= vmin < CRUISE - 2.0                   # 옆을 지날 땐 서행하지만 멈추지는 않는다


def test_sim_runner_darting_out_close_is_avoided():
    # 18m 앞 차도 가장자리(−2.5m)에서 2.5 m/s로 뛰어듦 — 진입까지 0.4초
    log = simulate(crossing(20.4, -2.5, 2.5))
    assert not collided(log)


def test_sim_dart_out_inside_stopping_distance_is_physically_unavoidable():
    # 한계 확인용: 범퍼 4m 앞에서 뛰어들면 30km/h·6 m/s²로도 못 선다 (정지 거리 5.8m) → 판단이 아니라 속도의 문제
    log = simulate(crossing(6.7, -1.8, 3.0))
    assert collided(log)



# ---- 주의 서행: 차도 위 가장자리 보행자 (보도 위는 대상 아님) ----

def edge_at(u):
    return lambda seg, side: u if side > 0 else None


def test_pedestrian_standing_on_road_edge_triggers_caution_not_yield():
    y = PedestrianYielder()
    r = y.step(STRAIGHT, (0.0, 0.0), FRONT, [ped(14.0, 2.1)], CRUISE, road_edge=edge_at(2.88))   # 약 11m 앞
    assert r is None                                           # 양보(정지) 대상은 아님
    assert y.caution_speed is not None and y.caution_speed < CRUISE


def test_pedestrian_on_sidewalk_does_not_trigger_caution():
    y = PedestrianYielder()
    y.step(STRAIGHT, (0.0, 0.0), FRONT, [ped(14.0, 3.3)], CRUISE, road_edge=edge_at(2.88))   # 보도 위 (경계 2.88 밖)
    assert y.caution_speed is None


def test_caution_never_forces_a_stop_and_relaxes_with_distance():
    y = PedestrianYielder()
    assert y.caution_limit(0.5) == pytest.approx(3.0)          # 바로 옆을 지날 때도 최저 3 m/s (멈추지 않음)
    assert y.caution_limit(10.0) < y.caution_limit(18.0) < y.caution_limit(40.0)
    v = y.caution_limit(18.0)
    assert v * 0.5 + v * v / (2 * 3.0) <= 18.0 - 5.0 + 1e-6    # 0.5초 지연 + 3 m/s²로 정지 간격 5m 앞에 설 수 있음


def test_caution_ignores_pedestrian_walking_away():
    y = PedestrianYielder()
    y.step(STRAIGHT, (0.0, 0.0), FRONT, [ped(14.0, 2.1, vy=1.4)], CRUISE, road_edge=edge_at(2.88))   # 멀어지는 중
    assert y.caution_speed is None


def test_sim_late_perception_dart_out_brakes_hard_but_not_maximum():
    # 18m 앞 차도 가장자리(보도 경계 2.88 안쪽)에서 3 m/s로 뛰어듦, 인지 0.4초 지연 (모델 실측)
    def walk(t, start=0.5):
        if t < start:
            return 22.8, -2.1, 0.0, 0.0
        y = min(-2.1 + 3.0 * (t - start), 6.0)
        return 22.8, y, 0.0, (3.0 if y < 6.0 else 0.0)
    log = simulate(walk, seconds=12, lag=0.4, edge=2.88)
    assert not collided(log)
    assert min(a for *_, a in log) >= -4.0                     # 주의 서행 없이(또는 여유 1m)는 −6 한계까지 밟았다
