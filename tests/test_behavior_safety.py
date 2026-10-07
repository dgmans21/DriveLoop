"""3단계 비교에서 찾은 위험 장면에 대한 안전장치 (planning/behavior.py 문서 참고)."""
import math

import pytest

from driveloop.config import DrivingConfig
from driveloop.perception.types import PerceptionOutput, TLState
from driveloop.planning.behavior import BehaviorState as S
from driveloop.planning.behavior import TrafficLightBehavior

CRUISE = 30 / 3.6
U, R, Y, G = TLState.UNKNOWN, TLState.RED, TLState.YELLOW, TLState.GREEN


def run(b, seq):
    """seq: (신호, 정지선 거리, 속도) → 마지막 Decision."""
    d = None
    for tl, dist, v in seq:
        d = b.step(PerceptionOutput(tl, dist), v)
    return d


@pytest.fixture
def b():
    return TrafficLightBehavior(DrivingConfig())


# ---- 1) 색을 모르면 딜레마 존에 들어가지 않는다 ----

def test_unknown_far_away_does_not_slow_down(b):
    d = run(b, [(U, 45.0, CRUISE)])
    assert d.state is S.CAUTION and d.target_speed == pytest.approx(CRUISE)


def test_unknown_near_line_limits_speed_to_comfortable_stop(b):
    d = run(b, [(U, 12.0, CRUISE)])
    assert d.target_speed == pytest.approx(math.sqrt(2 * 2.0 * (12.0 - 2.0)))
    assert d.target_speed < CRUISE


def test_light_seen_after_caution_needs_only_comfortable_braking(b):
    # 시드5 사례: 23m에서 노랑으로 바뀌었는데 12m까지 모름 → CAUTION 덕에 속도가 이미 제한됨
    run(b, [(U, 20.0, CRUISE), (U, 15.0, 7.5), (U, 12.0, 6.3)])
    d = b.step(PerceptionOutput(Y, 12.0), 6.3)
    assert d.state is S.STOPPING


def test_unknown_until_the_line_stops_there_then_goes_on_green(b):
    d = run(b, [(U, 10.0, 5.0), (U, 3.0, 1.0), (U, 2.5, 0.1)])
    assert d.state is S.STOPPED and d.target_speed == 0.0
    assert b.step(PerceptionOutput(G, 2.5), 0.0).state is S.CRUISE


def test_unknown_right_after_green_keeps_cruising(b):
    # 정지선 바로 앞에서 신호가 화면 밖으로 나가 UNKNOWN → 초록 직후라면 감속하지 않는다
    d = run(b, [(G, 6.0, CRUISE), (U, 5.0, CRUISE), (U, 4.0, CRUISE)])
    assert d.state is S.CRUISE and d.target_speed == pytest.approx(CRUISE)


def test_unknown_long_after_green_becomes_caution(b):
    seq = [(G, 40.0, CRUISE)] + [(U, 40.0 - i * 0.4, CRUISE) for i in range(1, 25)]
    assert run(b, seq).state is S.CAUTION


# ---- 2) 초록 → 빨강은 없는 순서: 설 수 없으면 노란불 딜레마로 ----

def test_red_right_after_green_when_cannot_stop_is_treated_as_yellow(b):
    # 시드3 사례: 정지선 4m 앞, 28 km/h에서 노랑을 빨강으로 1프레임 오인
    d = run(b, [(G, 5.0, 7.8), (R, 4.2, 7.8)])
    assert d.state is S.PROCEED_YELLOW


def test_red_right_after_green_but_stoppable_still_stops(b):
    d = run(b, [(G, 30.0, CRUISE), (R, 29.0, CRUISE)])
    assert d.state is S.STOPPING


def test_red_without_recent_green_still_stops_even_if_hard(b):
    # 초록을 본 적이 없으면(처음 보는 신호가 빨강) 기존대로 정지 — 위반보다 급제동
    assert run(b, [(R, 4.0, 7.8)]).state is S.STOPPING


def test_green_memory_resets_on_next_light(b):
    d = run(b, [(G, 3.0, 7.8), (None, None, 7.8), (R, 4.0, 7.8)])
    assert d.state is S.STOPPING
