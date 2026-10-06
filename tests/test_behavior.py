import pytest

from driveloop.config import DrivingConfig
from driveloop.perception.types import PerceptionOutput, TLState
from driveloop.planning.behavior import BehaviorState as S
from driveloop.planning.behavior import TrafficLightBehavior

CRUISE = 30 / 3.6


@pytest.fixture
def b():
    return TrafficLightBehavior(DrivingConfig())


def see(state, dist):
    return PerceptionOutput(state, dist)


def test_no_light_cruises(b):
    d = b.step(PerceptionOutput(), CRUISE)
    assert d.state is S.CRUISE and d.target_speed == pytest.approx(CRUISE)


def test_green_cruises(b):
    assert b.step(see(TLState.GREEN, 10), CRUISE).state is S.CRUISE


def test_red_far_starts_smooth_braking(b):
    d = b.step(see(TLState.RED, 40), CRUISE)
    assert d.state is S.STOPPING
    assert d.target_speed <= CRUISE


def test_red_target_speed_decreases_toward_line(b):
    targets = [b.step(see(TLState.RED, dist), 5.0).target_speed for dist in (30, 20, 10, 5, 2)]
    assert targets == sorted(targets, reverse=True)
    assert targets[-1] == 0.0  # 정지 여유거리(2m) 도달


def test_red_then_stopped_then_green(b):
    b.step(see(TLState.RED, 20), CRUISE)
    assert b.step(see(TLState.RED, 2.5), 0.1).state is S.STOPPED
    assert b.step(see(TLState.RED, 2.5), 0.0).target_speed == 0.0
    assert b.step(see(TLState.GREEN, 2.5), 0.0).state is S.CRUISE


def test_yellow_far_stops(b):
    # 30km/h, 정지선 30m → 필요 감속 ~1.2 m/s² → 정지
    assert b.step(see(TLState.YELLOW, 30), CRUISE).state is S.STOPPING


def test_yellow_close_and_fast_proceeds(b):
    # 30km/h, 정지선 5m (gap 3m) → 필요 감속 ~11.6 m/s² → 통과
    d = b.step(see(TLState.YELLOW, 5), CRUISE)
    assert d.state is S.PROCEED_YELLOW
    assert d.target_speed == pytest.approx(CRUISE)


def test_committed_yellow_keeps_going_when_red(b):
    b.step(see(TLState.YELLOW, 5), CRUISE)
    assert b.step(see(TLState.RED, 1), CRUISE).state is S.PROCEED_YELLOW
    # 정지선 통과 후 신호등이 사라지면 순항 복귀
    assert b.step(PerceptionOutput(), CRUISE).state is S.CRUISE


def test_yellow_while_stopping_keeps_stopping(b):
    b.step(see(TLState.RED, 20), CRUISE)
    assert b.step(see(TLState.YELLOW, 15), 4.0).state is S.STOPPING


def test_unknown_keeps_previous_decision(b):
    b.step(see(TLState.RED, 20), CRUISE)
    assert b.step(see(TLState.UNKNOWN, 18), 6.0).state is S.STOPPING


def test_red_already_past_line_does_not_brake(b):
    assert b.step(see(TLState.RED, -1.0), CRUISE).state is S.CRUISE
