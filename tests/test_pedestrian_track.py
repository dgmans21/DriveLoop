import pytest

from driveloop.perception.pedestrian_track import PedestrianTracker

DT = 0.05


def test_needs_two_frames_to_confirm():
    t = PedestrianTracker(DT)
    assert t.update([(20.0, 3.0)]) == ()                      # 1프레임 오검출로는 보행자가 생기지 않는다
    assert len(t.update([(20.0, 3.0)])) == 1


def test_crossing_speed_converges():
    t = PedestrianTracker(DT)
    out = ()
    for k in range(60):                                        # y 방향 1.4 m/s (차도를 건넘)
        out = t.update([(25.0, 3.0 - 1.4 * DT * k)])
    (p,) = out
    assert p.vy == pytest.approx(-1.4, abs=0.15) and abs(p.vx) < 0.1


def test_two_pedestrians_keep_their_ids():
    t = PedestrianTracker(DT)
    for k in range(10):
        out = t.update([(20.0, 3.0 - 0.07 * k), (30.0, -3.0)])
    ids = {round(p.x): p.id for p in out}
    for k in range(10, 20):                                    # 측정 순서가 바뀌어도 같은 사람은 같은 id
        out = t.update([(30.0, -3.0), (20.0, 3.0 - 0.07 * k)])
    assert {round(p.x): p.id for p in out} == ids


def test_short_miss_is_held_then_dropped():
    t = PedestrianTracker(DT, max_missed=3)
    t.update([(20.0, 0.0)])
    t.update([(20.0, 0.0)])
    for _ in range(3):
        assert len(t.update([])) == 1                          # 3프레임 놓쳐도 유지 (예측 위치)
    assert t.update([]) == ()


def test_far_jump_starts_a_new_track():
    t = PedestrianTracker(DT)
    t.update([(20.0, 0.0)])
    a = t.update([(20.0, 0.0)])[0].id
    out = t.update([(26.0, 0.0)])                              # 6m 점프 = 다른 사람 (또는 오검출)
    assert all(p.id == a for p in out)                         # 새 측정은 아직 확정 전, 기존 사람은 유지
    assert {p.id for p in t.update([(26.0, 0.0)])} == {a, a + 1}   # 새 사람 확정 + 기존은 놓친 채 유지(3프레임까지)


def test_established_pedestrian_is_held_for_two_seconds_when_lost():
    t = PedestrianTracker(DT)                                  # 10프레임(0.5초) 이상 본 사람은 40프레임(2초) 유지
    for _ in range(12):
        t.update([(9.0, 0.3)])
    for _ in range(40):
        (p,) = t.update([])                                    # 비 오는 밤 차선 가운데에서 2초 놓쳐도 계속 있음
    assert p.x == pytest.approx(9.0, abs=0.05)
    assert t.update([]) == ()


def test_brief_false_detection_is_still_dropped_quickly():
    t = PedestrianTracker(DT)
    t.update([(15.0, 0.0)])
    t.update([(15.0, 0.0)])                                    # 2프레임 오검출 → 확정은 되지만
    for _ in range(3):
        t.update([])
    assert t.update([]) == ()                                  # 3프레임 넘게 놓치면 바로 지운다 (헛양보 방지)


def test_coasting_speed_decays():
    t = PedestrianTracker(DT)
    for k in range(20):
        t.update([(20.0, 3.0 - 1.4 * DT * k)])
    (p,) = t.update([])
    v0 = abs(p.vy)
    for _ in range(10):
        (p,) = t.update([])
    assert abs(p.vy) < 0.5 * v0                                # 놓친 동안 걷는 속도를 계속 끌고 가지 않음
