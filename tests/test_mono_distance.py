import pytest

from driveloop.perception.mono_distance import ground_distance, width_distance

F, CY, H = 640.0, 360.0, 1.5


def test_ground_distance_matches_pinhole_geometry():
    # 20m 앞 도로 위 점은 지평선 아래 f·H/Z = 48 px
    assert ground_distance(CY + 48.0, F, CY, H) == pytest.approx(20.0)


def test_closer_car_has_lower_bottom_edge():
    assert ground_distance(CY + 200, F, CY, H) < ground_distance(CY + 50, F, CY, H)


def test_bottom_at_or_above_horizon_is_not_on_the_road():
    assert ground_distance(CY, F, CY, H) is None
    assert ground_distance(CY - 30, F, CY, H) is None


def test_width_distance():
    assert width_distance(64.0, F, 1.8) == pytest.approx(18.0)
    assert width_distance(0.0, F) is None
