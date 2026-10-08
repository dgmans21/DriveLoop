import pytest

from driveloop.perception.mono_distance import ground_distance, ray_road_hit, width_distance

F, CY, H = 640.0, 360.0, 1.5


def test_ground_distance_matches_pinhole_geometry():
    # 20m 앞 도로 위 점은 지평선 아래 f·H/Z = 48 px
    assert ground_distance(CY + 48.0, F, CY, H) == pytest.approx(20.0)


def test_closer_car_has_lower_bottom_edge():
    assert ground_distance(CY + 200, F, CY, H) < ground_distance(CY + 50, F, CY, H)


def test_bottom_at_or_above_horizon_is_not_on_the_road():
    assert ground_distance(CY, F, CY, H) is None
    assert ground_distance(CY - 30, F, CY, H) is None


def _ray(v_px, f=F, cy=CY):
    """핀홀: 지평선 아래 v_px 픽셀 → 수평 1, 아래로 v_px/f 기울기의 광선 (x 앞, z 위)."""
    return (1.0, 0.0, -v_px / f)


def test_ray_on_flat_road_matches_ground_distance():
    road = [(float(x), 0.0, 0.0) for x in range(0, 100)]
    hit = ray_road_hit((0.0, 0.0, H), _ray(48.0), road)
    assert hit[0] == pytest.approx(ground_distance(CY + 48.0, F, CY, H), abs=0.05)   # 20m


def test_uphill_road_gives_closer_point_than_flat_assumption():
    # 10m부터 5° 오르막: 같은 픽셀이라도 실제 도로와 더 가까이서 만난다 (평면 가정은 멀게 봄 → 늦은 제동)
    import math
    road = [(float(x), 0.0, max(0.0, (x - 10) * math.tan(math.radians(5)))) for x in range(0, 100)]
    flat = ground_distance(CY + 30.0, F, CY, H)
    hit = ray_road_hit((0.0, 0.0, H), _ray(30.0), road)
    assert hit[0] < flat - 5.0


def test_ray_above_horizon_never_hits():
    road = [(float(x), 0.0, 0.0) for x in range(0, 100)]
    assert ray_road_hit((0.0, 0.0, H), (1.0, 0.0, 0.01), road) is None


def test_width_distance():
    assert width_distance(64.0, F, 1.8) == pytest.approx(18.0)
    assert width_distance(0.0, F) is None
