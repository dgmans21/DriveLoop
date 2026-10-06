import math

import pytest

from driveloop.planning.speed_limit import curve_speed_limit, path_curvatures

CAP = 30 / 3.6


def straight(n=30, step=2.0):
    return [(i * step, 0.0) for i in range(n)]


def straight_then_arc(straight_len, radius, step=2.0):
    """x축으로 직진 후 반경 radius로 90도 좌회전."""
    pts = [(i * step, 0.0) for i in range(int(straight_len / step) + 1)]
    x0 = pts[-1][0]
    n_arc = int((math.pi / 2 * radius) / step)
    for k in range(1, n_arc + 1):
        a = k * step / radius
        pts.append((x0 + radius * math.sin(a), -radius * (1 - math.cos(a))))
    return pts


def test_straight_has_zero_curvature_and_no_limit():
    assert max(path_curvatures(straight())) == pytest.approx(0.0)
    assert curve_speed_limit(straight(), 2.5, 2.0, CAP) == CAP


def test_curvature_matches_radius():
    pts = straight_then_arc(0, 10.0)
    assert path_curvatures(pts)[5] == pytest.approx(1 / 10.0, rel=0.05)


def test_limit_inside_curve():
    # 반경 10m, 횡가속 2.5 → 커브 속도 5 m/s. 첫 커브점이 2m 앞이라 sqrt(5^2 + 2*2*2)
    pts = straight_then_arc(0, 10.0)
    assert curve_speed_limit(pts, 2.5, 2.0, CAP) == pytest.approx(math.sqrt(25 + 8), rel=0.05)


def test_limit_relaxes_with_distance_to_curve():
    near = curve_speed_limit(straight_then_arc(4, 10.0), 2.5, 2.0, CAP)
    far = curve_speed_limit(straight_then_arc(8, 10.0), 2.5, 2.0, CAP)
    very_far = curve_speed_limit(straight_then_arc(40, 10.0), 2.5, 2.0, CAP)
    assert near < far < CAP
    assert very_far == CAP


def test_gentle_curve_not_limited():
    assert curve_speed_limit(straight_then_arc(0, 200.0), 2.5, 2.0, CAP) == CAP
