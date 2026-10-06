import numpy as np

from driveloop.data.capture_policy import CaptureScheduler, yellow_in_view
from driveloop.data.snapshot import camera_intrinsics

W, H = 320, 180
K = np.array(camera_intrinsics(W, H, 90))
CAM = {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "roll": 0}
# 카메라 20m 앞, 정면(+y축)이 카메라를 향함
CATALOG = {"1": {"light_boxes": [{"x": 20, "y": 0, "z": 0, "ex": 0.2, "ey": 0.2, "ez": 0.6,
                                  "pitch": 0, "yaw": 90, "roll": 0}]}}


def run(scheduler, ticks, yellow_ticks=()):
    return [(k, r) for k in range(ticks) if (r := scheduler.decide(k, k in yellow_ticks))]


def test_regular_only_matches_old_fixed_interval():
    assert [k for k, _ in run(CaptureScheduler(10), 35)] == [0, 10, 20, 30]


def test_yellow_captures_densely_then_regular_resumes():
    got = run(CaptureScheduler(10, 2), 30, yellow_ticks=set(range(3, 9)))
    assert got[:5] == [(0, "regular"), (3, "yellow"), (5, "yellow"), (7, "yellow"), (17, "regular")]


def test_fast_disabled_ignores_yellow():
    assert [k for k, _ in run(CaptureScheduler(10, None), 25, yellow_ticks=set(range(25)))] == [0, 10, 20]


def test_yellow_in_view_true_when_facing_camera():
    assert yellow_in_view({"1": "yellow"}, CATALOG, CAM, K, W, H, 70, 80)


def test_yellow_in_view_false_for_other_states_back_side_or_far():
    assert not yellow_in_view({"1": "red"}, CATALOG, CAM, K, W, H, 70, 80)
    back = {"1": {"light_boxes": [{**CATALOG["1"]["light_boxes"][0], "yaw": -90}]}}
    assert not yellow_in_view({"1": "yellow"}, back, CAM, K, W, H, 70, 80)
    assert not yellow_in_view({"1": "yellow"}, CATALOG, CAM, K, W, H, 70, 10)


def test_yellow_behind_camera_not_in_view():
    behind = {"1": {"light_boxes": [{**CATALOG["1"]["light_boxes"][0], "x": -20, "yaw": -90}]}}
    assert not yellow_in_view({"1": "yellow"}, behind, CAM, K, W, H, 70, 80)
