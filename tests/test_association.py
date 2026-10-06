import numpy as np

from driveloop.data.snapshot import camera_intrinsics
from driveloop.perception.association import (Detection, TemporalVoter, associate,
                                              expected_head_boxes)

W, H = 320, 180
K = np.array(camera_intrinsics(W, H, 90))
CAM = {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "roll": 0}
HEAD = {"x": 20, "y": 0, "z": 0, "ex": 0.2, "ey": 0.2, "ez": 0.6, "pitch": 0, "yaw": 90, "roll": 0}  # 카메라를 향함


def test_expected_box_for_facing_head_and_none_for_back_side():
    boxes = expected_head_boxes([HEAD], CAM, K, W, H)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 < W / 2 < x2 and y1 < H / 2 < y2
    assert expected_head_boxes([{**HEAD, "yaw": -90}], CAM, K, W, H) == []


def test_associate_picks_detection_at_expected_location():
    exp = [(158, 85, 162, 95)]
    dets = [Detection("tl_green", (157, 84, 163, 96), 0.6),     # 내 신호
            Detection("tl_red", (40, 60, 44, 70), 0.95),        # 다른 차선 신호 (확신도 높아도 무시)
            Detection("tl_red", (150, 160, 156, 170), 0.5)]     # 노면 반사광
    a = associate(exp, dets)
    assert a is not None and a.state == "GREEN" and a.det.box == (157, 84, 163, 96)


def test_associate_uses_center_distance_for_tiny_far_lights():
    exp = [(100, 50, 102, 56)]                     # 6px 높이, 투영 오차로 IoU 0
    dets = [Detection("tl_red", (103, 52, 105, 58), 0.4)]
    assert associate(exp, dets).state == "RED"


def test_associate_ignores_vehicles_and_far_detections():
    exp = [(158, 85, 162, 95)]
    assert associate(exp, [Detection("vehicle", (150, 80, 170, 100), 0.9)]) is None
    assert associate(exp, [Detection("tl_red", (10, 10, 14, 20), 0.9)]) is None
    assert associate([], [Detection("tl_red", (158, 85, 162, 95), 0.9)]) is None


def test_voter_majority_and_unknown_until_enough_votes():
    v = TemporalVoter(window=5, min_votes=2, red_votes=2)
    assert v.update("GREEN") == "UNKNOWN"
    assert v.update("GREEN") == "GREEN"
    assert v.update(None) == "GREEN"          # 한 프레임 놓쳐도 유지


def test_voter_red_priority():
    v = TemporalVoter(window=5, min_votes=2, red_votes=2)
    for s in ("GREEN", "GREEN", "GREEN", "RED", "RED"):
        out = v.update(s)
    assert out == "RED"


def test_voter_single_red_glitch_does_not_flip():
    v = TemporalVoter(window=5, min_votes=2, red_votes=2)
    for s in ("GREEN", "GREEN", "RED", "GREEN"):
        out = v.update(s)
    assert out == "GREEN"


def test_voter_goes_unknown_when_light_lost_for_long():
    v = TemporalVoter(window=5, min_votes=2, red_votes=2)
    for s in ("GREEN", "GREEN", None, None, None, None):
        out = v.update(s)
    assert out == "UNKNOWN"
