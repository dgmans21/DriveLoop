import numpy as np
import pytest

from driveloop.eval.detection import greedy_match, iou_matrix, match_image

TL = {"tl_red": "traffic_light", "tl_yellow": "traffic_light", "tl_green": "traffic_light"}


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0


def test_greedy_prefers_high_confidence_prediction():
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], float)
    assert greedy_match(gt, pred, np.array([0.3, 0.9])) == [(0, 1)]


def test_correct_detection_and_false_positive():
    gt = [{"cls": "vehicle", "box": (0, 0, 10, 10)}]
    pred = [{"cls": "vehicle", "box": (0, 0, 10, 10), "conf": 0.9},
            {"cls": "vehicle", "box": (50, 50, 60, 60), "conf": 0.8}]
    r = match_image(gt, pred, class_agnostic_groups=TL)
    assert r["gt"][0]["matched_cls"] == "vehicle" and len(r["fp"]) == 1


def test_traffic_light_color_confusion_is_recorded():
    gt = [{"cls": "tl_red", "box": (0, 0, 4, 10)}]
    pred = [{"cls": "tl_yellow", "box": (0, 0, 4, 10), "conf": 0.7}]
    r = match_image(gt, pred, class_agnostic_groups=TL)
    assert r["gt"][0]["matched_cls"] == "tl_yellow" and r["fp"] == []


def test_vehicle_never_matches_traffic_light():
    gt = [{"cls": "vehicle", "box": (0, 0, 10, 10)}]
    pred = [{"cls": "tl_red", "box": (0, 0, 10, 10), "conf": 0.9}]
    r = match_image(gt, pred, class_agnostic_groups=TL)
    assert r["gt"][0]["matched_cls"] is None and len(r["fp"]) == 1


def test_missed_object():
    r = match_image([{"cls": "tl_green", "box": (0, 0, 4, 10)}], [], class_agnostic_groups=TL)
    assert r["gt"][0]["matched_cls"] is None
