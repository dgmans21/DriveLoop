import numpy as np
import pytest

from driveloop.data.export import apply_ignore_mask, yolo_line


def test_yolo_line_normalized_center_format():
    assert yolo_line(2, (100, 50, 200, 150), 400, 200) == "2 0.375000 0.500000 0.250000 0.500000"


def test_ignore_mask_fills_region_with_pad():
    img = np.zeros((20, 20, 3), np.uint8)
    out = apply_ignore_mask(img, [(5, 5, 10, 10)], [], (114, 114, 114), pad=1)
    assert (out[4:11, 4:11] == 114).all()
    assert (out[0:3, 0:3] == 0).all()
    assert (img == 0).all()  # 원본은 그대로


def test_ignore_mask_never_covers_labeled_objects():
    img = np.zeros((20, 20, 3), np.uint8)
    out = apply_ignore_mask(img, [(0, 0, 20, 20)], [(8, 8, 12, 12)], (114, 114, 114), pad=0)
    assert (out[8:12, 8:12] == 0).all()
    assert (out[0:8, :] == 114).all()


def test_no_ignore_returns_same_image():
    img = np.ones((4, 4, 3), np.uint8)
    assert apply_ignore_mask(img, [], [(0, 0, 2, 2)], (114, 114, 114), 2) is img


@pytest.mark.parametrize("box", [(0, 0, 400, 200), (399, 199, 400, 200)])
def test_yolo_values_within_unit_range(box):
    vals = [float(v) for v in yolo_line(0, box, 400, 200).split()[1:]]
    assert all(0 <= v <= 1 for v in vals)
