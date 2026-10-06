import numpy as np
import pytest

from driveloop.data.geometry import box_corners, box_to_2d, project, rotation_matrix, world_to_camera
from driveloop.data.snapshot import camera_intrinsics

K = np.array(camera_intrinsics(1280, 720, 90))  # fx = 640
CAM = {"x": 0, "y": 0, "z": 0, "pitch": 0, "yaw": 0, "roll": 0}


def test_rotation_yaw_90_maps_x_to_y():
    assert rotation_matrix(0, 90, 0) @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-9)


def test_point_straight_ahead_projects_to_center():
    uv = project(world_to_camera(np.array([[10.0, 0, 0]]), CAM), K)
    assert uv[0] == pytest.approx([640, 360])


def test_right_and_up_directions():
    uv = project(world_to_camera(np.array([[10.0, 10.0, 0], [10.0, 0, 5.0]]), CAM), K)
    assert uv[0] == pytest.approx([1280, 360])   # 오른쪽 45° → 화면 오른쪽 끝 (fov 90)
    assert uv[1] == pytest.approx([640, 40])     # 위쪽 → v 감소


def test_camera_pose_is_applied():
    cam = {**CAM, "x": 5.0, "yaw": 90.0}      # (5,0)에서 +y 방향을 봄
    uv = project(world_to_camera(np.array([[5.0, 10.0, 0]]), cam), K)
    assert uv[0] == pytest.approx([640, 360])


def test_box_to_2d_symmetric_box():
    bb = {"x": 10, "y": 0, "z": 0, "ex": 1, "ey": 1, "ez": 1, "pitch": 0, "yaw": 0, "roll": 0}
    (x1, y1, x2, y2), depth = box_to_2d(box_corners(bb), CAM, K, 1280, 720)
    assert depth == pytest.approx(9.0)
    assert (x1 + x2) / 2 == pytest.approx(640) and (y1 + y2) / 2 == pytest.approx(360)
    assert x2 - x1 == pytest.approx(2 * 640 / 9)  # 가까운 면(깊이 9) 기준


def test_box_behind_camera_is_none():
    bb = {"x": -10, "y": 0, "z": 0, "ex": 1, "ey": 1, "ez": 1, "pitch": 0, "yaw": 0, "roll": 0}
    assert box_to_2d(box_corners(bb), CAM, K, 1280, 720) is None


def test_box_is_clipped_to_image():
    bb = {"x": 10, "y": 12, "z": 0, "ex": 1, "ey": 3, "ez": 1, "pitch": 0, "yaw": 0, "roll": 0}
    (x1, _, x2, _), _ = box_to_2d(box_corners(bb), CAM, K, 1280, 720)
    assert x2 == 1280 and x1 < 1280
