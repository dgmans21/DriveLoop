"""CARLA 좌표 → 이미지 투영 (CARLA 비의존, numpy).

CARLA(UE4) 좌표계: 왼손 좌표계, x 전방 · y 오른쪽 · z 위, 각도는 degree.
카메라 좌표 (x 전방, y 오른쪽, z 위) → 이미지 좌표 (u 오른쪽, v 아래): u = fx·y/x + cx, v = -fy·z/x + cy
"""
from __future__ import annotations

import math

import numpy as np


def rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """carla.Transform.get_matrix()와 같은 회전 행렬."""
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])


def transform_matrix(t: dict) -> np.ndarray:
    """{x,y,z,pitch,yaw,roll} → 4x4 (로컬 → 월드)."""
    m = np.eye(4)
    m[:3, :3] = rotation_matrix(t["pitch"], t["yaw"], t["roll"])
    m[:3, 3] = (t["x"], t["y"], t["z"])
    return m


_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def box_corners(bb: dict) -> np.ndarray:
    """bbox dict {x,y,z,ex,ey,ez,pitch,yaw,roll} → 기준 좌표계에서의 꼭짓점 8개 (8, 3)."""
    ext = np.array([bb["ex"], bb["ey"], bb["ez"]])
    rot = rotation_matrix(bb["pitch"], bb["yaw"], bb["roll"])
    return (_CORNERS * ext) @ rot.T + np.array([bb["x"], bb["y"], bb["z"]])


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def world_to_camera(points_world: np.ndarray, camera_tf: dict) -> np.ndarray:
    """월드 좌표 (N, 3) → 카메라 좌표 (N, 3)."""
    inv = np.linalg.inv(transform_matrix(camera_tf))
    return (to_homogeneous(points_world) @ inv.T)[:, :3]


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """카메라 좌표 (N, 3) → 픽셀 (N, 2). 깊이(x)가 양수인 점만 의미 있음."""
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    u = K[0, 0] * y / x + K[0, 2]
    v = -K[1, 1] * z / x + K[1, 2]
    return np.stack([u, v], axis=1)


def box_to_2d(corners_world: np.ndarray, camera_tf: dict, K: np.ndarray, width: int, height: int,
              min_depth: float = 0.5) -> tuple[tuple[float, float, float, float], float] | None:
    """3D 박스 꼭짓점 → (이미지 안으로 자른 2D 박스 x1,y1,x2,y2, 최소 깊이).

    꼭짓점이 하나라도 카메라 뒤(깊이 < min_depth)에 있으면 투영이 왜곡되므로 None.
    화면 밖이면 None.
    """
    cam = world_to_camera(corners_world, camera_tf)
    if (cam[:, 0] < min_depth).any():
        return None
    uv = project(cam, K)
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2), float(cam[:, 0].min())
