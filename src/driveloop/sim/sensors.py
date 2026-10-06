"""센서 부착과 동기 모드 프레임 수신."""
from __future__ import annotations

import queue

import carla
import numpy as np

from driveloop.config import CameraConfig

# 차량 뒤쪽 위에서 내려다보는 3인칭 카메라 (화면 표시용)
CHASE_CAMERA = carla.Transform(carla.Location(x=-6.0, z=3.0), carla.Rotation(pitch=-12.0))


def attach_rgb_camera(world: carla.World, parent: carla.Actor, cam: CameraConfig,
                      transform: carla.Transform = CHASE_CAMERA) -> tuple[carla.Sensor, queue.Queue]:
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(cam.width))
    bp.set_attribute("image_size_y", str(cam.height))
    bp.set_attribute("fov", str(cam.fov))
    sensor = world.spawn_actor(bp, transform, attach_to=parent)
    q: queue.Queue = queue.Queue()
    sensor.listen(q.put)
    return sensor, q


def get_frame(q: queue.Queue, frame: int, timeout: float = 5.0):
    """world.tick()이 돌려준 frame 번호와 일치하는 센서 데이터를 꺼낸다 (이전 프레임은 버림)."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame >= frame:
            return data


def image_to_rgb(image: carla.Image) -> np.ndarray:
    """carla.Image(BGRA) → (H, W, 3) RGB uint8."""
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return bgra[:, :, 2::-1]
