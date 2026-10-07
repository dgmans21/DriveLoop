"""내 모델(YOLO)로 하는 인지 — 1단계 정답값 인지(GroundTruthPerception)를 대체한다.

위치는 지도, 색은 모델:
  1) 지도로 내 차선 신호등과 정지선 거리를 찾는다 (GroundTruthPerception.nearest_light, 상태는 읽지 않음)
  2) 그 신호등 헤드를 전방 카메라 화면에 투영 → 예상 위치
  3) 예상 위치 근처의 모델 검출을 내 신호로 고르고 (association.associate)
  4) 최근 몇 프레임을 다수결로 안정화 (TemporalVoter, RED 우선)

판단·제어 코드는 PerceptionOutput만 보므로 그대로 재사용된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import carla
import numpy as np

from driveloop.data.snapshot import bbox_to_dict, camera_intrinsics, transform_to_dict
from driveloop.perception.association import (Association, Detection, TemporalVoter, associate,
                                              expected_head_boxes)
from driveloop.perception.ground_truth import GroundTruthPerception
from driveloop.perception.types import PerceptionOutput, TLState
from driveloop.planning.route import RoutePlanner
from driveloop.sim.sensors import image_to_rgb


@dataclass
class ModelDebug:
    """HUD·녹화·로그용 (판단에는 쓰지 않음)."""
    detections: list[Detection] = field(default_factory=list)
    expected: list = field(default_factory=list)
    association: Association | None = None
    raw_state: str | None = None
    voted_state: str = "UNKNOWN"
    tl_id: int | None = None
    infer_ms: float = 0.0


_MODELS: dict[str, object] = {}


def load_yolo(weights: str):
    """같은 프로세스에서 여러 번 주행할 때(비교 실험) 모델을 한 번만 올린다."""
    if weights not in _MODELS:
        from ultralytics import YOLO  # 무거운 import는 사용할 때만
        _MODELS[weights] = YOLO(weights)
    return _MODELS[weights]


class ModelPerception:
    def __init__(self, world: carla.World, ego: carla.Vehicle, route: RoutePlanner, camera: carla.Sensor,
                 width: int, height: int, fov: float, weights: str, lookahead: float,
                 conf: float = 0.10, imgsz: int = 1280, voter: TemporalVoter | None = None) -> None:
        self._map = GroundTruthPerception(world, ego, route, lookahead)
        self._camera = camera
        self._W, self._H = width, height
        self._K = np.array(camera_intrinsics(width, height, fov))
        self._heads = {tl.id: [bbox_to_dict(b) for b in tl.get_light_boxes()]
                       for tl in world.get_actors().filter("traffic.traffic_light")}
        self._model = load_yolo(weights)
        self._names = self._model.names
        self._conf, self._imgsz = conf, imgsz
        self._voter = voter or TemporalVoter()
        self._current_tl: int | None = None
        self.debug = ModelDebug()

    def perceive(self, image: carla.Image) -> PerceptionOutput:
        import time

        bgr = np.ascontiguousarray(image_to_rgb(image)[:, :, ::-1])  # ultralytics는 numpy 입력을 BGR로 본다
        t0 = time.perf_counter()
        r = self._model.predict(bgr, conf=self._conf, imgsz=self._imgsz, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        dets = [Detection(self._names[int(c)], tuple(b), float(s))
                for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]

        nearest = self._map.nearest_light()
        if nearest is None:
            self._voter.reset()
            self._current_tl = None
            self.debug = ModelDebug(detections=dets, infer_ms=infer_ms)
            return PerceptionOutput()
        tl, dist = nearest
        if tl.id != self._current_tl:   # 다음 신호등으로 넘어가면 이전 신호의 투표를 버린다
            self._voter.reset()
            self._current_tl = tl.id
        expected = expected_head_boxes(self._heads[tl.id], transform_to_dict(self._camera.get_transform()),
                                       self._K, self._W, self._H)
        assoc = associate(expected, dets)
        raw = assoc.state if assoc else None
        voted = self._voter.update(raw)
        self.debug = ModelDebug(dets, expected, assoc, raw, voted, tl.id, infer_ms)
        return PerceptionOutput(TLState(voted), dist)
