"""내 모델(YOLO)로 하는 인지 — 1단계 정답값 인지(GroundTruthPerception)를 대체한다.

위치는 지도, 색은 모델:
  1) 지도로 내 차선 신호등과 정지선 거리를 찾는다 (GroundTruthPerception.nearest_light, 상태는 읽지 않음)
  2) 그 신호등 헤드를 전방 카메라 화면에 투영 → 예상 위치
  3) 예상 위치 근처의 모델 검출을 내 신호로 고르고 (association.associate)
  4) 최근 몇 프레임을 다수결로 안정화 (TemporalVoter, RED 우선)

앞차 (B단계, 지도 없이 카메라만):
  5) 차량 검출(신뢰도 ≥ lead_min_conf)의 박스 아래 가운데 점 → 바닥선 거리(mono_distance) + 좌우 위치 → 월드 좌표
  6) 정답값과 같은 find_lead로 내 경로 위 가장 가까운 차 → LeadTracker로 확정(2프레임)·속도 추정

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
from driveloop.perception.lead import LeadTracker, Obstacle, find_lead
from driveloop.perception.mono_distance import ground_distance, ray_road_hit
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
    lead_det: Detection | None = None      # 앞차로 고른 차량 검출 (이번 프레임 측정)
    lead_measured: float | None = None     # 그 검출의 바닥선 거리 → 범퍼 간격 (추적 전)


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
                 conf: float = 0.10, imgsz: int = 1280, voter: TemporalVoter | None = None, *,
                 lead_lookahead: float = 50.0, lane_half_width: float = 1.75, cam_height: float = 1.556,
                 lead_min_conf: float = 0.3, dt: float = 0.05, road_aware: bool = True,
                 ground_offset: float = 0.144) -> None:
        self._map = GroundTruthPerception(world, ego, route, lookahead)
        self._ego, self._route = ego, route
        self._front_offset = ego.bounding_box.extent.x
        self._lead_lookahead, self._lane_half_width = lead_lookahead, lane_half_width
        self._cam_height, self._lead_min_conf = cam_height, lead_min_conf
        self._road_aware, self._ground_offset = road_aware, ground_offset
        self._tracker = LeadTracker(dt)
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

        # 앞차: 차량 검출 → 바닥선 거리 → 경로 위 가장 가까운 차 → 추적(확정·속도)
        lead_det, measured = self._measure_lead(dets)
        v = self._ego.get_velocity()
        lead = self._tracker.update(measured, (v.x ** 2 + v.y ** 2) ** 0.5)
        lead_kw = dict(lead_distance=lead.distance, lead_speed=lead.speed, lead_id=lead.id) if lead else {}

        nearest = self._map.nearest_light()
        if nearest is None:
            self._voter.reset()
            self._current_tl = None
            self.debug = ModelDebug(detections=dets, infer_ms=infer_ms, lead_det=lead_det, lead_measured=measured)
            return PerceptionOutput(**lead_kw)
        tl, dist = nearest
        if tl.id != self._current_tl:   # 다음 신호등으로 넘어가면 이전 신호의 투표를 버린다
            self._voter.reset()
            self._current_tl = tl.id
        expected = expected_head_boxes(self._heads[tl.id], transform_to_dict(self._camera.get_transform()),
                                       self._K, self._W, self._H)
        assoc = associate(expected, dets)
        raw = assoc.state if assoc else None
        voted = self._voter.update(raw)
        self.debug = ModelDebug(dets, expected, assoc, raw, voted, tl.id, infer_ms, lead_det, measured)
        return PerceptionOutput(TLState(voted), dist, **lead_kw)

    def _measure_lead(self, dets: list[Detection]) -> tuple[Detection | None, float | None]:
        """차량 박스 아래 가운데 점 → 월드 좌표 → 경로 위 가장 가까운 차의 범퍼 간격.

        road_aware(기본, B-7b): 카메라 광선이 '지도의 도로 높이'(내 경로 waypoint z)와 만나는 점 → 경사 보정
          카메라 자세(위치·pitch 포함)는 시뮬레이터 값 = 완벽한 측위 + IMU 가정
        아니면: 평평한 도로 가정의 바닥선 거리 (B-4, 오르막에서 멀게 봄 → acc_scen_v3 정차 차량을 20m에서야 인식)
        """
        fx, fy, cx, cy = self._K[0, 0], self._K[1, 1], self._K[0, 2], self._K[1, 2]
        tf = self._camera.get_transform()
        fwd, right, up = tf.get_forward_vector(), tf.get_right_vector(), tf.get_up_vector()
        road = [(w.transform.location.x, w.transform.location.y, w.transform.location.z)
                for w in self._route.waypoints] if self._road_aware else None
        cands, obstacles = [], []
        for d in dets:
            if d.cls != "vehicle" or d.conf < self._lead_min_conf:
                continue
            x1, _, x2, y2 = d.box
            u, v = (x1 + x2) / 2, min(y2, self._H)
            if road is not None:
                a, b = (u - cx) / fx, -(v - cy) / fy               # 카메라 좌표: x 앞, y 오른쪽, z 위
                ray = (fwd.x + right.x * a + up.x * b, fwd.y + right.y * a + up.y * b, fwd.z + right.z * a + up.z * b)
                hit = ray_road_hit((tf.location.x, tf.location.y, tf.location.z), ray, road,
                                   max_dist=self._lead_lookahead + 15.0, ground_offset=self._ground_offset)
                if hit is None:
                    continue
                wx, wy = hit[0], hit[1]
            else:
                z = ground_distance(v, fy, cy, self._cam_height)
                if z is None:
                    continue
                x = (u - cx) * z / fx
                wx = tf.location.x + fwd.x * z + right.x * x
                wy = tf.location.y + fwd.y * z + right.y * x
            obstacles.append(Obstacle(len(cands), wx, wy, half_length=0.0))   # 점 = 차 뒷면 → 길이 보정 없음
            cands.append(d)
        if not obstacles:
            return None, None
        loc = self._ego.get_location()
        lead = find_lead(self._route.points_xy(), (loc.x, loc.y), self._front_offset, obstacles,
                         self._lane_half_width, self._lead_lookahead)
        if lead is None:
            return None, None
        return cands[lead.id], max(0.0, lead.distance)
