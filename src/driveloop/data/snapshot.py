"""프레임 시점의 시뮬레이터 정답(raw ground truth)을 JSON 직렬화 가능한 dict로 추출.

여기서는 2D 박스를 만들지 않는다. 3D 정보 그대로 저장하고 2D 투영·가림 판정은
라벨링 단계(2-2)에서 한다 → 라벨 규칙을 바꿔도 재수집이 필요 없다.
"""
from __future__ import annotations

import math

import carla

_TL_STATE = {
    carla.TrafficLightState.Red: "red",
    carla.TrafficLightState.Yellow: "yellow",
    carla.TrafficLightState.Green: "green",
    carla.TrafficLightState.Off: "off",
}


def _r(v: float) -> float:
    return round(float(v), 3)


def transform_to_dict(tf: carla.Transform) -> dict:
    l, r = tf.location, tf.rotation
    return {"x": _r(l.x), "y": _r(l.y), "z": _r(l.z),
            "pitch": _r(r.pitch), "yaw": _r(r.yaw), "roll": _r(r.roll)}


def bbox_to_dict(bb: carla.BoundingBox) -> dict:
    return {"x": _r(bb.location.x), "y": _r(bb.location.y), "z": _r(bb.location.z),
            "ex": _r(bb.extent.x), "ey": _r(bb.extent.y), "ez": _r(bb.extent.z),
            "pitch": _r(bb.rotation.pitch), "yaw": _r(bb.rotation.yaw), "roll": _r(bb.rotation.roll)}


def _speed(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return _r(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


def traffic_light_catalog(world: carla.World) -> dict[str, dict]:
    """에피소드 동안 변하지 않는 신호등 정보 (헤드 박스는 월드 좌표)."""
    catalog = {}
    for tl in world.get_actors().filter("traffic.traffic_light"):
        catalog[str(tl.id)] = {
            "pole": transform_to_dict(tl.get_transform()),
            "light_boxes": [bbox_to_dict(b) for b in tl.get_light_boxes()],
        }
    return catalog


def camera_intrinsics(width: int, height: int, fov_deg: float) -> list[list[float]]:
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]


def frame_snapshot(world: carla.World, ego: carla.Vehicle, camera: carla.Sensor,
                   traffic_lights: list[carla.TrafficLight], radius: float) -> dict:
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location

    vehicles = []
    for v in world.get_actors().filter("vehicle.*"):
        if v.id == ego.id:
            continue
        tf = v.get_transform()
        dist = tf.location.distance(ego_loc)
        if dist > radius:
            continue
        vehicles.append({
            "id": v.id,
            "type_id": v.type_id,
            "base_type": v.attributes.get("base_type", "").lower(),
            "transform": transform_to_dict(tf),
            "bbox": bbox_to_dict(v.bounding_box),  # 차량 로컬 좌표
            "speed": _speed(v),
            "distance": _r(dist),
        })

    lights = {}
    for tl in traffic_lights:
        if tl.get_location().distance(ego_loc) <= radius:
            lights[str(tl.id)] = _TL_STATE.get(tl.get_state(), "unknown")

    affecting = ego.get_traffic_light()
    return {
        "ego": {
            "id": ego.id,  # 인스턴스 세그에서 자차(보닛) 픽셀을 제외하는 데 사용
            "transform": transform_to_dict(ego_tf),
            "speed": _speed(ego),
            "at_traffic_light": ego.is_at_traffic_light(),
            "affecting_light": str(affecting.id) if affecting is not None else None,
        },
        "camera": transform_to_dict(camera.get_transform()),
        "vehicles": vehicles,
        "traffic_lights": lights,
    }
