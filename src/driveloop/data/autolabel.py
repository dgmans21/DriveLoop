"""시뮬레이터 정답 + 인스턴스 세그멘테이션 → 2D 박스 라벨 (CARLA 비의존).

인스턴스 세그 PNG: R = semantic tag, G + 256·B = actor id (차량·신호등 모두 actor id와 일치함을 데이터로 확인).

차량  : 세그에 보이는 모든 차량 인스턴스(자차 제외)의 외접 박스. 수집 기록의 차량 목록이 아니라
        세그를 기준으로 한다 → 맵 배경 소품인 주차 차량, 자차 바로 옆(3D 박스 투영 불가) 차량도 포함.
        (v2까지는 기록 목록 기준이라 v1에서 약 1,000대 누락 — 규칙 v3에서 수정)

무시 영역(ignores): 보이지만 라벨 기준(크기·가림·거리)을 못 넘은 물체. 내보낼 때 회색으로 가려서
모델이 '보이는 차/신호등 = 배경'으로 잘못 배우지 않게 한다 (규칙 v4).
신호등: 헤드 3D 박스를 투영한 영역 안에서 이 신호등 ID 픽셀들의 외접 박스.
        헤드 정면(+y축)이 카메라를 향하지 않으면 색이 안 보이므로 제외.
보행자: 세그 태그 12(Pedestrian) 인스턴스의 외접 박스 (v4 데이터로 actor id 일치 확인). 사람은 가늘어서
        차량보다 작은 픽셀 기준. 오토바이·자전거 운전자(태그 13 Rider)는 탈것과 같은 인스턴스 id
        → 차량 박스에 포함 (규칙 v5. v4까지는 운전자가 빠진 오토바이 박스)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from driveloop.config import load_config
from driveloop.data.geometry import box_corners, box_to_2d, rotation_matrix, transform_matrix

TL_TAG = 7
PEDESTRIAN_TAG = 12
VEHICLE_TAGS = (13, 14, 15, 16, 18, 19)   # Rider, Car, Truck, Bus, Motorcycle, Bicycle
TL_CLASS = {"red": "tl_red", "yellow": "tl_yellow", "green": "tl_green"}


@dataclass
class VehicleRules:
    min_visible_px: int = 150
    min_box_height: float = 12
    max_distance: float = 80.0
    ignore_min_px: int = 20


@dataclass
class TrafficLightRules:
    min_visible_px: int = 20
    min_visible_ratio: float = 0.3
    min_box_height: float = 8
    max_facing_angle: float = 70.0
    max_distance: float = 80.0
    ignore_min_px: int = 5


@dataclass
class PedestrianRules:
    min_visible_px: int = 60
    min_box_height: float = 16
    max_distance: float = 60.0
    ignore_min_px: int = 10


@dataclass
class LabelRules:
    version: int = 5
    classes: list[str] = field(default_factory=lambda: ["vehicle", "tl_red", "tl_yellow", "tl_green", "pedestrian"])
    vehicle: VehicleRules = field(default_factory=VehicleRules)
    traffic_light: TrafficLightRules = field(default_factory=TrafficLightRules)
    pedestrian: PedestrianRules = field(default_factory=PedestrianRules)


def load_rules(path) -> LabelRules:
    return load_config(LabelRules, path)


def decode_instance(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seg = seg.astype(np.int32)
    return seg[..., 0], seg[..., 1] + 256 * seg[..., 2]


def _tight_box(mask: np.ndarray, x0: int = 0, y0: int = 0) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return (float(xs.min() + x0), float(ys.min() + y0), float(xs.max() + 1 + x0), float(ys.max() + 1 + y0))


def facing_angle(head_box: dict, camera_tf: dict) -> float:
    """헤드 정면(+y축)과 '헤드 → 카메라' 방향 사이 각 (deg, 수평면)."""
    front = rotation_matrix(head_box["pitch"], head_box["yaw"], head_box["roll"])[:, 1]
    to_cam = np.array([camera_tf["x"] - head_box["x"], camera_tf["y"] - head_box["y"]])
    f2 = front[:2]
    denom = np.linalg.norm(f2) * np.linalg.norm(to_cam)
    if denom < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(f2 @ to_cam / denom, -1.0, 1.0))))


def find_ego_instance(segs: list[np.ndarray]) -> int | None:
    """자차 ID가 기록되지 않은 데이터(v1)용: 화면 맨 아래 중앙(보닛)에 가장 자주 나오는 차량 인스턴스."""
    counts: dict[int, int] = {}
    for seg in segs:
        tag, inst = decode_instance(seg)
        h, w = tag.shape
        region = (slice(h - 6, h), slice(w * 7 // 16, w * 9 // 16))
        ids = inst[region][np.isin(tag[region], VEHICLE_TAGS)]
        for i in ids.tolist():
            counts[i] = counts.get(i, 0) + 1
    return max(counts, key=counts.get) if counts else None


def label_frame(frame: dict, episode: dict, seg: np.ndarray, rules: LabelRules,
                ego_id: int | None = None) -> list[dict]:
    return label_frame_full(frame, episode, seg, rules, ego_id)[0]


def _ignore(box, cls: str, reason: str, actor_id: int) -> dict:
    return {"cls": cls, "bbox": [round(c, 1) for c in box], "reason": reason, "actor_id": actor_id}


def label_frame_full(frame: dict, episode: dict, seg: np.ndarray, rules: LabelRules,
                     ego_id: int | None = None) -> tuple[list[dict], list[dict]]:
    """(objects, ignores)"""
    cam_cfg = episode["camera"]
    K = np.array(cam_cfg["K"])
    W, H = cam_cfg["width"], cam_cfg["height"]
    cam_tf = frame["camera"]
    tag, inst = decode_instance(seg)
    ego_id = frame["ego"].get("id", ego_id)
    objects, ignores = [], []

    vr = rules.vehicle
    actors = {v["id"]: v for v in frame["vehicles"]}
    vehicle_inst = np.where(np.isin(tag, VEHICLE_TAGS), inst, -1)
    ids, counts = np.unique(vehicle_inst[vehicle_inst >= 0], return_counts=True)
    for vid, visible in zip(ids.tolist(), counts.tolist()):
        if vid == ego_id or visible < vr.ignore_min_px:
            continue
        if visible < vr.min_visible_px:
            ignores.append(_ignore(_tight_box(vehicle_inst == vid), "vehicle", "too_few_px", vid))
            continue
        actor = actors.get(vid)
        depth, ratio = None, None
        if actor is not None:
            if actor["distance"] > vr.max_distance:
                ignores.append(_ignore(_tight_box(vehicle_inst == vid), "vehicle", "too_far", vid))
                continue
            m = transform_matrix(actor["transform"])
            proj = box_to_2d(box_corners(actor["bbox"]) @ m[:3, :3].T + m[:3, 3], cam_tf, K, W, H)
            if proj is not None:   # 자차 옆에 걸친 차량은 투영 불가 → 거리만 기록
                (px1, py1, px2, py2), depth = proj
                ratio = round(visible / max(1.0, (px2 - px1) * (py2 - py1)), 3)
            else:
                depth = actor["distance"]
        box = _tight_box(vehicle_inst == vid)
        if box[3] - box[1] < vr.min_box_height:
            ignores.append(_ignore(box, "vehicle", "too_small", vid))
            continue
        objects.append({
            "cls": "vehicle", "bbox": [round(c, 1) for c in box], "actor_id": vid,
            "visible_px": visible, "visible_ratio": ratio,
            "depth": None if depth is None else round(depth, 2),
            "base_type": actor["base_type"] if actor else None,
            "static": actor is None,   # 맵 배경 소품 (주차 차량)
        })

    pr = rules.pedestrian
    walkers = {w["id"]: w for w in frame.get("walkers", [])}     # v1~v3 기록에는 보행자 목록이 없다 (0명)
    ped_inst = np.where(tag == PEDESTRIAN_TAG, inst, -1)
    ids, counts = np.unique(ped_inst[ped_inst >= 0], return_counts=True)
    for pid, visible in zip(ids.tolist(), counts.tolist()):
        if visible < pr.ignore_min_px:
            continue
        box = _tight_box(ped_inst == pid)
        actor = walkers.get(pid)
        depth, ratio = None, None
        if actor is not None:
            m = transform_matrix(actor["transform"])
            proj = box_to_2d(box_corners(actor["bbox"]) @ m[:3, :3].T + m[:3, 3], cam_tf, K, W, H)
            if proj is not None:
                (px1, py1, px2, py2), depth = proj
                ratio = round(visible / max(1.0, (px2 - px1) * (py2 - py1)), 3)
            else:
                depth = actor["distance"]
        reason = ("too_few_px" if visible < pr.min_visible_px else
                  "too_far" if depth is not None and depth > pr.max_distance else
                  "too_small" if box[3] - box[1] < pr.min_box_height else None)
        if reason:
            ignores.append(_ignore(box, "pedestrian", reason, pid))
            continue
        objects.append({
            "cls": "pedestrian", "bbox": [round(c, 1) for c in box], "actor_id": pid,
            "visible_px": visible, "visible_ratio": ratio,
            "depth": None if depth is None else round(depth, 2),
            "static": actor is None,
        })

    tr = rules.traffic_light
    for tl_id, state in frame["traffic_lights"].items():
        cls = TL_CLASS.get(state)
        for head in episode["traffic_lights"][tl_id]["light_boxes"]:
            angle = facing_angle(head, cam_tf)
            if angle > tr.max_facing_angle:
                continue   # 옆·뒷면: 색이 안 보이므로 배경으로 둔다 (무시 영역 아님)
            proj = box_to_2d(box_corners(head), cam_tf, K, W, H)
            if proj is None:
                continue
            (px1, py1, px2, py2), depth = proj
            x1, y1, x2, y2 = int(px1), int(py1), int(math.ceil(px2)), int(math.ceil(py2))
            mask = (tag[y1:y2, x1:x2] == TL_TAG) & (inst[y1:y2, x1:x2] == int(tl_id))
            visible = int(mask.sum())
            if visible < tr.ignore_min_px:
                continue
            box = _tight_box(mask, x1, y1)
            ratio = visible / max(1.0, (px2 - px1) * (py2 - py1))
            dist = math.dist((head["x"], head["y"], head["z"]), (cam_tf["x"], cam_tf["y"], cam_tf["z"]))
            reason = ("state_unknown" if cls is None else
                      "too_far" if dist > tr.max_distance else
                      "occluded" if visible < tr.min_visible_px or ratio < tr.min_visible_ratio else
                      "too_small" if box[3] - box[1] < tr.min_box_height else None)
            if reason:
                ignores.append(_ignore(box, "traffic_light", reason, int(tl_id)))
                continue
            objects.append({
                "cls": cls, "bbox": [round(c, 1) for c in box], "actor_id": int(tl_id),
                "visible_px": visible, "visible_ratio": round(ratio, 3),
                "depth": round(depth, 2), "facing_angle": round(angle, 1),
                "affects_ego": tl_id == frame["ego"]["affecting_light"],
            })
    return objects, ignores
