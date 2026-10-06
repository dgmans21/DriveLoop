"""프레임/객체 메타데이터와 품질 검사 (CARLA 비의존).

frames 표 : 프레임 1행 — 조건 태그, 자차 상태, 클래스별 객체 수, 이미지 통계, dHash, QC 플래그
objects 표: 라벨 1행 — 박스 기하, 거리, 신호등 불빛 색 검증 결과, QC 플래그
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from driveloop.config import load_config


@dataclass
class ImageQC:
    min_mean_luma: float = 4.0
    min_std_luma: float = 3.0


@dataclass
class VehicleQC:
    max_box_area_ratio: float = 0.8


@dataclass
class TrafficLightQC:
    min_aspect: float = 1.2
    max_aspect: float = 6.0


@dataclass
class QCRules:
    version: int = 1
    image: ImageQC = field(default_factory=ImageQC)
    vehicle: VehicleQC = field(default_factory=VehicleQC)
    traffic_light: TrafficLightQC = field(default_factory=TrafficLightQC)


def load_qc_rules(path) -> QCRules:
    return load_config(QCRules, path)


# ---------- 이미지 통계 / 해시 ----------

def luma_stats(img: Image.Image) -> tuple[float, float]:
    g = np.asarray(img.convert("L").resize((320, 180)), dtype=np.float32)
    return float(g.mean()), float(g.std())


def dhash(img: Image.Image, size: int = 8) -> str:
    """difference hash (64bit, hex). 가까운 장면일수록 해밍 거리가 작다 → 2-2c 중복 제거에 사용."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ---------- 신호등 불빛 색 검증 ----------

_HUE = {  # degree
    "tl_red": lambda h: (h < 25) | (h > 320),
    "tl_yellow": lambda h: (h >= 30) & (h <= 75),
    "tl_green": lambda h: (h >= 85) & (h <= 200),
}
MIN_LIT_PX = 3
LIT_MIN_SAT = 0.25        # 비·안개로 색이 바랜 불빛도 잡도록 낮게
LIT_ABOVE_MEDIAN = 0.12   # 헤드 몸체(박스 중앙값)보다 이만큼 밝아야 불빛


def lamp_color_status(rgb_crop: Image.Image, cls: str) -> str:
    """라벨 색과 실제 켜진 불빛 색 비교.

    불빛 픽셀 = 박스 밝기 중앙값보다 확실히 밝고, 어느 정도 채도가 있는 픽셀 (상대 기준).
    v1 실측에서 절대 기준(채도·밝기 > 0.45)은 비·안개로 바랜 불빛을 놓쳤다.

    match     : 라벨 색 불빛이 보임
    mismatch  : 다른 색 불빛만 보임 (라벨 오류 의심 — 가장 중요한 플래그)
    saturated : 하얗게 번진 불빛만 보임 (색 판단 불가)
    dark      : 켜진 불빛이 안 보임 (멀거나, 너무 작거나, 옆면)
    """
    hsv = np.asarray(rgb_crop.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    lit = (s > LIT_MIN_SAT) & (v > max(0.3, float(np.median(v)) + LIT_ABOVE_MEDIAN))
    counts = {c: int((lit & f(h)).sum()) for c, f in _HUE.items()}
    if counts[cls] >= MIN_LIT_PX:
        return "match"
    if any(n >= MIN_LIT_PX for c, n in counts.items() if c != cls):
        return "mismatch"
    if int(((v > 0.9) & (s < 0.3)).sum()) >= MIN_LIT_PX:
        return "saturated"
    return "dark"


# ---------- 객체 / 프레임 행 ----------

def object_row(obj: dict, img: Image.Image, width: int, height: int, qc: QCRules) -> dict:
    x1, y1, x2, y2 = obj["bbox"]
    w, h = x2 - x1, y2 - y1
    flags = []
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        flags.append("bbox_out_of_image")
    row = {
        "cls": obj["cls"], "actor_id": obj["actor_id"],
        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "box_w": w, "box_h": h,
        "depth": obj["depth"], "visible_px": obj["visible_px"], "visible_ratio": obj["visible_ratio"],
        "facing_angle": obj.get("facing_angle"), "affects_ego": obj.get("affects_ego"),
        "base_type": obj.get("base_type"), "static": obj.get("static"), "lamp_color": None,
    }
    if obj["cls"] == "vehicle":
        if w * h > qc.vehicle.max_box_area_ratio * width * height:
            flags.append("vehicle_box_too_large")
    else:
        aspect = h / max(w, 1e-6)
        if not qc.traffic_light.min_aspect <= aspect <= qc.traffic_light.max_aspect:
            flags.append("tl_aspect_odd")
        row["lamp_color"] = lamp_color_status(img.crop((x1, y1, x2, y2)), obj["cls"])
        if row["lamp_color"] == "mismatch":
            flags.append("tl_color_mismatch")
    row["qc_flags"] = ",".join(flags)
    return row


CLASSES = ("vehicle", "tl_red", "tl_yellow", "tl_green")


def frame_row(frame: dict, labels: dict, episode: dict, img: Image.Image, objects: list[dict],
              qc: QCRules) -> dict:
    tags = episode["tags"]
    ego = frame["ego"]
    mean_l, std_l = luma_stats(img)
    flags = []
    if (img.width, img.height) != (episode["camera"]["width"], episode["camera"]["height"]):
        flags.append("wrong_image_size")
    if mean_l < qc.image.min_mean_luma or std_l < qc.image.min_std_luma:
        flags.append("blank_image")
    flags += sorted({f for o in objects for f in o["qc_flags"].split(",") if f})

    counts = {c: sum(o["cls"] == c for o in objects) for c in CLASSES}
    tl = [o for o in objects if o["cls"] != "vehicle"]
    veh = [o for o in objects if o["cls"] == "vehicle"]
    ego_tl = [o["cls"] for o in tl if o["affects_ego"]]
    affecting = ego["affecting_light"]
    return {
        "dataset": episode["dataset"], "episode_id": episode["episode_id"],
        "index": frame["index"], "frame": frame["frame"], "t": frame["t"],
        "rgb": frame["rgb"], "seg": frame["seg"],
        "map": tags["map"], "weather": tags["weather"], "time": tags["time"],
        "traffic": tags["traffic"], "repeat": tags["repeat"],
        "ego_speed": ego["speed"], "ego_stopped": ego["speed"] < 0.1,
        "ego_x": ego["transform"]["x"], "ego_y": ego["transform"]["y"], "ego_yaw": ego["transform"]["yaw"],
        "ego_light_state": frame["traffic_lights"].get(affecting) if affecting else None,
        "ego_light_visible": ego_tl[0] if ego_tl else None,
        **{f"n_{c}": counts[c] for c in CLASSES},
        "n_tl": len(tl),
        "n_vehicle_static": sum(bool(o["static"]) for o in veh),
        "n_vehicle_near": sum(o["depth"] is not None and o["depth"] < 30 for o in veh),
        "min_vehicle_depth": min((o["depth"] for o in veh if o["depth"] is not None), default=None),
        "min_tl_box_h": min((o["box_h"] for o in tl), default=None),
        "luma_mean": round(mean_l, 2), "luma_std": round(std_l, 2),
        "dhash": dhash(img),
        "qc_flags": ",".join(flags), "qc_pass": not flags,
    }
