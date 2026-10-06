"""YOLO 형식 내보내기 유틸 (CARLA 비의존)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driveloop.config import load_config


@dataclass
class ExportConfig:
    name: str
    sources: list[str]                      # 합칠 원본 버전들 (data/processed/<source>)
    classes: list[str]
    val_episodes: list[str]
    test_maps: list[str] = field(default_factory=list)   # 이 맵은 전부 test (학습·검증에 안 씀)
    use_dedup: bool = True
    drop_frame_flags: list[str] = field(default_factory=lambda: ["blank_image", "wrong_image_size"])
    ignore_object_flags: list[str] = field(default_factory=lambda: ["tl_color_mismatch", "tl_aspect_odd"])
    ignore_lamp_color: list[str] = field(default_factory=lambda: ["dark", "saturated"])
    ignore_fill: list[int] = field(default_factory=lambda: [114, 114, 114])
    ignore_pad_px: int = 2
    jpeg_quality: int = 95


def load_export_config(path) -> ExportConfig:
    return load_config(ExportConfig, path)


def yolo_line(cls_id: int, box, width: int, height: int) -> str:
    """x1,y1,x2,y2(px) → 'cls cx cy w h' (0~1 정규화)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def apply_ignore_mask(img: np.ndarray, ignore_boxes, keep_boxes, fill, pad: int) -> np.ndarray:
    """무시 영역을 회색으로 칠한다. 단, 라벨 박스와 겹치는 픽셀은 원본 유지 (라벨 물체를 가리지 않게)."""
    if not ignore_boxes:
        return img
    h, w = img.shape[:2]
    mask = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in ignore_boxes:
        mask[max(0, int(y1) - pad):min(h, int(np.ceil(y2)) + pad),
             max(0, int(x1) - pad):min(w, int(np.ceil(x2)) + pad)] = True
    for x1, y1, x2, y2 in keep_boxes:
        mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = False
    out = img.copy()
    out[mask] = fill
    return out
