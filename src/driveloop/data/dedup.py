"""에피소드 내 연속 중복 프레임 제거 (CARLA 비의존).

시간 순으로 훑으면서 '마지막으로 남긴 대표 프레임'과 비교한다 (직전 프레임과 비교하면
조금씩 변하는 장면이 연쇄적으로 전부 중복 처리되는 문제가 생긴다).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from driveloop.config import load_config
from driveloop.data.metadata import hamming

LABEL_KEYS = ("n_vehicle", "n_tl_red", "n_tl_yellow", "n_tl_green")


@dataclass
class DedupRules:
    version: int = 1
    max_move_m: float = 0.5
    max_hamming: int = 4
    require_same_labels: bool = True
    max_gap_s: float = 10.0


def load_dedup_rules(path) -> DedupRules:
    return load_config(DedupRules, path)


def dedup_episode(frames: list[dict], rules: DedupRules) -> list[dict]:
    """frames: 한 에피소드의 프레임 (index 순). 각 프레임에 대해 keep / dup_of / 판단 근거를 돌려준다."""
    out = []
    rep = None
    for f in frames:
        if rep is None:
            out.append({"index": f["index"], "keep": True, "dup_of": None, "reason": "first"})
            rep = f
            continue
        move = math.hypot(f["ego_x"] - rep["ego_x"], f["ego_y"] - rep["ego_y"])
        ham = hamming(f["dhash"], rep["dhash"])
        same_labels = all(f[k] == rep[k] for k in LABEL_KEYS)
        gap = f["t"] - rep["t"]
        if move >= rules.max_move_m:
            reason = "moved"
        elif ham > rules.max_hamming:
            reason = "image_changed"
        elif rules.require_same_labels and not same_labels:
            reason = "labels_changed"
        elif gap >= rules.max_gap_s:
            reason = "max_gap"
        else:
            out.append({"index": f["index"], "keep": False, "dup_of": rep["index"], "reason": "duplicate",
                        "hamming": ham, "move_m": round(move, 3)})
            continue
        out.append({"index": f["index"], "keep": True, "dup_of": None, "reason": reason,
                    "hamming": ham, "move_m": round(move, 3)})
        rep = f
    return out
