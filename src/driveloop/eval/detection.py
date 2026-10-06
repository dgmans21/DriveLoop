"""검출 결과 ↔ 정답 매칭 (CARLA·YOLO 비의존).

YOLO 기본 지표(mAP)는 전체 평균이라, 조건별·크기별 재현율과 신호등 색 혼동을 보려면
예측과 정답을 직접 짝지어야 한다.
"""
from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 (px)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) × (M,4) → (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(gt_boxes: np.ndarray, pred_boxes: np.ndarray, pred_conf: np.ndarray,
                 iou_thr: float = 0.5) -> list[tuple[int, int]]:
    """신뢰도 높은 예측부터 IoU가 가장 큰 미매칭 정답과 짝짓는다. [(gt_idx, pred_idx)]"""
    ious = iou_matrix(gt_boxes, pred_boxes)
    used_gt: set[int] = set()
    pairs = []
    for p in np.argsort(-pred_conf):
        if ious.shape[0] == 0:
            break
        cand = [(ious[g, p], g) for g in range(len(gt_boxes)) if g not in used_gt and ious[g, p] >= iou_thr]
        if cand:
            _, g = max(cand)
            used_gt.add(g)
            pairs.append((g, int(p)))
    return pairs


def match_image(gt: list[dict], pred: list[dict], iou_thr: float = 0.5,
                class_agnostic_groups: dict[str, str] | None = None) -> dict:
    """한 이미지의 정답/예측 매칭.

    gt/pred: [{"cls": str, "box": Box, ("conf": float)}]
    class_agnostic_groups: 클래스 → 그룹 (예: tl_red/yellow/green → traffic_light).
      같은 그룹 안에서는 클래스가 달라도 위치로 짝짓고, 색 혼동으로 기록한다.

    반환: {"gt": [{..., "matched_cls": str|None}], "fp": [pred...]}
    """
    groups = class_agnostic_groups or {}
    group_of = lambda c: groups.get(c, c)  # noqa: E731
    out_gt = [dict(g, matched_cls=None) for g in gt]
    pred_used = [False] * len(pred)
    for grp in sorted({group_of(g["cls"]) for g in gt} | {group_of(p["cls"]) for p in pred}):
        gi = [i for i, g in enumerate(gt) if group_of(g["cls"]) == grp]
        pi = [i for i, p in enumerate(pred) if group_of(p["cls"]) == grp]
        if not gi or not pi:
            continue
        gb = np.array([gt[i]["box"] for i in gi], dtype=float)
        pb = np.array([pred[i]["box"] for i in pi], dtype=float)
        pc = np.array([pred[i]["conf"] for i in pi], dtype=float)
        for g, p in greedy_match(gb, pb, pc, iou_thr):
            out_gt[gi[g]]["matched_cls"] = pred[pi[p]]["cls"]
            pred_used[pi[p]] = True
    return {"gt": out_gt, "fp": [p for p, used in zip(pred, pred_used) if not used]}
